"""Chapters 8 and 10: train the tiny Transformer on Tiny Shakespeare, then save it."""
import argparse
import csv
import math
import time
from contextlib import nullcontext
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader

from data import build_datasets
from generate import generate
from model import GPT, GPTConfig

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/input.txt")
p.add_argument("--steps", type=int, default=5000)
p.add_argument("--batch-size", type=int, default=64)
p.add_argument("--block-size", type=int, default=128)
p.add_argument("--n-layer", type=int, default=4)
p.add_argument("--n-head", type=int, default=4)
p.add_argument("--n-embd", type=int, default=256)
p.add_argument("--dropout", type=float, default=0.1)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--accum", type=int, default=1)        # gradient accumulation micro-batches
p.add_argument("--eval-every", type=int, default=250)
p.add_argument("--eval-iters", type=int, default=40)
p.add_argument("--amp", action="store_true")          # mixed precision with autocast
p.add_argument("--compile", action="store_true")      # torch.compile the model
p.add_argument("--seed", type=int, default=1337)
p.add_argument("--out", default="ckpt.pt")
p.add_argument("--log", default="loss.csv")
args = p.parse_args()

torch.manual_seed(args.seed)
acc = torch.accelerator
device = acc.current_accelerator().type if acc.is_available() else "cpu"  # cuda, mps or cpu
print("device:", device)

# ---- data -------------------------------------------------------------------
tok, train_ds, val_ds = build_datasets(args.data, args.block_size)
g = torch.Generator().manual_seed(args.seed)
train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, generator=g, drop_last=True)
val_loader = DataLoader(val_ds, args.batch_size, shuffle=True, generator=g, drop_last=True)


def forever(loader):          # loop over the DataLoader endlessly, reshuffling each epoch
    while True:
        yield from loader


train_iter, val_iter = forever(train_loader), forever(val_loader)

# ---- model + optimizer --------------------------------------------------------
cfg = GPTConfig(tok.vocab_size, args.block_size, args.n_embd, args.n_head, args.n_layer, args.dropout)
model = GPT(cfg).to(device)
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)   # weight_decay=0.01 default

raw_model = model                               # the plain nn.Module, used for saving
if args.compile:
    model = torch.compile(model)
autocast = torch.amp.autocast(device_type=device, dtype=torch.bfloat16) if args.amp else nullcontext()


@torch.no_grad()
def estimate_loss():
    model.eval()                                # dropout off while measuring
    out = {}
    for split, it in (("train", train_iter), ("val", val_iter)):
        losses = torch.zeros(args.eval_iters)
        for i in range(args.eval_iters):
            x, y = next(it)
            with autocast:
                _, loss = model(x.to(device), y.to(device))
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def sample(n=200):
    raw_model.eval()
    prompt = torch.tensor([tok.encode("ROMEO:")], device=device)
    text = tok.decode(generate(raw_model, prompt, n)[0].tolist())
    raw_model.train()
    return text


print(f"ln(vocab_size) = ln({tok.vocab_size}) = {math.log(tok.vocab_size):.4f}")
print("untrained sample:\n" + sample() + "\n")

# ---- training loop --------------------------------------------------------------
log = open(args.log, "w", newline="")
writer = csv.writer(log)
writer.writerow(["step", "train_loss", "val_loss", "elapsed_s"])
t0 = time.time()
for step in range(args.steps + 1):
    if step % args.eval_every == 0 or step == args.steps:
        losses = estimate_loss()
        elapsed = time.time() - t0
        print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | {elapsed:6.1f}s")
        writer.writerow([step, f"{losses['train']:.4f}", f"{losses['val']:.4f}", f"{elapsed:.1f}"])
        log.flush()
    if step == args.steps:
        break

    for _ in range(args.accum):
        x, y = next(train_iter)
        x, y = x.to(device), y.to(device)       # batch to the GPU
        with autocast:
            _, loss = model(x, y)               # forward + cross-entropy
        (loss / args.accum).backward()          # backprop: grads add up in .grad
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()                            # update the weights
    optimizer.zero_grad(set_to_none=True)       # reset grads for the next step

log.close()
print(f"trained {args.steps} steps in {time.time() - t0:.1f}s")
if device != "cpu":
    print(f"peak {device} memory allocated: {acc.max_memory_allocated() / 2**20:.0f} MiB")
torch.save({"model": raw_model.state_dict(), "config": asdict(cfg), "chars": tok.chars}, args.out)
print("saved", args.out)
print("trained sample:\n" + sample())
