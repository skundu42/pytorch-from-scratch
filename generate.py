"""Chapters 9 and 11: load trained weights and write new text one character at a time."""
import argparse

import torch
import torch.nn.functional as F

from model import GPT, GPTConfig


@torch.inference_mode()   # no autograd bookkeeping: faster and lighter than no_grad
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.block_size:]         # crop to the context window
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature            # scores for the NEXT character
        if top_k is not None:                              # keep only the k best scores
            kth_best = torch.topk(logits, top_k).values[:, [-1]]
            logits = logits.masked_fill(logits < kth_best, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # sample, don't just take the max
        idx = torch.cat([idx, next_id], dim=1)             # append and repeat
    return idx


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt["chars"]   # eval(): dropout off


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="ckpt.pt")
    p.add_argument("--prompt", default="ROMEO:")
    p.add_argument("--tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    acc = torch.accelerator
    device = acc.current_accelerator().type if acc.is_available() else "cpu"
    torch.manual_seed(args.seed)
    model, chars = load_model(args.ckpt, device)
    stoi = {ch: i for i, ch in enumerate(chars)}
    idx = torch.tensor([[stoi[ch] for ch in args.prompt]], device=device)
    out = generate(model, idx, args.tokens, args.temperature, args.top_k)
    print("".join(chars[i] for i in out[0].tolist()))
