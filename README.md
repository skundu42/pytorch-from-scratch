# PyTorch From Scratch: Quill

Code for the Forward Pass video **"PyTorch for Beginners: Train a Model That Writes Shakespeare"**.

We learn PyTorch by building one real project: **Quill**, a 3.2 million parameter character-level Transformer
that learns to write like Shakespeare. It trains in about 9 minutes on a laptop GPU.

## Files

| File | What it covers in the video |
| --- | --- |
| `basics.py` | Devices, tensors, autograd, `torch.nn`, loss functions, SGD vs AdamW, `Dataset` and `DataLoader` |
| `data.py` | Tokenizing Shakespeare: characters to ids, (input, target) windows, train/validation split |
| `model.py` | Embeddings, positions, causal self-attention, the Transformer block, next-character prediction |
| `train.py` | Training Quill: AdamW, gradient clipping, train vs validation loss, `--amp` (bf16), `--compile`, saving |
| `generate.py` | Loading the checkpoint and generating text with temperature and top-k sampling |
| `data/input.txt` | Tiny Shakespeare (1.1 MB), from [karpathy/char-rnn](https://github.com/karpathy/char-rnn) |

## Run it

Tested with Python 3.12 and PyTorch 2.13 and 2.15 (`requirements.txt` asks for 2.13 or newer). For an NVIDIA GPU, install the CUDA build of
PyTorch from the [install selector](https://pytorch.org/get-started/locally/) first.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python basics.py       # the PyTorch basics, printed section by section
python data.py         # the tokenizer and the training windows
python train.py        # trains Quill and writes ckpt.pt + loss.csv
python generate.py --prompt "ROMEO:" --temperature 0.8 --top-k 20
```

The scripts pick the GPU automatically: CUDA, Apple Silicon (MPS), or the CPU if there is no GPU.

Useful options:

```bash
python train.py --amp              # mixed precision (bfloat16 autocast)
python train.py --compile          # torch.compile the model
python train.py --steps 1000       # a shorter run
python generate.py --tokens 500 --temperature 0.5
```

## What our run looked like

Apple M3 Pro (MPS), PyTorch 2.15.0, default settings (5,000 steps, batch 64, context 128, 4 layers, 4 heads, 256-dim):

- 3,225,600 parameters
- 530 s of training, peak GPU memory 761 MiB
- best validation loss 1.478 at step 4,500 (it starts at 4.25)
- bf16 autocast was about 20% faster per step

A sample from `generate.py --prompt "JULIET:" --temperature 0.8 --top-k 20`:

```
JULIET:
Howsoever, to come hither.

Nurse:
I will tell him hence; I'll myself, be gone.
```

Your numbers will differ with your hardware and PyTorch version.
