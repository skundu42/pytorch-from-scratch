"""Chapters 6-7: text -> character ids -> (input, target) training pairs."""
import torch
from torch.utils.data import Dataset, DataLoader


class CharTokenizer:
    """Every distinct character in the text becomes one token id."""

    def __init__(self, text):
        self.chars = sorted(set(text))                      # the vocabulary
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}  # char -> id
        self.itos = {i: ch for i, ch in enumerate(self.chars)}  # id -> char

    def encode(self, s):
        return [self.stoi[ch] for ch in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


class CharDataset(Dataset):
    """Item i is a window of block_size ids (x) and the same window shifted by one (y)."""

    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, i):
        chunk = self.data[i : i + self.block_size + 1]
        x = chunk[:-1]   # "First Citize"
        y = chunk[1:]    # "irst Citizen": the next character at every position
        return x, y


def build_datasets(path, block_size, val_frac=0.1):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    # split the TEXT, not the windows: overlapping windows would leak val into train
    n = int(len(data) * (1 - val_frac))
    return tok, CharDataset(data[:n], block_size), CharDataset(data[n:], block_size)


if __name__ == "__main__":
    tok, train_ds, val_ds = build_datasets("data/input.txt", block_size=8)
    print("characters:", len(train_ds.data) + len(val_ds.data), "| train:", len(train_ds.data), "| val:", len(val_ds.data))
    print("vocab size:", tok.vocab_size)
    print("vocab:", repr("".join(tok.chars)))
    ids = tok.encode("ROMEO:")
    print("encode('ROMEO:') ->", ids)
    print("decode(ids)      ->", repr(tok.decode(ids)))
    print("train windows:", len(train_ds), "| val windows:", len(val_ds))

    x, y = train_ds[0]
    print("x:", x.tolist(), repr(tok.decode(x.tolist())))
    print("y:", y.tolist(), repr(tok.decode(y.tolist())))
    for t in range(3):
        print(f"  context {tok.decode(x[: t + 1].tolist())!r} -> target {tok.itos[y[t].item()]!r}")

    g = torch.Generator().manual_seed(1337)
    loader = DataLoader(train_ds, batch_size=4, shuffle=True, generator=g)
    xb, yb = next(iter(loader))
    print("batch x shape:", tuple(xb.shape), xb.dtype, "| batch y shape:", tuple(yb.shape))
    print(xb)
