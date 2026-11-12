"""Chapters 2-6: devices, tensors, autograd, torch.nn and a first training loop."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

torch.manual_seed(0)


def section(title):
    print(f"\n=== {title} ===")


section("2. which device?")
print("torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", torch.backends.mps.is_available())
print("current accelerator:", torch.accelerator.current_accelerator())

section("3a. creating tensors")
t = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(t)
print("shape:", t.shape, "| ndim:", t.ndim, "| dtype:", t.dtype)
print("t.float().dtype:", t.float().dtype)
print("bytes per element: int64", t.element_size(), "| float32", t.float().element_size(),
      "| bfloat16", t.bfloat16().element_size())
print("zeros:", torch.zeros(2, 3))
print("arange:", torch.arange(6))
print("randn:", torch.randn(2, 3))

section("3b. indexing, slicing, reshaping")
print("t[0]     ->", t[0])
print("t[:, 1]  ->", t[:, 1])
print("t[1, 1:] ->", t[1, 1:])
print("t.view(3, 2) ->", t.view(3, 2))
print("t.T ->", t.T)
print("t.unsqueeze(0).shape ->", t.unsqueeze(0).shape)
print("t.flatten() ->", t.flatten())

section("3c. math and broadcasting")
a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print("a * 10 ->", a * 10)
print("a @ a.T ->", a @ a.T)                     # (2, 3) @ (3, 2) -> (2, 2)
print("a.sum(dim=0) ->", a.sum(dim=0), "| a.mean(dim=1) ->", a.mean(dim=1))
row = torch.tensor([10.0, 20.0, 30.0])
print("a + row: (2, 3) + (3,) ->", a + row)      # row is reused, not copied
tok = torch.zeros(2, 4, 3)                       # (batch, time, channels)
pos = torch.arange(12.0).view(4, 3)              # (time, channels)
print("tok + pos: (2, 4, 3) + (4, 3) ->", (tok + pos).shape)

section("3d. moving between CPU and GPU")
acc = torch.accelerator
device = acc.current_accelerator().type if acc.is_available() else "cpu"
g = a.to(device)
print("a.device:", a.device, "| g.device:", g.device)
try:
    a + g
except RuntimeError as e:
    print("a + g ->", str(e).splitlines()[0])
print("(g * 2).cpu() ->", (g * 2).cpu())

section("4a. autograd on y = w * x + b")
w = torch.tensor(2.0, requires_grad=True)       # parameters: track gradients
b = torch.tensor(1.0, requires_grad=True)
x = torch.tensor(3.0)                           # data: no gradient needed
y = w * x + b                                   # 2 * 3 + 1 = 7
loss = (y - 10) ** 2                            # target 10: (7 - 10)^2 = 9
print("y =", y.item(), "| loss =", loss.item())
fn, chain = loss.grad_fn, []
while fn is not None:                           # walk the graph autograd recorded
    chain.append(fn.name())
    fn = fn.next_functions[0][0] if fn.next_functions else None
print("graph:", " <- ".join(chain))
loss.backward()                                 # chain rule from loss back to w and b
print("dloss/dw =", w.grad.item(), "  # 2 * (y - 10) * x = 2 * -3 * 3")
print("dloss/db =", b.grad.item(), "   # 2 * (y - 10)     = 2 * -3")
print("w.is_leaf:", w.is_leaf, "| y.is_leaf:", y.is_leaf, "| y.requires_grad:", y.requires_grad)

section("4b. gradients accumulate until you zero them")
loss = (w * x + b - 10) ** 2
loss.backward()                                 # second backward, no zeroing
print("w.grad after a 2nd backward:", w.grad.item(), "  # -18 + -18")
opt = torch.optim.SGD([w, b], lr=0.01)
opt.zero_grad()                                 # set_to_none=True by default
print("after opt.zero_grad(): w.grad =", w.grad)
xs, ts = torch.tensor([3.0, 1.0]), torch.tensor([10.0, 2.0])
((w * xs + b - ts) ** 2).mean().backward()      # one batch of 2
print("grad from one batch of 2:     ", w.grad.item())
opt.zero_grad()
for xi, ti in zip(xs, ts):                      # two micro-batches of 1
    ((w * xi + b - ti) ** 2 / 2).backward()
print("grad accumulated over 2 x 1:  ", w.grad.item())

section("4c. one gradient-descent step")
opt.zero_grad()
loss = (w * x + b - 10) ** 2
loss.backward()
opt.step()                                      # w <- w - lr * w.grad
print("w:", round(w.item(), 4), "| b:", round(b.item(), 4))
print("new loss:", round(((w * x + b - 10) ** 2).item(), 4), "(was 9.0)")
with torch.no_grad():
    z = w * x
print("inside no_grad: z.requires_grad =", z.requires_grad)

section("5a. layers and activations")
lin = nn.Linear(3, 2)
print("Linear(3, 2): weight", tuple(lin.weight.shape), "bias", tuple(lin.bias.shape), "| lin(a).shape", lin(a).shape)
emb = nn.Embedding(65, 4)                       # 65 characters, 4 numbers each
ids = torch.tensor([30, 27, 25])                # 'R', 'O', 'M'
print("Embedding(65, 4)(ids).shape:", emb(ids).shape)
z = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
print("relu:", F.relu(z))
print("gelu:", F.gelu(z))
print("softmax:", F.softmax(z, dim=0))


class TinyNet(nn.Module):                        # a bigram model: one char -> next-char scores
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(65, 8)
        self.fc = nn.Linear(8, 65)

    def forward(self, ids):
        return self.fc(F.relu(self.emb(ids)))


section("5b. nn.Module, forward() and parameters")
net = TinyNet()
for name, p in net.named_parameters():
    print(f"  {name:10s} {tuple(p.shape)}")
print("total parameters:", sum(p.numel() for p in net.parameters()))
print("net(ids).shape:", net(ids).shape)

section("5c. loss functions")
logits, target = torch.tensor([[2.0, 1.0, 0.1]]), torch.tensor([0])
print("cross_entropy:", round(F.cross_entropy(logits, target).item(), 4),
      "= -log softmax[0] =", round(-F.log_softmax(logits, dim=-1)[0, 0].item(), 4))
print("mse_loss(7, 10):", F.mse_loss(torch.tensor([7.0]), torch.tensor([10.0])).item())

section("5d. SGD vs AdamW on loss = (w - 3)^2, lr = 0.1")
for name, Opt in (("SGD", torch.optim.SGD), ("AdamW", torch.optim.AdamW)):
    w = torch.tensor(0.0, requires_grad=True)
    opt, path = Opt([w], lr=0.1), []
    for _ in range(5):
        loss = (w - 3) ** 2
        opt.zero_grad()
        loss.backward()
        opt.step()
        path.append(round(w.item(), 4))
    print(f"  {name:6s} w after each step: {path}")

section("6. Dataset, DataLoader and a basic training loop: fit y = 2x + 1")
X = torch.linspace(-1, 1, 100).unsqueeze(1)                  # (100, 1)
Y = 2 * X + 1 + 0.1 * torch.randn_like(X)                     # noisy line
train_set, val_set = random_split(TensorDataset(X, Y), [80, 20])
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
xb, yb = next(iter(train_loader))
print("batches per epoch:", len(train_loader), "| one batch:", tuple(xb.shape), tuple(yb.shape))
xv, yv = next(iter(DataLoader(val_set, batch_size=20)))     # the whole val set, one batch
model = nn.Linear(1, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
for epoch in range(1, 31):
    for xb, yb in train_loader:
        loss = F.mse_loss(model(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    if epoch % 10 == 0:
        with torch.no_grad():
            val = F.mse_loss(model(xv), yv)
        print(f"  epoch {epoch:2d} | train loss {loss.item():.4f} | val loss {val.item():.4f}")
print("learned w:", round(model.weight.item(), 3), "| b:", round(model.bias.item(), 3), " (true 2, 1)")
