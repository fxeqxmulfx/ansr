import math

import torch
import torch.nn as nn

from ansr.ansr_torch import ANSR
from ansr.callbacks import TorchEarlyStopCallback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)


# --- Sphere (128D, scaled to [0,1] like Rust optimizers repo) ---
class Sphere(nn.Module):
    def __init__(self, dims: int):
        super().__init__()
        self.x = nn.Parameter(torch.randn(dims))
        self._max_val = dims / 2 * 50.0  # 64 pairs * max(x²+y²)=50

    def forward(self) -> torch.Tensor:
        return (self.x**2).sum() / self._max_val


def sphere_loss(output: torch.Tensor) -> torch.Tensor:
    return output


sphere_05 = torch.compile(Sphere(128).to(device))  # type: ignore[assignment]
opt_05 = ANSR(
    sphere_05.parameters(),
    model=sphere_05,
    batch_size=16,
    popsize=64,
    sigma=0.05,
    self_instead_neighbour=0.05,
    bound=5.0,
    seed=0,
)
cb_05 = TorchEarlyStopCallback(stop_residual=0.01)

for step in range(5000):
    loss = opt_05.step(sphere_loss)
    if cb_05(loss):
        break
print(
    f"sphere  ANSR  | step={step:5d} | f_calls={64*(step+1):7d} | loss={loss.item():.2e} | restarts={opt_05.restarts}"
)

sphere_95 = torch.compile(Sphere(128).to(device))  # type: ignore[assignment]
opt_95 = ANSR(
    sphere_95.parameters(),
    model=sphere_95,
    batch_size=16,
    popsize=64,
    sigma=0.05,
    self_instead_neighbour=0.95,
    bound=5.0,
    seed=0,
)
cb_95 = TorchEarlyStopCallback(stop_residual=0.01)

for step in range(5000):
    loss = opt_95.step(sphere_loss)
    if cb_95(loss):
        break
print(
    f"sphere  ANSR  | step={step:5d} | f_calls={64*(step+1):7d} | loss={loss.item():.2e} | restarts={opt_95.restarts} | p_self=0.95"
)

sphere_adam = torch.compile(Sphere(128).to(device))  # type: ignore[assignment]
opt_adam = torch.optim.AdamW(sphere_adam.parameters(), lr=0.01)
for step in range(5000):
    opt_adam.zero_grad()
    loss = sphere_adam()
    loss.backward()
    opt_adam.step()
    if loss.item() <= 0.01:
        break
print(f"sphere  AdamW | step={step:5d} | f_calls={step+1:7d} | loss={loss.item():.2e}")


# --- Shubert (64D, scaled to [0,1] like Rust optimizers repo) ---
class Shubert(nn.Module):
    def __init__(self, dims: int):
        super().__init__()
        self.x = nn.Parameter(torch.zeros(dims))
        self._n_pairs = dims // 2

    def forward(self) -> torch.Tensor:
        i = torch.arange(1, 6, dtype=self.x.dtype, device=self.x.device)
        x_r = self.x.view(-1, 1)
        dims = self.x.shape[0]
        idx0 = torch.arange(dims, device=self.x.device) % 2 == 0
        idx1 = ~idx0
        v0 = (i * torch.cos((i + 1) * x_r[idx0] + i)).sum(dim=1)
        v1 = (i * torch.cos((i + 1) * x_r[idx1] + i)).sum(dim=1)
        # scale each pair to [0,1] like Rust: scale(v0*v1, -186.7309, 210.0, 0, 1)
        pair_vals = (v0 * v1 - (-186.7309)) / (210.0 - (-186.7309))
        return pair_vals.mean()


shubert = torch.compile(Shubert(64).to(device))  # type: ignore[assignment]
opt2 = ANSR(
    shubert.parameters(),
    model=shubert,
    batch_size=16,
    popsize=64,
    sigma=0.05,
    self_instead_neighbour=0.05,
    bound=10.0,
    seed=42,
)
cb2 = TorchEarlyStopCallback(stop_residual=0.01)


def shubert_loss(output: torch.Tensor) -> torch.Tensor:
    return output


for step in range(5000):
    loss = opt2.step(shubert_loss)
    if cb2(loss):
        break
print(
    f"shubert ANSR  | step={step:5d} | f_calls={64*(step+1):7d} | loss={loss.item():.2e} | restarts={opt2.restarts}"
)

shubert_adam = torch.compile(Shubert(64).to(device))  # type: ignore[assignment]
opt2_adam = torch.optim.AdamW(shubert_adam.parameters(), lr=0.01)
for step in range(100_000):
    opt2_adam.zero_grad()
    loss = shubert_adam()
    loss.backward()
    opt2_adam.step()
    if loss.item() <= 0.01:
        break
print(f"shubert AdamW | step={step:5d} | f_calls={step+1:7d} | loss={loss.item():.2e}")


# --- Transformer (copy task, 796 params) ---
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(16, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=d_model * 2,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=1, enable_nested_tensor=False
        )
        self.head = nn.Linear(d_model, vocab_size)
        self.scale = math.sqrt(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device)
        x = self.embed(x) * self.scale + self.pos(positions)
        return self.head(self.encoder(x))


torch.manual_seed(0)
vocab_size = 4
n_samples = 64
seq_len = 3
all_src = torch.randint(0, vocab_size, (n_samples, seq_len), device=device)
all_tgt = all_src.clone()

train_src, test_src = all_src[:48], all_src[48:]
train_tgt, test_tgt = all_tgt[:48], all_tgt[48:]

model = torch.compile(
    TinyTransformer(vocab_size=vocab_size, d_model=8, nhead=1).to(device)
)  # type: ignore[assignment]
opt3 = ANSR(
    model.parameters(),
    model=model,
    batch_size=16,
    popsize=64,
    sigma=0.05,
    self_instead_neighbour=0.05,
    bound=20.0,
    seed=0,
)
cb3 = TorchEarlyStopCallback(stop_residual=0.1)


def loss_fn(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits.view(-1, vocab_size), train_tgt.view(-1))


for step in range(5000):
    loss = opt3.step(loss_fn, train_src)
    if cb3(loss):
        break
with torch.no_grad():
    train_acc = (model(train_src).argmax(-1) == train_tgt).float().mean()
    test_logits = model(test_src)
    test_loss = nn.functional.cross_entropy(test_logits.view(-1, vocab_size), test_tgt.view(-1))
    test_acc = (test_logits.argmax(-1) == test_tgt).float().mean()
print(
    f"transf  ANSR  | step={step:5d} | f_calls={64*(step+1):7d} | train_loss={loss.item():.2e} | train_acc={train_acc:.2%}"
    f" | test_loss={test_loss.item():.2e} | test_acc={test_acc:.2%} | restarts={opt3.restarts}"
)

torch.manual_seed(0)
model_95 = torch.compile(
    TinyTransformer(vocab_size=vocab_size, d_model=8, nhead=1).to(device)
)  # type: ignore[assignment]
opt3_95 = ANSR(
    model_95.parameters(),
    model=model_95,
    batch_size=16,
    popsize=64,
    sigma=0.05,
    self_instead_neighbour=0.95,
    bound=20.0,
    seed=0,
)
cb3_95 = TorchEarlyStopCallback(stop_residual=0.1)

for step in range(5000):
    loss = opt3_95.step(loss_fn, train_src)
    if cb3_95(loss):
        break
with torch.no_grad():
    train_acc = (model_95(train_src).argmax(-1) == train_tgt).float().mean()
    test_logits = model_95(test_src)
    test_loss = nn.functional.cross_entropy(test_logits.view(-1, vocab_size), test_tgt.view(-1))
    test_acc = (test_logits.argmax(-1) == test_tgt).float().mean()
print(
    f"transf  ANSR  | step={step:5d} | f_calls={64*(step+1):7d} | train_loss={loss.item():.2e} | train_acc={train_acc:.2%}"
    f" | test_loss={test_loss.item():.2e} | test_acc={test_acc:.2%} | restarts={opt3_95.restarts} | p_self=0.95"
)

torch.manual_seed(0)
model_adam = torch.compile(
    TinyTransformer(vocab_size=vocab_size, d_model=8, nhead=1).to(device)
)  # type: ignore[assignment]
opt3_adam = torch.optim.AdamW(model_adam.parameters(), lr=0.01)
for step in range(5000):
    opt3_adam.zero_grad()
    logits = model_adam(train_src)
    loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), train_tgt.view(-1))
    loss.backward()
    opt3_adam.step()
    if loss.item() <= 0.1:
        break
with torch.no_grad():
    train_acc = (model_adam(train_src).argmax(-1) == train_tgt).float().mean()
    test_logits = model_adam(test_src)
    test_loss = nn.functional.cross_entropy(test_logits.view(-1, vocab_size), test_tgt.view(-1))
    test_acc = (test_logits.argmax(-1) == test_tgt).float().mean()
print(
    f"transf  AdamW | step={step:5d} | f_calls={step+1:7d} | train_loss={loss.item():.2e} | train_acc={train_acc:.2%}"
    f" | test_loss={test_loss.item():.2e} | test_acc={test_acc:.2%}"
)
