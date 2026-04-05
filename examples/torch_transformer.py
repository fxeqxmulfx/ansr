import math

import torch
import torch.nn as nn

from ansr.ansr_torch import ANSR
from ansr.callbacks import TorchEarlyStopCallback

# --- tiny transformer for sequence copy task ---

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(16, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 2, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, vocab_size)
        self.scale = math.sqrt(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        x = self.embed(x) * self.scale + self.pos(positions)
        x = self.encoder(x)
        return self.head(x)


# task: copy input sequence (identity mapping)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = 4
seq_len = 3
n_samples = 16

src = torch.randint(0, vocab_size, (n_samples, seq_len), device=device)
tgt = src.clone()

model = torch.compile(TinyTransformer(vocab_size=vocab_size, d_model=8, nhead=1).to(device))

optimizer = ANSR(
    model.parameters(),
    model=model,
    batch_size=16,
    popsize=64,
    sigma=0.05,
    self_instead_neighbour=0.05,
    bound=5.0,
    seed=0,
)
callback = TorchEarlyStopCallback(stop_residual=0.1)

def loss_fn(logits):
    return nn.functional.cross_entropy(logits.view(-1, vocab_size), tgt.view(-1))

for step in range(5000):
    loss = optimizer.step(loss_fn, src)
    if step % 500 == 0:
        print(f"step {step:4d}  loss={float(loss):.4f}")
    if callback(loss):
        print(f"converged at step {step}  loss={float(loss):.4f}")
        break

# check accuracy
with torch.no_grad():
    pred = model(src).argmax(-1)
    acc = (pred == tgt).float().mean()
print(f"final loss={float(loss):.4f}  accuracy={acc:.2%}  restarts={optimizer.restarts}")
