import torch
import torch.nn as nn

from ansr.ansr_torch import ANSR
from ansr.callbacks import TorchEarlyStopCallback

# target function: y = 2*x + 3*x^2
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.linspace(-1, 1, 50, device=device).unsqueeze(1)
y = 2.0 * x + 3.0 * x ** 2

model = torch.compile(nn.Sequential(
    nn.Linear(1, 16),
    nn.Tanh(),
    nn.Linear(16, 1),
).to(device))

optimizer = ANSR(
    model.parameters(),
    model=model,
    batch_size=8,
    popsize=8,
    sigma=0.05,
    self_instead_neighbour=0.9,
    bound=5.0,
    seed=0,
)
callback = TorchEarlyStopCallback(stop_residual=0.01)

def loss_fn(output):
    return nn.functional.mse_loss(output, y)

for step in range(5000):
    loss = optimizer.step(loss_fn, x)
    if step % 500 == 0:
        print(f"step {step:4d}  loss={float(loss):.6f}")
    if callback(loss):
        print(f"converged at step {step}  loss={float(loss):.6f}")
        break

print(f"final loss={float(loss):.6f}")
