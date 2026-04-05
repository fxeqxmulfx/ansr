Across Neighbourhood Search with Restarts (ANSR)

Links:
- [arxiv.org: Across neighbourhood search for numerical optimization](https://arxiv.org/abs/1401.3376)

```bash
pip install git+https://github.com/fxeqxmulfx/ansr           # numpy only
pip install "ansr[torch] @ git+https://github.com/fxeqxmulfx/ansr"  # or with torch
```

## NumPy

```python
import numpy as np

from ansr.ansr import ansr_minimize
from ansr.callbacks import EarlyStopCallback

shubert_bounds = ((-10, 10), (-10, 10))


def shubert(x: np.ndarray) -> float:
    i = np.array((1, 2, 3, 4, 5))
    x = x.reshape(-1, 1)
    index_0 = np.arange(x.size) % 2 == 0
    index_1 = np.logical_not(index_0)
    return (
        np.sum(
            np.sum(i * np.cos((i + 1) * x[index_0] + i), axis=1)
            * np.sum(i * np.cos((i + 1) * x[index_1] + i), axis=1)
        )
        / x.size
        * 2
        + 186.7309
    )


result = ansr_minimize(
    shubert,
    shubert_bounds * 32,
    callback=EarlyStopCallback(shubert),
    seed=42,
)
print(result)
```

## PyTorch

ANSR is available as a `torch.optim.Optimizer`. No gradients needed.
Supports GPU and `torch.compile`.

### Sequential (closure)

Each step calls the closure `popsize` times, one population member at a time.

```python
import torch
import torch.nn as nn

from ansr.ansr_torch import ANSR

torch.manual_seed(0)
x = torch.linspace(-1, 1, 50).unsqueeze(1)
y = 2.0 * x + 3.0 * x ** 2

model = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 1))

optimizer = ANSR(model.parameters(), popsize=8, sigma=0.05, bound=5.0, seed=0)

def closure():
    return nn.functional.mse_loss(model(x), y)

for step in range(5000):
    loss = optimizer.step(closure)
```

### Batched (vmap)

Pass `model` and `batch_size` to evaluate multiple population members in parallel
via `torch.vmap`. The step takes a loss function and model inputs instead of a closure.

```python
optimizer = ANSR(
    model.parameters(),
    model=model,
    batch_size=8,
    popsize=8,
    sigma=0.05,
    bound=5.0,
    seed=0,
)

def loss_fn(output):
    return nn.functional.mse_loss(output, y)

for step in range(5000):
    loss = optimizer.step(loss_fn, x)

print(optimizer.restarts)  # total number of restarted particles
```

## Benchmark (ANSR vs AdamW)

5000 steps, early stop at loss <= 0.01 (sphere/shubert) or 0.1 (transformer).
Functions scaled to [0, 1] output range. See `examples/benchmark.py`.

| task | optimizer | options | steps | f calls | loss | restarts | accuracy |
|---|---|---|---|---|---|---|---|
| sphere 128D | ANSR | popsize=64, sigma=0.05, p_self=0.05, bound=5 | 66 | 4,288 | 9.49e-03 | 0 | --- |
| sphere 128D | ANSR | popsize=64, sigma=0.05, p_self=0.95, bound=5 | 227 | 14,592 | 9.44e-03 | 0 | --- |
| sphere 128D | AdamW | lr=0.01 | 88 | 89 | 9.98e-03 | --- | --- |
| shubert 64D | ANSR | popsize=64, sigma=0.05, p_self=0.05, bound=10 | 1162 | 74,432 | 1.35e-03 | 43 | --- |
| shubert 64D | AdamW | lr=0.01 | 99999 | 100,000 | 4.71e-01 | --- | --- |
| transformer 796p | ANSR | popsize=64, sigma=0.05, p_self=0.05, bound=20 | 4999 | 320,000 | 1.31e-01 | 1 | 93.06% / 95.83% |
| transformer 796p | ANSR | popsize=64, sigma=0.05, p_self=0.95, bound=20 | 4999 | 320,000 | 1.10e+00 | 0 | 49.31% / 54.17% |
| transformer 796p | AdamW | lr=0.01 | 34 | 35 | 8.84e-02 | --- | 100% / 100% |

Transformer uses train/test split (48/16 samples), accuracy shown as train/test.

ANSR wins on **shubert** (multimodal) --- gradients lead AdamW to a local minimum,
while ANSR's population + restarts find the global optimum. AdamW wins on **sphere**
and **transformer** where the landscape is smooth and gradients are informative.
Transformer behaves similar to unimodal --- low p_self is essential, high p_self
degrades accuracy from 93% to 49%.

Use ANSR when gradients are unavailable, misleading, or the landscape is multimodal.

## Parameter regime patterns

**Easy (unimodal):** Low sigma (0.12) with p_self ~ 0 (pure social learning) --- on a
single basin, every neighbour's best is informative.

**Terrain (multimodal, smooth):** High sigma (0.28--0.36) with p_self ~ 0.95 (almost
pure individual learning) --- neighbours are likely in different basins, so following
them is destructive.

**Periodic (Shubert):** Minimal perturbation (sigma = 0.04). Basins are narrow and
separated by steep ridges --- large steps waste evaluations. ANSR can afford p_self ~ 0
because restarts maintain diversity; ANS compensates with higher p_self (0.24--0.56)
but still degrades.

**Discrete:** ANS-family perturbations scaled by particle distances are ineffective on
step landscapes where gradient signal is zero everywhere.
