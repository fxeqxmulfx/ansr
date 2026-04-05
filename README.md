Across Neighbourhood Search with Restarts (ANSR)

Links:
- [arxiv.org: Across neighbourhood search for numerical optimization](https://arxiv.org/abs/1401.3376)

```bash
pip install git+https://github.com/fxeqxmulfx/ansr
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

5000 steps, early stop at loss <= 0.1. See `examples/benchmark.py`.

| task | optimizer | options | steps | loss | restarts | accuracy |
|---|---|---|---|---|---|---|
| sphere 796D | ANSR | popsize=64, sigma=0.1, p_self=0.95, bound=10 | 4999 | 3.27e-01 | 135 | --- |
| sphere 796D | AdamW | lr=0.01 | 560 | 9.95e-02 | --- | --- |
| shubert 64D | ANSR | popsize=35, sigma=0.04, p_self=0.05, bound=10 | 1837 | 1.67e-02 | 562 | --- |
| shubert 64D | AdamW | lr=0.01 | 4999 | 1.87e+02 | --- | --- |
| transformer 796p | ANSR | popsize=64, sigma=0.05, p_self=0.05, bound=20 | 4856 | 9.94e-02 | 0 | 97.92% |
| transformer 796p | AdamW | lr=0.01 | 34 | 8.56e-02 | --- | 100.00% |

ANSR wins on **shubert** (multimodal) --- gradients lead AdamW to a local minimum,
while ANSR's population + restarts find the global optimum. AdamW wins on **sphere**
and **transformer** where the landscape is smooth and gradients are informative.

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
