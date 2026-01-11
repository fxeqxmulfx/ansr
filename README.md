Across Neighbourhood Search with Restarts (ANSR)

Links:
- [arxiv.org: Across neighbourhood search for numerical optimization](https://arxiv.org/abs/1401.3376)

```bash
pip install git+https://github.com/fxeqxmulfx/ansr
```

```python
import numpy as np
import numpy.typing as npt

from ansr.ansr import ansr_minimize, WindowEarlyStopCallback

shubert_bounds = ((-10, 10), (-10, 10))


def shubert(x: npt.NDArray[np.float64]) -> float:
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


if __name__ == "__main__":
    result = ansr_minimize(
        shubert,
        shubert_bounds * 32,
        callback=WindowEarlyStopCallback(shubert),
        rng=np.random.default_rng(42),
    )
    print(result)
```