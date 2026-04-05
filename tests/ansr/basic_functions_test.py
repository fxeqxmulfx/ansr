import numpy as np

from ansr.ansr import ansr_minimize
from ansr.callbacks import EarlyStopCallback

sphere_bounds = ((-10, 10), (-10, 10))


def sphere(x: np.ndarray) -> float:
    return np.sum(x**2) / x.size * 2


def test_sphere_1():
    n = 10
    fun = np.zeros(n)
    nfev = np.zeros(n)
    for i in range(n):
        result = ansr_minimize(
            sphere,
            sphere_bounds,
            callback=EarlyStopCallback(sphere),
            seed=i,
        )
        fun[i] = result.fun
        nfev[i] = result.nfev
    assert float(np.mean(fun)) <= 0.1
    assert float(np.mean(nfev)) == 1689.6


def test_sphere_32():
    n = 10
    fun = np.zeros(n)
    nfev = np.zeros(n)
    for i in range(n):
        result = ansr_minimize(
            sphere,
            sphere_bounds * 32,
            callback=EarlyStopCallback(sphere),
            seed=i,
        )
        fun[i] = result.fun
        nfev[i] = result.nfev
    assert float(np.mean(fun)) <= 0.1
    assert float(np.mean(nfev)) == 19251.2


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


def test_shubert_1():
    n = 10
    fun = np.zeros(n)
    nfev = np.zeros(n)
    for i in range(n):
        result = ansr_minimize(
            shubert,
            shubert_bounds,
            sigma=0.04,
            self_instead_neighbour=0.05,
            callback=EarlyStopCallback(shubert),
            seed=i,
        )
        fun[i] = result.fun
        nfev[i] = result.nfev
    assert float(np.mean(fun)) <= 0.1
    assert float(np.mean(nfev)) == 1702.4


def test_shubert_32():
    n = 10
    fun = np.zeros(n)
    nfev = np.zeros(n)
    for i in range(n):
        result = ansr_minimize(
            shubert,
            shubert_bounds * 32,
            maxiter=200_000,
            sigma=0.04,
            self_instead_neighbour=0.05,
            callback=EarlyStopCallback(shubert),
            seed=i,
        )
        fun[i] = result.fun
        nfev[i] = result.nfev
    assert float(np.mean(fun)) <= 0.1
    assert float(np.mean(nfev)) == 79987.2
