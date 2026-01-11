import numpy as np
import numpy.typing as npt

from ansr.ansr import EarlyStopCallback, ansr_minimize

sphere_bounds = ((-10, 10), (-10, 10))


def sphere(x: npt.NDArray[np.float64]) -> float:
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
            rng=np.random.default_rng(i),
        )
        fun[i] = result.fun
        nfev[i] = result.nfev
    assert float(np.mean(fun)) <= 0.1
    assert float(np.mean(nfev)) == 180.8


def test_sphere_32():
    n = 10
    fun = np.zeros(n)
    nfev = np.zeros(n)
    for i in range(n):
        result = ansr_minimize(
            sphere,
            sphere_bounds * 32,
            callback=EarlyStopCallback(sphere),
            rng=np.random.default_rng(i),
        )
        fun[i] = result.fun
        nfev[i] = result.nfev
    assert float(np.mean(fun)) <= 0.1
    assert float(np.mean(nfev)) == 6371.2


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


def test_shubert_1():
    n = 10
    fun = np.zeros(n)
    nfev = np.zeros(n)
    for i in range(n):
        result = ansr_minimize(
            shubert,
            shubert_bounds,
            callback=EarlyStopCallback(shubert),
            rng=np.random.default_rng(i),
        )
        fun[i] = result.fun
        nfev[i] = result.nfev
    assert float(np.mean(fun)) <= 0.1
    assert float(np.mean(nfev)) == 472.0


def test_shubert_32():
    n = 10
    fun = np.zeros(n)
    nfev = np.zeros(n)
    for i in range(n):
        result = ansr_minimize(
            shubert,
            shubert_bounds * 32,
            callback=EarlyStopCallback(shubert),
            rng=np.random.default_rng(i),
        )
        fun[i] = result.fun
        nfev[i] = result.nfev
    assert float(np.mean(fun)) <= 0.1
    assert float(np.mean(nfev)) == 55649.6
