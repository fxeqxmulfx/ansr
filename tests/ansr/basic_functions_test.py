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


def sphere_batched(x: np.ndarray) -> np.ndarray:
    return np.sum(x ** 2, axis=-1) / x.shape[-1] * 2


def test_batched_matches_sequential():
    sequential = ansr_minimize(sphere, sphere_bounds * 4, maxiter=2_000, seed=0)
    batched = ansr_minimize(sphere_batched, sphere_bounds * 4, maxiter=2_000, seed=0, batched=True)
    assert sequential.fun == batched.fun
    assert sequential.nfev == batched.nfev
    assert sequential.nrestarts == batched.nrestarts
    np.testing.assert_array_equal(sequential.x, batched.x)


def test_batched_converges():
    result = ansr_minimize(
        sphere_batched,
        sphere_bounds * 8,
        callback=EarlyStopCallback(sphere),
        seed=0,
        batched=True,
    )
    assert result.fun <= 0.1


def test_batched_rejects_workers():
    import pytest
    with pytest.raises(ValueError):
        ansr_minimize(sphere_batched, sphere_bounds, batched=True, workers=2)
