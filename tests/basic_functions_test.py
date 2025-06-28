import numpy as np
import numpy.typing as npt

from ansr import EarlyStopCallback, ansr_minimize

sphere_bounds = ((-10, 10), (-10, 10))


def sphere(x: npt.NDArray[np.float64]) -> float:
    return np.sum(x**2)


def test_sphere_1():
    result = ansr_minimize(
        sphere,
        sphere_bounds,
        callback=EarlyStopCallback(sphere),
    )
    assert result.fun <= 0.06930957495042038
    assert result.nfev <= 224


def test_sphere_32():
    result = ansr_minimize(
        sphere,
        sphere_bounds * 32,
        callback=EarlyStopCallback(sphere),
    )
    assert result.fun <= 0.09533376966916197
    assert result.nfev <= 12256


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
    result = ansr_minimize(
        shubert,
        shubert_bounds,
        callback=EarlyStopCallback(shubert),
    )
    assert result.fun <= 0.08625861203225327
    assert result.nfev <= 352


def test_shubert_32():
    result = ansr_minimize(
        shubert,
        shubert_bounds * 32,
        callback=EarlyStopCallback(shubert),
    )
    assert result.fun <= 0.06685240127401926
    assert result.nfev <= 33088
