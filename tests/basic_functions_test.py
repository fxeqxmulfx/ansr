import numpy as np
import numpy.typing as npt

from ansr import ansr_minimize

sphere_bounds = ((-10, 10), (-10, 10))


def sphere(x: npt.NDArray[np.float64]) -> float:
    return np.sum(x**2)


def test_sphere_1():
    np.testing.assert_almost_equal(
        ansr_minimize(
            sphere,
            sphere_bounds,
            maxiter=200,
        ).fun,
        0,
        decimal=1,
    )


def test_sphere_32():
    np.testing.assert_almost_equal(
        ansr_minimize(
            sphere,
            sphere_bounds * 32,
            maxiter=13_000,
        ).fun,
        0,
        decimal=1,
    )


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
    np.testing.assert_almost_equal(
        ansr_minimize(
            shubert,
            shubert_bounds,
            maxiter=400,
        ).fun,
        0,
        decimal=1,
    )


def test_shubert_32():
    np.testing.assert_almost_equal(
        ansr_minimize(
            shubert,
            shubert_bounds * 32,
            maxiter=33_000,
        ).fun,
        0,
        decimal=1,
    )
