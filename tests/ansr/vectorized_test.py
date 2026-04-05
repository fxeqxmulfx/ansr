import numpy as np

from ansr.ansr import ansr_minimize
from ansr.callbacks import EarlyStopCallback


# --- 2D test functions scaled to [0,1] ---

def _scale(v, in_min, in_max):
    return (v - in_min) / (in_max - in_min)

def _sphere(x, y):
    return _scale(x**2 + y**2, 0.0, 50.0)

def _ellipsoid(x, y):
    return _scale(x**2 + 1_000_000.0 * y**2, 0.0, 25_000_025.0)

def _rosenbrock(x, y):
    return _scale(100.0 * (x**2 - y) ** 2 + (x - 1.0) ** 2, 0.0, 90_036.0)

def _discus(x, y):
    return _scale(1_000_000.0 * x**2 + y**2, 0.0, 25_000_025.0)

def _different_powers(x, y):
    return _scale(x**2 + y**6, 0.0, 15_650.0)

def _shubert(x, y):
    i = np.arange(1, 6)
    s1 = np.sum(i * np.cos((i + 1) * np.asarray(x)[..., None] + i), axis=-1)
    s2 = np.sum(i * np.cos((i + 1) * np.asarray(y)[..., None] + i), axis=-1)
    return _scale(s1 * s2, -186.7309088, 0.0)

def _broadcast(func_2d):
    def wrapper(x):
        return np.mean(func_2d(x[0::2], x[1::2]))
    return wrapper

LMMAES_TEST_FUNCTIONS = {
    "sphere":           (_sphere,           ((-5, 5), (-5, 5))),
    "ellipsoid":        (_ellipsoid,        ((-5, 5), (-5, 5))),
    "rosenbrock":       (_rosenbrock,       ((-5, 5), (-5, 5))),
    "discus":           (_discus,           ((-5, 5), (-5, 5))),
    "different_powers": (_different_powers, ((-5, 5), (-5, 5))),
}


# --- function extrema ---

def test_sphere_min():
    assert abs(_sphere(0, 0)) < 1e-6

def test_sphere_max():
    assert abs(_sphere(5, 5) - 1) < 1e-6

def test_ellipsoid_min():
    assert abs(_ellipsoid(0, 0)) < 1e-6

def test_ellipsoid_max():
    assert abs(_ellipsoid(5, 5) - 1) < 1e-6

def test_rosenbrock_min():
    assert abs(_rosenbrock(1, 1)) < 1e-6

def test_rosenbrock_max():
    assert abs(_rosenbrock(-5, -5) - 1) < 1e-6

def test_discus_min():
    assert abs(_discus(0, 0)) < 1e-6

def test_discus_max():
    assert abs(_discus(5, 5) - 1) < 1e-6

def test_different_powers_min():
    assert abs(_different_powers(0, 0)) < 1e-6

def test_different_powers_max():
    assert abs(_different_powers(5, 5) - 1) < 1e-3

def test_shubert_min():
    assert abs(_shubert(-1.42513, -0.80032)) < 1e-4

def test_shubert_positive_at_origin():
    assert _shubert(0.0, 0.0) > 0.0


# --- ANSR convergence (16D, 100k evals) ---

def _check_convergence(name):
    fn2d, bounds_2d = LMMAES_TEST_FUNCTIONS[name]
    func = _broadcast(fn2d)
    bounds = bounds_2d * 8  # 16D
    cb = EarlyStopCallback(func, stop_residual=0.01)
    result = ansr_minimize(
        func, bounds, maxiter=100_000,
        popsize=4, restart_tolerance=0.01, sigma=0.05,
        self_instead_neighbour=0.9, callback=cb,
        seed=0,
    )
    assert result.fun <= 0.01, f"{name} did not converge: fun={result.fun}"
    return result.nfev

def test_convergence_sphere():
    _check_convergence("sphere")

def test_convergence_ellipsoid():
    _check_convergence("ellipsoid")

def test_convergence_rosenbrock():
    _check_convergence("rosenbrock")

def test_convergence_discus():
    _check_convergence("discus")

def test_convergence_different_powers():
    _check_convergence("different_powers")


# --- shubert (16D) ---

def test_shubert_16d():
    func = _broadcast(_shubert)
    bounds = ((-10, 10), (-10, 10)) * 8  # 16D
    cb = EarlyStopCallback(func, stop_residual=0.01)
    result = ansr_minimize(
        func, bounds, maxiter=500_000,
        popsize=64, restart_tolerance=1e-8, sigma=0.04,
        self_instead_neighbour=0.0, callback=cb,
        seed=0,
    )
    assert result.fun <= 0.01, f"shubert did not converge: fun={result.fun}"


# --- determinism ---

def test_determinism():
    func = _broadcast(_sphere)
    bounds = ((-5, 5), (-5, 5)) * 8
    cb1 = EarlyStopCallback(func, stop_residual=0.01)
    cb2 = EarlyStopCallback(func, stop_residual=0.01)
    r1 = ansr_minimize(func, bounds, maxiter=10_000, callback=cb1, seed=42)
    r2 = ansr_minimize(func, bounds, maxiter=10_000, callback=cb2, seed=42)
    assert r1.fun == r2.fun
    assert r1.nfev == r2.nfev
