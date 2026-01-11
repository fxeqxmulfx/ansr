from concurrent.futures import ProcessPoolExecutor
from itertools import combinations, product
from typing import Callable, NamedTuple, TypeVarTuple

import numpy as np
import numpy.typing as npt

Ts = TypeVarTuple("Ts")


class EarlyStopCallback:
    __slots__ = (
        "func",
        "args",
        "stop_residual",
    )

    def __init__(
        self,
        func: Callable[[npt.NDArray[np.float64], *Ts], float],
        args: tuple[*Ts] = (),
        stop_residual: float = 0.1,
    ) -> None:
        self.func = func
        self.args = args
        self.stop_residual = stop_residual

    def __call__(self, x: npt.NDArray[np.float64], *args, **kwargs) -> bool:
        residual = self.func(x, *self.args)
        if residual <= self.stop_residual:
            return True
        return False


class WindowEarlyStopCallback:
    __slots__ = (
        "func",
        "args",
        "window_size",
        "min_difference",
        "last_residual",
        "current_call",
    )

    def __init__(
        self,
        func: Callable[[npt.NDArray[np.float64], *Ts], float],
        args: tuple[*Ts] = (),
        window_size: int = 512,
        min_difference: float = 0.01,
    ) -> None:
        self.func = func
        self.args = args
        self.window_size = window_size
        self.min_difference = min_difference
        self.last_residual = np.finfo(np.float32).max
        self.current_call = 0

    def __call__(self, x: npt.NDArray[np.float64], *args, **kwargs) -> bool:
        residual = self.func(x, *self.args)
        if self.current_call % self.window_size == 0:
            difference = self.last_residual - residual
            if difference <= self.min_difference:
                return True
            self.last_residual = residual
        self.current_call += 1
        return False


class OptimizeResult(NamedTuple):
    x: npt.NDArray[np.float64]
    fun: float
    nfev: int
    nit: int


class FuncWrapper:
    __slots__ = (
        "func",
        "args",
    )

    def __init__(
        self,
        func: Callable[[npt.NDArray[np.float64], *Ts], float],
        args: tuple[*Ts],
    ) -> None:
        self.func = func
        self.args = args

    def __call__(self, x: npt.NDArray[np.float64]) -> float:
        return self.func(x, *self.args)


def ansr_minimize(
    func: Callable[[npt.NDArray[np.float64], *Ts], float],
    bounds: tuple[tuple[float, float], ...],
    args: tuple[*Ts] = (),
    maxiter: int = 100_000,
    popsize: int = 16,
    sigma: float = 1e-1,
    tol: float = 1e-4,
    x0: npt.NDArray[np.float64] | None = None,
    workers: int = 1,
    rng: np.random.Generator | None = None,
    callback: Callable[[npt.NDArray[np.float64]], bool] | None = None,
) -> OptimizeResult:
    params = len(bounds)
    epoch = 0
    max_epoch = int(round(maxiter / popsize))
    range_min = np.array(tuple(map(lambda d: bounds[d][0], range(params))))
    range_max = np.array(tuple(map(lambda d: bounds[d][1], range(params))))
    if rng is None:
        rng = np.random.default_rng(42)
    if x0 is None:
        x0 = np.zeros(shape=params, dtype=np.float64)
        for d in range(params):
            x0[d] = rng.uniform(range_min[d], range_max[d])
    current_positions = np.zeros(shape=(popsize, params), dtype=np.float64)
    for p, d in product(range(1, popsize), range(params)):
        current_positions[p, d] = rng.uniform(range_min[d], range_max[d])
    current_positions[0] = x0
    best_positions = np.zeros(shape=(popsize, params), dtype=np.float64)
    best_residuals = np.full(
        shape=popsize, fill_value=np.finfo(np.float32).max, dtype=np.float64
    )
    func_ = FuncWrapper(func, args)
    process_pool = None
    if workers > 1:
        process_pool = ProcessPoolExecutor(workers)
    ind = 0
    for epoch in range(max_epoch):
        if process_pool is not None:
            current_residuals = tuple(process_pool.map(func_, current_positions))
        else:
            current_residuals = tuple(func(x, *args) for x in current_positions)
        for p in range(popsize):
            if current_residuals[p] < best_residuals[p]:
                best_residuals[p] = current_residuals[p]
                best_positions[p] = current_positions[p]
                if best_residuals[p] < best_residuals[ind]:
                    ind = p
        if callback is not None and callback(best_positions[ind]):
            break
        for i, j in combinations(range(popsize), 2):
            if (
                best_residuals[i] != np.finfo(np.float32).max
                and best_residuals[j] != np.finfo(np.float32).max
                and max(best_residuals[i], best_residuals[j]) != 0
                and abs(
                    (best_residuals[i] - best_residuals[j])
                    / max(best_residuals[i], best_residuals[j])
                )
                < tol
            ):
                best_residuals[j] = np.finfo(np.float32).max
                for d in range(params):
                    best_positions[j, d] = rng.uniform(range_min[d], range_max[d])
        for p, d in product(range(popsize), range(params)):
            r = rng.integers(0, popsize)
            current_positions[p, d] = min(
                max(
                    best_positions[r, d]
                    + rng.normal(0, sigma)
                    * np.abs(best_positions[r, d] - current_positions[p, d]),
                    range_min[d],
                ),
                range_max[d],
            )
    if process_pool is not None:
        process_pool.shutdown()
    return OptimizeResult(
        x=best_positions[ind],
        fun=best_residuals[ind],
        nit=epoch + 1,
        nfev=(epoch + 1) * popsize,
    )
