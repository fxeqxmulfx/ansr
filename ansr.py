from concurrent.futures import ProcessPoolExecutor
from itertools import product
from typing import Callable, NamedTuple, TypeVarTuple

import numpy as np
import numpy.typing as npt

Ts = TypeVarTuple("Ts")


class EarlyStopCallback:
    def __init__(
        self,
        func: Callable[[npt.NDArray[np.float64], *Ts], float],
        args: tuple[*Ts] = (),
        stop_error: float = 0.1,
    ) -> None:
        self.func = func
        self.args = args
        self.stop_error = stop_error

    def __call__(self, x: npt.NDArray[np.float64]) -> bool:
        error = self.func(x, *self.args)
        if error <= self.stop_error:
            return True
        return False


class OptimizeResult(NamedTuple):
    x: npt.NDArray[np.float64]
    fun: float
    nfev: int
    nit: int


class FuncWrapper:
    __slots__ = ("func", "args")

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
    best_errors = np.full(
        shape=popsize, fill_value=np.finfo(np.float32).max, dtype=np.float64
    )
    func_ = FuncWrapper(func, args)
    process_pool = None
    if workers > 1:
        process_pool = ProcessPoolExecutor(workers)
    ind = 0
    for epoch in range(max_epoch):
        if epoch > 0:
            for p, d in product(range(popsize), range(params)):
                r = rng.integers(0, popsize)
                if (
                    p != r
                    and best_errors[r] != np.finfo(np.float32).max
                    and max(best_errors[p], best_errors[r]) != 0
                    and abs(
                        (best_errors[p] - best_errors[r])
                        / max(best_errors[p], best_errors[r])
                    )
                    < tol
                ):
                    for d2 in range(params):
                        best_positions[r, d2] = rng.uniform(
                            range_min[d2], range_max[d2]
                        )
                        best_errors[r] = np.finfo(np.float32).max
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
            current_errors = tuple(process_pool.map(func_, current_positions))
        else:
            current_errors = tuple(func(x, *args) for x in current_positions)
        for p in range(popsize):
            if current_errors[p] < best_errors[p]:
                best_errors[p] = current_errors[p]
                best_positions[p] = current_positions[p]
                if best_errors[p] < best_errors[ind]:
                    ind = p
        if callback is not None and callback(best_positions[ind]):
            break
    if process_pool is not None:
        process_pool.shutdown()
    return OptimizeResult(
        x=best_positions[ind],
        fun=best_errors[ind],
        nit=epoch + 1,
        nfev=(epoch + 1) * popsize,
    )
