from concurrent.futures import ProcessPoolExecutor
from itertools import product
from typing import Callable, NamedTuple, TypeVarTuple

import numpy as np
import numpy.typing as npt


class OptimizeResult(NamedTuple):
    x: npt.NDArray[np.float64]
    fun: float
    nfev: int
    nit: int


Ts = TypeVarTuple("Ts")


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
    maxiter: int = 10_000,
    popsize: int = 16,
    sigma: float = 1e-2,
    max_sigma: float = 1e-1,
    min_sigma: float = 1e-8,
    sigma_memory_size: int = 8,
    x0: npt.NDArray[np.float64] | None = None,
    workers: int = 1,
    rng: np.random.Generator | None = None,
    callback: Callable[[npt.NDArray[np.float64]], bool] | None = None,
) -> OptimizeResult:
    params = len(bounds)
    epoch = 0
    max_epoch = int(round(maxiter / popsize))
    range_min = tuple(map(lambda d: bounds[d][0], range(params)))
    range_max = tuple(map(lambda d: bounds[d][1], range(params)))
    sigma_memory = [sigma for _ in range(sigma_memory_size)]
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
    best_errors = np.full(shape=popsize, fill_value=np.inf, dtype=np.float64)
    restart = False
    func_ = FuncWrapper(func, args)
    process_pool = None
    if workers > 1:
        process_pool = ProcessPoolExecutor(workers)
    ind = 0
    last_diff = 0
    last_error = np.inf
    for epoch in range(max_epoch):
        if epoch > 0 and not restart:
            for p, d in product(range(popsize), range(params)):
                r = round(rng.random() * (popsize - 1))
                current_positions[p, d] = min(
                    max(
                        best_positions[r, d]
                        + rng.normal(0, sigma)
                        * np.abs(best_positions[r, d] - current_positions[p, d]),
                        range_min[d],
                    ),
                    range_max[d],
                )
        if restart:
            current_positions = np.zeros(shape=(popsize, params), dtype=np.float64)
            for p, d in product(range(1, popsize), range(params)):
                current_positions[p, d] = rng.uniform(range_min[d], range_max[d])
            current_positions[0] = x0
            best_positions = np.zeros(shape=(popsize, params), dtype=np.float64)
            best_errors = np.full(shape=popsize, fill_value=np.inf, dtype=np.float64)
            restart = False
        if process_pool is not None:
            current_errors = tuple(process_pool.map(func_, current_positions))
        else:
            current_errors = tuple(func(x, *args) for x in current_positions)
        for p in range(popsize):
            if current_errors[p] < best_errors[p]:
                best_errors[p] = current_errors[p]
                best_positions[p] = current_positions[p]
        min_best_error = np.min(best_errors)
        max_best_error = np.max(best_errors)
        for i in range(popsize):
            if best_errors[i] < best_errors[ind]:
                ind = i
        if abs((max_best_error - min_best_error) / max_best_error) < sigma / 10:
            diff = last_error - min_best_error
            if diff >= last_diff:
                sigma_memory.append(min(sigma * 2, max_sigma))
            else:
                sigma_memory.append(max(sigma / 2, min_sigma))
            sigma_memory = sigma_memory[1:]
            sigma = float(np.mean(sigma_memory))
            last_diff = diff
            last_error = min_best_error
            x0 = best_positions[ind]
            restart = True
        if callback is not None:
            if callback(best_positions[ind]):
                break
    if process_pool is not None:
        process_pool.shutdown()
    return OptimizeResult(
        x=best_positions[ind],
        fun=best_errors[ind],
        nit=epoch + 1,
        nfev=(epoch + 1) * popsize,
    )
