from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, NamedTuple

import numpy as np
import numpy.typing as npt

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
        func: Callable[..., float],
        args: tuple[Any, ...],
    ) -> None:
        self.func = func
        self.args = args

    def __call__(self, x: npt.NDArray[np.float64]) -> float:
        return self.func(x, *self.args)


def ansr_minimize(
    func: Callable[..., float],
    bounds: tuple[tuple[float, float], ...],
    args: tuple[Any, ...] = (),
    maxiter: int = 100_000,
    popsize: int = 64,
    sigma: float = 0.05,
    restart_tolerance: float = 1e-8,
    self_instead_neighbour: float = 0.95,
    x0: npt.NDArray[np.float64] | None = None,
    workers: int = 1,
    seed: int = 0,
    callback: Callable[[npt.NDArray[np.float64]], bool] | None = None,
) -> OptimizeResult:
    dims = len(bounds)
    max_epoch = int(round(maxiter / popsize))
    range_min = np.array([b[0] for b in bounds])
    range_max = np.array([b[1] for b in bounds])
    range_span = range_max - range_min

    rng = np.random.default_rng(seed)

    # initialise positions in [0,1] normalised space
    pos = rng.uniform(0.0, 1.0, size=(popsize, dims))
    if x0 is not None:
        pos[0] = (x0 - range_min) / range_span

    best_pos = np.zeros_like(pos)
    best_res = np.full(popsize, np.inf)
    ind = 0
    epoch = 0

    # precompute fixed index arrays
    p_idx = np.arange(popsize)[:, None]
    d_idx = np.arange(dims)
    ii, jj = np.triu_indices(popsize, k=1)

    # pre-allocate reused buffers
    mapped = np.empty((popsize, dims))
    current_res = np.empty(popsize)

    func_ = FuncWrapper(func, args) if args else func
    process_pool = None
    if workers > 1:
        process_pool = ProcessPoolExecutor(workers)

    for epoch in range(max_epoch):
        # map positions to original space
        np.multiply(pos, range_span, out=mapped)
        mapped += range_min

        # evaluate
        if process_pool is not None:
            results = process_pool.map(func_, mapped)
            for p, v in enumerate(results):
                current_res[p] = v
        else:
            for p in range(popsize):
                current_res[p] = func(mapped[p], *args)

        # vectorized update best (in-place)
        improved = current_res < best_res
        best_res[improved] = current_res[improved]
        best_pos[improved] = pos[improved]
        ind = int(np.argmin(best_res))

        if callback is not None and callback(range_min + best_pos[ind] * range_span):
            break

        # vectorized restart: evaluate all pairs simultaneously
        ri, rj = best_res[ii], best_res[jj]
        mx = np.maximum(ri, rj)
        mn = np.minimum(ri, rj)
        converged = np.isfinite(mx) & (mx != 0.0) & ((mx - mn) / mx < restart_tolerance)
        if converged.any():
            is_i_winner = (ii == ind) | ((jj != ind) & (ri < rj))
            losers = np.unique(np.where(is_i_winner, jj, ii)[converged])
            best_res[losers] = np.inf
            best_pos[losers] = rng.uniform(0.0, 1.0, size=(len(losers), dims))
            pos[losers] = rng.uniform(0.0, 1.0, size=(len(losers), dims))

        # vectorized perturbation
        noise = rng.normal(0.0, sigma, size=(popsize, dims))
        if self_instead_neighbour == 1.0:
            guide = best_pos
        elif self_instead_neighbour == 0.0:
            r = rng.integers(0, popsize - 1, size=(popsize, dims))
            r += (r >= p_idx)
            guide = best_pos[r, d_idx]
        else:
            use_self = rng.uniform(size=(popsize, dims)) <= self_instead_neighbour
            r = rng.integers(0, popsize - 1, size=(popsize, dims))
            r += (r >= p_idx)
            guide = np.where(use_self, best_pos, best_pos[r, d_idx])
        delta = guide - pos
        np.abs(delta, out=delta)
        delta *= noise
        delta += guide
        np.clip(delta, 0.0, 1.0, out=pos)

    if process_pool is not None:
        process_pool.shutdown()

    return OptimizeResult(
        x=range_min + best_pos[ind] * range_span,
        fun=best_res[ind],
        nit=epoch + 1,
        nfev=(epoch + 1) * popsize,
    )
