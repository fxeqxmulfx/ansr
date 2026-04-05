from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import numpy.typing as npt


class EarlyStopCallback:
    __slots__ = (
        "func",
        "args",
        "stop_residual",
    )

    def __init__(
        self,
        func: Callable[..., float],
        args: tuple[Any, ...] = (),
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
        func: Callable[..., float],
        args: tuple[Any, ...] = (),
        window_size: int = 512,
        min_difference: float = 0.01,
    ) -> None:
        self.func = func
        self.args = args
        self.window_size = window_size
        self.min_difference = min_difference
        self.last_residual: float = float(np.finfo(np.float32).max)
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


class TorchEarlyStopCallback:
    """Early stop callback for the torch ANSR optimizer.

    Usage::

        callback = TorchEarlyStopCallback(stop_residual=0.01)
        for _ in range(max_steps):
            loss = optimizer.step(closure)
            if callback(loss):
                break
    """
    __slots__ = ("stop_residual",)

    def __init__(self, stop_residual: float = 0.1) -> None:
        self.stop_residual = stop_residual

    def __call__(self, loss: float) -> bool:
        return float(loss) <= self.stop_residual


class TorchWindowEarlyStopCallback:
    """Window-based early stop callback for the torch ANSR optimizer.

    Stops when loss improvement over a window of steps falls below min_difference.

    Usage::

        callback = TorchWindowEarlyStopCallback(window_size=100, min_difference=0.01)
        for _ in range(max_steps):
            loss = optimizer.step(closure)
            if callback(loss):
                break
    """
    __slots__ = ("window_size", "min_difference", "last_loss", "current_call")

    def __init__(
        self,
        window_size: int = 512,
        min_difference: float = 0.01,
    ) -> None:
        self.window_size = window_size
        self.min_difference = min_difference
        self.last_loss = float("inf")
        self.current_call = 0

    def __call__(self, loss: float) -> bool:
        loss = float(loss)
        if self.current_call % self.window_size == 0:
            difference = self.last_loss - loss
            if difference <= self.min_difference:
                return True
            self.last_loss = loss
        self.current_call += 1
        return False
