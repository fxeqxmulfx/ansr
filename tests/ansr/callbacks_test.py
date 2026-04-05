import numpy as np

from ansr.ansr import FuncWrapper
from ansr.callbacks import (
    EarlyStopCallback, WindowEarlyStopCallback,
    TorchEarlyStopCallback, TorchWindowEarlyStopCallback,
)


def sphere(x):
    return float(np.sum(x**2))


def weighted_sphere(x, w):
    return float(np.sum(w * x**2))


# --- EarlyStopCallback ---

def test_early_stop_below_residual():
    cb = EarlyStopCallback(sphere, stop_residual=0.1)
    assert cb(np.array([0.0, 0.0])) is True

def test_early_stop_above_residual():
    cb = EarlyStopCallback(sphere, stop_residual=0.1)
    assert cb(np.array([1.0, 1.0])) is False

def test_early_stop_at_residual():
    cb = EarlyStopCallback(sphere, stop_residual=2.0)
    assert cb(np.array([1.0, 1.0])) is True

def test_early_stop_with_args():
    w = np.array([2.0, 3.0])
    cb = EarlyStopCallback(weighted_sphere, args=(w,), stop_residual=0.1)
    assert cb(np.array([0.0, 0.0])) is True
    assert cb(np.array([1.0, 1.0])) is False

def test_early_stop_default_residual():
    cb = EarlyStopCallback(sphere)
    assert cb.stop_residual == 0.1


# --- WindowEarlyStopCallback ---

def test_window_stops_on_no_improvement():
    cb = WindowEarlyStopCallback(sphere, window_size=1, min_difference=0.01)
    # first call: sets last_residual, difference = max - 0 > 0.01, continues
    assert cb(np.array([0.0, 0.0])) is False
    # second call: residual=0, difference=0-0=0 <= 0.01, stops
    assert cb(np.array([0.0, 0.0])) is True

def test_window_continues_with_improvement():
    cb = WindowEarlyStopCallback(sphere, window_size=1, min_difference=0.01)
    assert cb(np.array([10.0, 10.0])) is False  # last_residual = 200
    assert cb(np.array([1.0, 1.0])) is False     # diff = 200-2 = 198 > 0.01
    assert cb(np.array([0.5, 0.5])) is False     # diff = 2-0.5 = 1.5 > 0.01

def test_window_skips_between_windows():
    cb = WindowEarlyStopCallback(sphere, window_size=3, min_difference=0.01)
    # call 0: window boundary, sets last_residual=200
    assert cb(np.array([10.0, 10.0])) is False
    # calls 1, 2: not on window boundary, always False
    assert cb(np.array([0.0, 0.0])) is False
    assert cb(np.array([0.0, 0.0])) is False
    # call 3: window boundary, big improvement (200->0), continues, sets last_residual=0
    assert cb(np.array([0.0, 0.0])) is False
    # calls 4, 5: not on window boundary
    assert cb(np.array([0.0, 0.0])) is False
    assert cb(np.array([0.0, 0.0])) is False
    # call 6: window boundary, no improvement (0->0), stops
    assert cb(np.array([0.0, 0.0])) is True

def test_window_with_args():
    w = np.array([1.0, 1.0])
    cb = WindowEarlyStopCallback(weighted_sphere, args=(w,), window_size=1,
                                 min_difference=0.01)
    assert cb(np.array([0.0, 0.0])) is False
    assert cb(np.array([0.0, 0.0])) is True

def test_window_default_params():
    cb = WindowEarlyStopCallback(sphere)
    assert cb.window_size == 512
    assert cb.min_difference == 0.01


# --- FuncWrapper ---

def test_func_wrapper_no_args():
    fw = FuncWrapper(sphere, ())
    assert fw(np.array([3.0, 4.0])) == 25.0

def test_func_wrapper_with_args():
    w = np.array([2.0, 3.0])
    fw = FuncWrapper(weighted_sphere, (w,))
    # 2*1 + 3*4 = 14
    assert fw(np.array([1.0, 2.0])) == 14.0

def test_func_wrapper_returns_float():
    fw = FuncWrapper(sphere, ())
    result = fw(np.array([0.0, 0.0]))
    assert result == 0.0


# --- TorchEarlyStopCallback ---

def test_torch_early_stop_below():
    cb = TorchEarlyStopCallback(stop_residual=0.1)
    assert cb(0.0) is True

def test_torch_early_stop_above():
    cb = TorchEarlyStopCallback(stop_residual=0.1)
    assert cb(1.0) is False

def test_torch_early_stop_at():
    cb = TorchEarlyStopCallback(stop_residual=0.5)
    assert cb(0.5) is True

def test_torch_early_stop_default():
    cb = TorchEarlyStopCallback()
    assert cb.stop_residual == 0.1


# --- TorchWindowEarlyStopCallback ---

def test_torch_window_stops_on_no_improvement():
    cb = TorchWindowEarlyStopCallback(window_size=1, min_difference=0.01)
    assert cb(0.0) is False   # sets last_loss=0
    assert cb(0.0) is True    # diff=0-0=0 <= 0.01, stops

def test_torch_window_continues_with_improvement():
    cb = TorchWindowEarlyStopCallback(window_size=1, min_difference=0.01)
    assert cb(200.0) is False  # last_loss=200
    assert cb(2.0) is False    # diff=198 > 0.01
    assert cb(0.5) is False    # diff=1.5 > 0.01

def test_torch_window_skips_between_windows():
    cb = TorchWindowEarlyStopCallback(window_size=3, min_difference=0.01)
    assert cb(200.0) is False  # call 0: boundary, last_loss=200
    assert cb(0.0) is False    # call 1: skip
    assert cb(0.0) is False    # call 2: skip
    assert cb(0.0) is False    # call 3: boundary, improvement 200->0, last_loss=0
    assert cb(0.0) is False    # call 4: skip
    assert cb(0.0) is False    # call 5: skip
    assert cb(0.0) is True     # call 6: boundary, no improvement, stops

def test_torch_window_default_params():
    cb = TorchWindowEarlyStopCallback()
    assert cb.window_size == 512
    assert cb.min_difference == 0.01
