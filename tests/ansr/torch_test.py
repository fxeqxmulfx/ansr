import pytest

torch = pytest.importorskip("torch")

from ansr.ansr_torch import ANSR
from ansr.callbacks import TorchEarlyStopCallback, TorchWindowEarlyStopCallback


# --- convergence ---

def test_sphere_convergence():
    dim = 16
    params = [torch.randn(dim, generator=torch.Generator().manual_seed(0))]
    optimizer = ANSR(params, popsize=4, sigma=0.05, self_instead_neighbour=0.9, seed=0)

    def closure():
        return (params[0] ** 2).sum()

    for _ in range(2000):
        loss = optimizer.step(closure)

    assert float(loss) < 0.1


def test_rosenbrock_convergence():
    g = torch.Generator().manual_seed(0)
    params = [torch.randn(2, generator=g)]
    optimizer = ANSR(params, popsize=8, sigma=0.05, self_instead_neighbour=0.9, seed=0)

    def closure():
        x, y = params[0][0], params[0][1]
        return (1.0 - x) ** 2 + 100.0 * (y - x ** 2) ** 2

    for _ in range(5000):
        loss = optimizer.step(closure)

    assert float(loss) < 1.0


# --- determinism ---

def test_determinism():
    dim = 8

    def run(seed):
        g = torch.Generator().manual_seed(seed)
        p = [torch.randn(dim, generator=g)]
        opt = ANSR(p, popsize=4, sigma=0.05, seed=seed)
        def closure():
            return (p[0] ** 2).sum()
        for _ in range(50):
            loss = opt.step(closure)
        return float(loss), p[0].clone()

    loss1, p1 = run(42)
    loss2, p2 = run(42)
    assert loss1 == loss2
    assert torch.equal(p1, p2)


def test_different_seeds_differ():
    dim = 8

    def run(seed):
        g = torch.Generator().manual_seed(seed)
        p = [torch.randn(dim, generator=g)]
        opt = ANSR(p, popsize=4, sigma=0.05, seed=seed)
        def closure():
            return (p[0] ** 2).sum()
        for _ in range(50):
            loss = opt.step(closure)
        return float(loss)

    assert run(0) != run(1)


# --- params handling ---

def test_sets_best_params():
    dim = 4
    params = [torch.ones(dim) * 10.0]
    optimizer = ANSR(params, popsize=4, sigma=0.05, seed=0)

    def closure():
        return (params[0] ** 2).sum()

    initial_loss = float(closure())
    optimizer.step(closure)
    assert float((params[0] ** 2).sum()) <= initial_loss


def test_multiple_param_groups():
    p1 = torch.randn(4, generator=torch.Generator().manual_seed(0))
    p2 = torch.randn(6, generator=torch.Generator().manual_seed(1))
    optimizer = ANSR([p1, p2], popsize=4, sigma=0.05, seed=0)

    def closure():
        return (p1 ** 2).sum() + (p2 ** 2).sum()

    initial_loss = float(closure())
    for _ in range(200):
        loss = optimizer.step(closure)
    assert float(loss) < initial_loss


def test_multidim_params():
    g = torch.Generator().manual_seed(0)
    params = [torch.randn(4, 4, generator=g)]
    optimizer = ANSR(params, popsize=4, sigma=0.05, seed=0)

    def closure():
        return (params[0] ** 2).sum()

    initial_loss = float(closure())
    for _ in range(200):
        loss = optimizer.step(closure)
    assert float(loss) < initial_loss


# --- loss monotonicity ---

def test_best_loss_non_increasing():
    dim = 8
    params = [torch.randn(dim, generator=torch.Generator().manual_seed(0))]
    optimizer = ANSR(params, popsize=4, sigma=0.05, seed=0)

    def closure():
        return (params[0] ** 2).sum()

    prev = float("inf")
    for _ in range(100):
        loss = float(optimizer.step(closure))
        assert loss <= prev
        prev = loss


# --- closure ---

def test_closure_required():
    params = [torch.zeros(4)]
    optimizer = ANSR(params)
    with pytest.raises(RuntimeError, match="requires a closure"):
        optimizer.step(None)


def test_closure_called_popsize_times():
    params = [torch.zeros(4)]
    optimizer = ANSR(params, popsize=8, seed=0)
    call_count = 0

    def closure():
        nonlocal call_count
        call_count += 1
        return (params[0] ** 2).sum()

    optimizer.step(closure)
    assert call_count == 8


# --- callbacks ---

def test_early_stop_callback():
    dim = 8
    params = [torch.randn(dim, generator=torch.Generator().manual_seed(0))]
    optimizer = ANSR(params, popsize=4, sigma=0.05, self_instead_neighbour=0.9, seed=0)
    callback = TorchEarlyStopCallback(stop_residual=0.5)

    def closure():
        return (params[0] ** 2).sum()

    steps = 0
    for _ in range(10000):
        loss = optimizer.step(closure)
        steps += 1
        if callback(loss):
            break

    assert float(loss) <= 0.5
    assert steps < 10000


def test_window_callback():
    dim = 4
    params = [torch.randn(dim, generator=torch.Generator().manual_seed(0))]
    optimizer = ANSR(params, popsize=4, sigma=0.05, seed=0)
    callback = TorchWindowEarlyStopCallback(window_size=50, min_difference=0.001)

    def closure():
        return (params[0] ** 2).sum()

    steps = 0
    for _ in range(10000):
        loss = optimizer.step(closure)
        steps += 1
        if callback(loss):
            break

    assert steps < 10000
