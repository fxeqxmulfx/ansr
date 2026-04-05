from __future__ import annotations

import torch
from torch import nn
from torch.func import functional_call, vmap
from torch.optim import Optimizer


class ANSR(Optimizer):
    """Across Neighbourhood Search with Restarts (ANSR) optimizer for PyTorch.

    Derivative-free optimizer. Like LBFGS, requires a closure in step().
    Each step() call evaluates the closure `popsize` times.

    Usage (sequential)::

        optimizer = ANSR(model.parameters(), popsize=4, sigma=0.05)

        def closure():
            return loss_fn(model(x), y)

        optimizer.step(closure)

    Usage (batched with vmap)::

        optimizer = ANSR(
            model.parameters(), model=model, batch_size=8, popsize=64,
        )

        def loss_fn(output):
            return nn.functional.mse_loss(output, y)

        optimizer.step(loss_fn, x)
    """

    def __init__(
        self,
        params,
        popsize: int = 64,
        sigma: float = 0.05,
        restart_tolerance: float = 1e-8,
        self_instead_neighbour: float = 0.95,
        bound: float = 10.0,
        seed: int = 0,
        model: nn.Module | None = None,
        batch_size: int = 1,
    ):
        defaults = dict(
            popsize=popsize,
            sigma=sigma,
            restart_tolerance=restart_tolerance,
            self_instead_neighbour=self_instead_neighbour,
            bound=bound,
        )
        super().__init__(params, defaults)
        self._device = self.param_groups[0]["params"][0].device
        self._generator = torch.Generator(device=self._device)
        self._generator.manual_seed(seed)
        self._ansr_state: dict[str, object] = {}
        self._model = model
        self._batch_size = batch_size
        if model is not None:
            self._param_names = [n for n, _ in model.named_parameters()]
            self._param_shapes = [p.shape for p in model.parameters()]
            self._buffers = dict(model.named_buffers())

    @property
    def restarts(self) -> int:
        return self._ansr_state.get("restarts", 0)  # type: ignore[return-value]

    def _flat_params(self) -> torch.Tensor:
        return torch.cat([p.data.view(-1) for group in self.param_groups
                          for p in group["params"]])

    def _set_flat_params(self, flat: torch.Tensor) -> None:
        offset = 0
        for group in self.param_groups:
            for p in group["params"]:
                numel = p.numel()
                p.data.copy_(flat[offset:offset + numel].view_as(p))
                offset += numel

    def _to_real(self, unit: torch.Tensor, bound: float) -> torch.Tensor:
        return unit * 2.0 * bound - bound

    def _to_unit(self, real: torch.Tensor, bound: float) -> torch.Tensor:
        return (real + bound) / (2.0 * bound)

    def _init_state(self) -> dict[str, object]:
        orig_flat = self._flat_params()
        dims = orig_flat.numel()
        group = self.param_groups[0]
        popsize = group["popsize"]
        bound = group["bound"]
        dev = self._device
        pos = torch.rand(popsize, dims, device=dev, generator=self._generator)
        pos[0] = self._to_unit(orig_flat, bound).clamp(0.0, 1.0)
        ii, jj = torch.triu_indices(popsize, popsize, offset=1, device=dev)
        self._ansr_state["dims"] = dims
        self._ansr_state["pos"] = pos
        self._ansr_state["best_pos"] = torch.zeros(popsize, dims, device=dev)
        self._ansr_state["best_res"] = torch.full((popsize,), float("inf"), device=dev)
        self._ansr_state["ind"] = torch.tensor(0, device=dev)
        self._ansr_state["ii"] = ii
        self._ansr_state["jj"] = jj
        self._ansr_state["p_idx"] = torch.arange(popsize, device=dev).unsqueeze(1)
        self._ansr_state["d_idx"] = torch.arange(dims, device=dev)
        self._ansr_state["current_res"] = torch.empty(popsize, device=dev)
        self._ansr_state["real_buf"] = torch.empty(dims, device=dev)
        self._ansr_state["restarts"] = 0
        return self._ansr_state

    def _eval_sequential(
        self, closure, pos, current_res, bound: float, real_buf,  # type: ignore[no-untyped-def]
    ) -> None:
        bound2 = 2.0 * bound
        popsize = pos.size(0)  # type: ignore[union-attr]
        for p in range(popsize):
            torch.mul(pos[p], bound2, out=real_buf)  # type: ignore[arg-type]
            real_buf.sub_(bound)
            self._set_flat_params(real_buf)  # type: ignore[arg-type]
            with torch.enable_grad():
                loss = closure()
            current_res[p] = loss.detach()

    def _eval_batched(
        self, loss_fn, inputs: tuple, pos, current_res, bound: float,  # type: ignore[no-untyped-def]
    ) -> None:
        assert self._model is not None
        model = self._model
        buffers = self._buffers
        popsize = pos.size(0)  # type: ignore[union-attr]
        batch_size = self._batch_size
        real_pos = self._to_real(pos, bound)  # type: ignore[arg-type]

        def compute_loss(params: dict[str, torch.Tensor]) -> torch.Tensor:
            out = functional_call(model, (params, buffers), inputs)
            return loss_fn(out)

        batched_fn = vmap(compute_loss, randomness="different")

        for start in range(0, popsize, batch_size):
            end = min(start + batch_size, popsize)
            batch_flat = real_pos[start:end]

            stacked: dict[str, torch.Tensor] = {}
            offset = 0
            for name, shape in zip(self._param_names, self._param_shapes):
                numel = shape.numel()
                stacked[name] = batch_flat[:, offset:offset + numel].view(-1, *shape)
                offset += numel

            with torch.enable_grad():
                losses = batched_fn(stacked)

            current_res[start:end] = losses.detach()

    @torch.no_grad()
    def step(self, closure_or_loss_fn, *inputs):  # type: ignore[override]
        if closure_or_loss_fn is None:
            raise RuntimeError("ANSR requires a closure or loss_fn")

        if not self._ansr_state:
            state = self._init_state()
        else:
            state = self._ansr_state

        group = self.param_groups[0]
        popsize = group["popsize"]
        sigma = group["sigma"]
        restart_tolerance = group["restart_tolerance"]
        self_instead_neighbour = group["self_instead_neighbour"]
        bound = group["bound"]

        pos = state["pos"]
        best_pos = state["best_pos"]
        best_res = state["best_res"]
        dims = state["dims"]
        ii = state["ii"]
        jj = state["jj"]
        p_idx = state["p_idx"]
        d_idx = state["d_idx"]

        dev = self._device
        current_res = state["current_res"]
        real_buf = state["real_buf"]

        # evaluate all population members
        if self._model is not None and self._batch_size > 1:
            self._eval_batched(closure_or_loss_fn, inputs, pos, current_res, bound)
        else:
            self._eval_sequential(closure_or_loss_fn, pos, current_res, bound, real_buf)

        # update best
        improved = current_res < best_res
        best_res[improved] = current_res[improved]
        best_pos[improved] = pos[improved]  # type: ignore[index]
        ind = torch.argmin(best_res)
        state["ind"] = ind

        # restart: check all pairs
        ri, rj = best_res[ii], best_res[jj]
        mx = torch.maximum(ri, rj)
        mn = torch.minimum(ri, rj)
        converged = torch.isfinite(mx) & (mx != 0.0) & ((mx - mn) / mx < restart_tolerance)
        if converged.any():
            is_i_winner = (ii == ind) | ((jj != ind) & (ri < rj))
            losers = torch.unique(torch.where(is_i_winner, jj, ii)[converged])
            best_res[losers] = float("inf")
            best_pos[losers] = torch.rand(len(losers), dims, device=dev, generator=self._generator)  # type: ignore[index, arg-type]
            pos[losers] = torch.rand(len(losers), dims, device=dev, generator=self._generator)  # type: ignore[index, arg-type]
            state["restarts"] = state["restarts"] + len(losers)  # type: ignore[operator]

        # perturbation
        noise = torch.randn(popsize, dims, device=dev, generator=self._generator) * sigma  # type: ignore[arg-type]
        if self_instead_neighbour == 1.0:
            guide = best_pos
        elif self_instead_neighbour == 0.0:
            r = torch.randint(0, popsize - 1, (popsize, dims), device=dev, generator=self._generator)  # type: ignore[arg-type]
            r = r + (r >= p_idx).long()
            guide = best_pos[r, d_idx]  # type: ignore[index]
        else:
            use_self = torch.rand(popsize, dims, device=dev, generator=self._generator) <= self_instead_neighbour  # type: ignore[arg-type]
            r = torch.randint(0, popsize - 1, (popsize, dims), device=dev, generator=self._generator)  # type: ignore[arg-type]
            r = r + (r >= p_idx).long()
            guide = torch.where(use_self, best_pos, best_pos[r, d_idx])  # type: ignore[index]

        pos.copy_(((guide - pos).abs() * noise + guide).clamp(0.0, 1.0))  # type: ignore[union-attr]

        # set params to best found
        bound2 = 2.0 * bound
        torch.mul(best_pos[ind], bound2, out=real_buf)  # type: ignore[arg-type, index]
        real_buf.sub_(bound)
        self._set_flat_params(real_buf)  # type: ignore[arg-type]

        return best_res[ind]
