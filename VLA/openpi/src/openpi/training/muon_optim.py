"""AdamW-fused Muon / Pion optimizers used by the openpi (π0.5) trainer.

This module exposes two ``torch.optim.Optimizer`` subclasses:

  * ``MuonAdamW``         -- standard Muon (Newton-Schulz orthogonalization)
                             on 2D+ parameters, with AdamW automatically
                             dispatched on 0/1-D parameters.
  * ``DefaultPionAdamW``  -- the two-stage Promotion + Suppression
                             (high-pass NS) Pion iteration on 2D+ parameters,
                             with AdamW on 0/1-D parameters.

Both classes split each parameter group on ``ndim`` at ``step`` time:
``ndim >= 2`` -> the Muon / Pion polynomial; ``ndim < 2`` -> a per-element
AdamW update.  The expensive matrix polynomial is sharded across DDP
ranks via :mod:`openpi.training.muon_distributed`; the AdamW branch runs
identically on every rank.
"""

from __future__ import annotations

import math
import torch
from torch import optim

from openpi.training import muon_distributed as _muon_dist


# ---------------------------------------------------------------------------
# Shared AdamW step (used by both optimizers on 0/1-D parameters)
# ---------------------------------------------------------------------------

def _adamw_step(
    p: torch.Tensor,
    grad: torch.Tensor,
    state: dict,
    lr: float,
    wd: float,
    beta1: float,
    beta2: float,
    eps: float,
) -> None:
    if wd != 0:
        p.mul_(1 - lr * wd)
    exp_avg    = state["exp_avg"]
    exp_avg_sq = state["exp_avg_sq"]
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    bias1 = 1 - beta1 ** state["step"]
    bias2 = 1 - beta2 ** state["step"]
    denom = (exp_avg_sq.sqrt() / math.sqrt(bias2)).add_(eps)
    p.addcdiv_(exp_avg, denom, value=-(lr / bias1))


# ---------------------------------------------------------------------------
# Single-GPU fallback polynomials (DDP path uses muon_distributed.fast_*)
# ---------------------------------------------------------------------------

def muon_update(
    g: torch.Tensor,
    momentum_buffer: torch.Tensor,
    beta: float,
    nesterov: bool,
    ns_steps: int = 5,
    scale_factor: float = 0.2,
) -> torch.Tensor:
    """Whole-matrix Muon update (single-GPU fallback)."""
    momentum_buffer.lerp_(g, 1 - beta)
    g = g.lerp(momentum_buffer, beta) if nesterov else momentum_buffer.clone()
    orig_shape = g.shape
    if g.ndim == 4:
        g = g.view(len(g), -1)
    rows, cols = g.shape
    update = _muon_dist.fast_newton_schulz(g, ns_steps)
    return (update.to(g.dtype) * (scale_factor * max(rows, cols) ** 0.5)).view(orig_shape)


def pion_update(
    g: torch.Tensor,
    momentum_buffer: torch.Tensor,
    beta: float,
    nesterov: bool,
    promotion_steps: int,
    suppression_steps: int,
    scale_factor: float = 0.2,
) -> torch.Tensor:
    """Whole-matrix Pion update (single-GPU fallback)."""
    momentum_buffer.lerp_(g, 1 - beta)
    g = g.lerp(momentum_buffer, beta) if nesterov else momentum_buffer.clone()
    orig_shape = g.shape
    if g.ndim == 4:
        g = g.view(len(g), -1)
    rows, cols = g.shape
    update = _muon_dist.fast_pion(g, promotion_steps, suppression_steps)
    return (update.to(g.dtype) * (scale_factor * max(rows, cols) ** 0.5)).view(orig_shape)


# ---------------------------------------------------------------------------
# MuonAdamW
# ---------------------------------------------------------------------------

class MuonAdamW(optim.Optimizer):
    """MuonAdamW: Muon on ndim >= 2 + AdamW on ndim < 2.

    Args:
        use_muon:      ``None`` (default) picks Muon vs AdamW automatically
                       by parameter ``ndim``.  Set ``True`` / ``False`` to
                       force the whole optimizer onto a single branch, or
                       set it per group via
                       ``optimizer.param_groups[i]["use_muon"]``.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        nesterov: bool = False,
        ns_steps: int = 5,
        scale_factor: float = 2.0,
        use_muon: bool | None = None,
    ):
        defaults = dict(
            lr=lr, momentum=momentum, betas=betas, eps=eps,
            weight_decay=weight_decay, nesterov=nesterov,
            ns_steps=ns_steps, scale_factor=scale_factor, use_muon=use_muon,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr          = group["lr"]
            wd          = group["weight_decay"]
            beta1, beta2= group["betas"]
            eps         = group["eps"]
            momentum    = group["momentum"]
            nesterov    = group["nesterov"]
            ns_steps    = group["ns_steps"]
            scale_factor   = group.get("scale_factor", 0.2)
            group_use_muon = group.get("use_muon", None)

            muon_params: list[torch.Tensor] = []
            adamw_params: list[torch.Tensor] = []
            for p in group["params"]:
                if p.grad is None:
                    continue
                use_muon = (p.ndim >= 2) if group_use_muon is None else bool(group_use_muon)
                (muon_params if use_muon else adamw_params).append(p)

            for p in adamw_params:
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"]    = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                _adamw_step(p, p.grad, state, lr, wd, beta1, beta2, eps)

            if not muon_params:
                continue

            if _muon_dist.should_use_distributed_orth(muon_params):
                _muon_dist.distributed_orth_step(
                    self.state,
                    muon_params,
                    lr=lr, wd=wd, momentum=momentum, nesterov=nesterov,
                    scale_factor=scale_factor,
                    poly_fn=lambda X, _k=ns_steps: _muon_dist.fast_newton_schulz(X, _k),
                )
            else:
                for p in muon_params:
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["momentum_buffer"] = torch.zeros_like(p)
                    state["step"] += 1
                    update = muon_update(
                        p.grad, state["momentum_buffer"],
                        beta=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, scale_factor=scale_factor,
                    )
                    if wd != 0:
                        p.mul_(1 - lr * wd)
                    p.add_(update, alpha=-lr)

        return loss


# ---------------------------------------------------------------------------
# DefaultPionAdamW
# ---------------------------------------------------------------------------

class DefaultPionAdamW(optim.Optimizer):
    """DefaultPionAdamW: Pion (high-pass NS) on ndim >= 2 + AdamW on ndim < 2.

    Args:
        promotion_steps: Promotion iterations ``k_p`` (default ``0``).
                         The Pion polynomial first runs ``k_p`` Promotion
                         steps, then ``ns_steps - k_p`` Suppression steps.
                         Pure-Suppression (``k_p=0``) is the recommended
                         RLVR / VLA setting.
        ns_steps:        Total NS iterations.  Suppression iterations
                         ``k_s = ns_steps - promotion_steps``.
        use_muon:        ``None`` (default) picks Pion vs AdamW
                         automatically by parameter ``ndim``.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        nesterov: bool = False,
        ns_steps: int = 5,
        promotion_steps: int = 0,
        scale_factor: float = 2.0,
        use_muon: bool | None = None,
    ):
        if promotion_steps > ns_steps:
            raise ValueError(
                f"promotion_steps ({promotion_steps}) must be <= ns_steps ({ns_steps})"
            )
        defaults = dict(
            lr=lr, momentum=momentum, betas=betas, eps=eps,
            weight_decay=weight_decay, nesterov=nesterov,
            ns_steps=ns_steps, promotion_steps=promotion_steps,
            scale_factor=scale_factor, use_muon=use_muon,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr          = group["lr"]
            wd          = group["weight_decay"]
            beta1, beta2= group["betas"]
            eps         = group["eps"]
            ns_steps        = group["ns_steps"]
            promotion_steps = min(group["promotion_steps"], ns_steps)
            suppression_steps = ns_steps - promotion_steps
            momentum    = group["momentum"]
            nesterov    = group["nesterov"]
            scale_factor   = group.get("scale_factor", 0.2)
            group_use_muon = group.get("use_muon", None)

            pion_params: list[torch.Tensor] = []
            adamw_params: list[torch.Tensor] = []
            for p in group["params"]:
                if p.grad is None:
                    continue
                use_muon = (p.ndim >= 2) if group_use_muon is None else bool(group_use_muon)
                (pion_params if use_muon else adamw_params).append(p)

            for p in adamw_params:
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"]    = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                _adamw_step(p, p.grad, state, lr, wd, beta1, beta2, eps)

            if not pion_params:
                continue

            if _muon_dist.should_use_distributed_orth(pion_params):
                _muon_dist.distributed_orth_step(
                    self.state,
                    pion_params,
                    lr=lr, wd=wd, momentum=momentum, nesterov=nesterov,
                    scale_factor=scale_factor,
                    poly_fn=lambda X, _p=promotion_steps, _s=suppression_steps:
                        _muon_dist.fast_pion(X, _p, _s),
                )
            else:
                for p in pion_params:
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["momentum_buffer"] = torch.zeros_like(p)
                    state["step"] += 1
                    update = pion_update(
                        g=p.grad, momentum_buffer=state["momentum_buffer"],
                        beta=momentum, nesterov=nesterov,
                        promotion_steps=promotion_steps,
                        suppression_steps=suppression_steps,
                        scale_factor=scale_factor,
                    )
                    if wd != 0:
                        p.mul_(1 - lr * wd)
                    p.add_(update, alpha=-lr)

        return loss
