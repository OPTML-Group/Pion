"""AdamW-fused Muon / Pion optimizers used by the verl RLVR trainer.

This module exposes four ``torch.optim.Optimizer`` subclasses:

  * ``MuonAdamW``         -- standard Muon (Newton-Schulz orthogonalization)
                             on ndim >= 2, AdamW on ndim < 2.
  * ``DefaultPionAdamW``  -- the two-stage Promotion + Suppression
                             (high-pass NS) Pion iteration on ndim >= 2,
                             AdamW on ndim < 2.
  * ``PerHeadMuonAdamW``  -- per-attention-head Muon (head reshape on the
                             chosen axis) on ndim >= 2, AdamW on ndim < 2.
  * ``PerHeadPionAdamW``  -- per-attention-head Pion (high-pass NS) on
                             ndim >= 2, AdamW on ndim < 2.

The per-head classes are particularly useful for transformers with GQA
(Qwen3 family), where the whole-matrix scale would otherwise differ
between Q / K / V / O projections.  Splitting each weight into
``(num_heads, head_dim, in_dim)`` (or the transposed variant for O proj)
makes the per-step scale identical across all attention projections.
"""

from __future__ import annotations

import math
import torch
from torch import optim


# ---------------------------------------------------------------------------
# Shared AdamW step helper (used by all four classes on ndim < 2 params)
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
# Whole-matrix update functions (used by MuonAdamW / DefaultPionAdamW)
# ---------------------------------------------------------------------------

def muon_update(
    g: torch.Tensor,
    momentum_buffer: torch.Tensor,
    beta: float,
    nesterov: bool,
    ns_steps: int = 5,
    scale_factor: float = 0.2,
) -> torch.Tensor:
    """Whole-matrix Muon update with momentum (Newton-Schulz on the gram)."""
    momentum_buffer.lerp_(g, 1 - beta)
    g = g.lerp(momentum_buffer, beta) if nesterov else momentum_buffer.clone()

    orig_shape = g.shape
    if g.ndim >= 3:
        g = g.view(g.shape[0], -1)

    a, b, c = 3.4445, -4.7750, 2.0315
    X = g.bfloat16()
    X /= X.norm() + 1e-7

    rows, cols = X.shape[0], X.shape[1]
    transposed = rows > cols
    if transposed:
        X = X.T

    for _ in range(ns_steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T

    return (X.to(g.dtype) * (scale_factor * max(rows, cols) ** 0.5)).view(orig_shape)


def pion_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    beta: float = 0.95,
    nesterov: bool = False,
    promotion_steps: int = 0,
    suppression_steps: int = 5,
    scale_factor: float = 0.2,
) -> torch.Tensor:
    """Whole-matrix Pion (Promotion + Suppression) update with momentum.

    ``promotion_steps=0`` (pure Suppression) is the recommended RLVR / GRPO
    setting: Suppression retains the strongest gradient components while
    zeroing weak ones, preserving the natural off-principal gradient
    structure that RLVR relies on.  Promotion amplifies weak gradient
    directions first, confusing Suppression and degrading toward Muon-like
    full orthogonalization.
    """
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp(momentum, beta) if nesterov else momentum.clone()

    orig_shape = update.shape
    if update.ndim >= 3:
        update = update.view(update.shape[0], -1)

    X = update.bfloat16()
    X /= X.norm() + 1e-7

    rows, cols = X.shape[0], X.shape[1]
    transposed = rows > cols
    if transposed:
        X = X.T

    for _ in range(promotion_steps):
        A = X @ X.T
        B = -1.25 * A + 0.375 * (A @ A)
        X = 1.875 * X + B @ X

    for _ in range(suppression_steps):
        A = X @ X.T
        B = 2.5 * A - 1.5 * (A @ A)
        X = B @ X

    if transposed:
        X = X.T

    return (X.to(update.dtype) * (scale_factor * max(rows, cols) ** 0.5)).view(orig_shape)


# ---------------------------------------------------------------------------
# Per-head building blocks (used by PerHeadMuonAdamW / PerHeadPionAdamW)
# ---------------------------------------------------------------------------

def _batched_ns(X: torch.Tensor, ns_steps: int) -> torch.Tensor:
    """Newton-Schulz on batched tensor X: (..., rows, cols), rows <= cols."""
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(ns_steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X


def _batched_pion(
    X: torch.Tensor,
    promotion_steps: int,
    suppression_steps: int,
) -> torch.Tensor:
    """Pion (Promotion + Suppression) on batched X: (..., rows, cols), rows <= cols."""
    for _ in range(promotion_steps):
        A = X @ X.mT
        B = -1.25 * A + 0.375 * (A @ A)
        X = 1.875 * X + B @ X
    for _ in range(suppression_steps):
        A = X @ X.mT
        B = 2.5 * A - 1.5 * (A @ A)
        X = B @ X
    return X


def _multihead_ortho_2d(
    g: torch.Tensor,
    num_heads: int,
    head_split_dim: int,
    poly_fn,
    scale_factor: float = 0.2,
) -> torch.Tensor | None:
    """Per-head orthogonalization for a 2D gradient g: (out_dim, in_dim).

      head_split_dim=0  -> heads on OUTPUT side  (Q/K/V proj)
          g: (num_heads * head_dim, in_dim)  ->  (num_heads, head_dim, in_dim)
          scale uses max(head_dim, in_dim)
      head_split_dim=1  -> heads on INPUT side  (O proj)
          g: (out_dim, num_heads * head_dim)  ->  (num_heads, out_dim, head_dim)
          scale uses max(out_dim, head_dim)

    Returns ``None`` if ``num_heads`` does not divide the target dimension
    (the caller falls back to whole-matrix mode).

    Qwen3-4B (32 Q-heads, 8 KV-heads, head_dim=128, hidden=2560):
      Q (4096x2560, heads=32, dim=0): head shape (128, 2560), scale = 0.2*sqrt(2560)
      K (1024x2560, heads=8,  dim=0): head shape (128, 2560), scale = 0.2*sqrt(2560)
      O (2560x4096, heads=32, dim=1): head shape (2560, 128), scale = 0.2*sqrt(2560)
    All four projections receive identical scale -- GQA handled automatically.
    """
    out_dim, in_dim = g.shape

    if head_split_dim == 0:
        if out_dim % num_heads != 0:
            return None
        head_dim = out_dim // num_heads
        X = g.view(num_heads, head_dim, in_dim).bfloat16()
        head_rows, head_cols = head_dim, in_dim
    else:  # head_split_dim == 1
        if in_dim % num_heads != 0:
            return None
        head_dim = in_dim // num_heads
        X = g.view(out_dim, num_heads, head_dim).permute(1, 0, 2).bfloat16()
        head_rows, head_cols = out_dim, head_dim

    norms = X.norm(dim=(-2, -1), keepdim=True)
    X = X / (norms + 1e-7)

    transposed = head_rows > head_cols
    if transposed:
        X = X.mT

    X = poly_fn(X)

    if transposed:
        X = X.mT

    scale = scale_factor * max(head_rows, head_cols) ** 0.5
    X = X.to(g.dtype) * scale

    if head_split_dim == 0:
        return X.reshape(out_dim, in_dim)
    else:
        return X.permute(1, 0, 2).reshape(out_dim, in_dim)


def perhead_muon_update(
    g: torch.Tensor,
    momentum_buffer: torch.Tensor,
    beta: float,
    nesterov: bool,
    ns_steps: int,
    num_heads: int,
    head_split_dim: int = 0,
    scale_factor: float = 0.2,
) -> torch.Tensor:
    """Per-head Muon update. Falls back to whole-matrix if shape is incompatible."""
    momentum_buffer.lerp_(g, 1 - beta)
    update = g.lerp(momentum_buffer, beta) if nesterov else momentum_buffer.clone()

    if update.ndim == 2:
        result = _multihead_ortho_2d(
            update, num_heads, head_split_dim,
            poly_fn=lambda X: _batched_ns(X, ns_steps),
            scale_factor=scale_factor,
        )
        if result is not None:
            return result

    orig_shape = update.shape
    if update.ndim >= 3:
        update = update.view(update.shape[0], -1)
    a, b, c = 3.4445, -4.7750, 2.0315
    X = update.bfloat16()
    X /= X.norm() + 1e-7
    rows, cols = X.shape[0], X.shape[1]
    transposed = rows > cols
    if transposed:
        X = X.T
    for _ in range(ns_steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return (X.to(update.dtype) * (scale_factor * max(rows, cols) ** 0.5)).view(orig_shape)


def multihead_pion_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    beta: float,
    nesterov: bool,
    promotion_steps: int,
    suppression_steps: int,
    num_heads: int,
    head_split_dim: int = 0,
    scale_factor: float = 0.2,
) -> torch.Tensor:
    """Per-head Pion update. Falls back to whole-matrix if shape is incompatible."""
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp(momentum, beta) if nesterov else momentum.clone()

    if update.ndim == 2:
        result = _multihead_ortho_2d(
            update, num_heads, head_split_dim,
            poly_fn=lambda X: _batched_pion(X, promotion_steps, suppression_steps),
            scale_factor=scale_factor,
        )
        if result is not None:
            return result

    orig_shape = update.shape
    if update.ndim >= 3:
        update = update.view(update.shape[0], -1)
    X = update.bfloat16()
    X /= X.norm() + 1e-7
    rows, cols = X.shape[0], X.shape[1]
    transposed = rows > cols
    if transposed:
        X = X.T
    for _ in range(promotion_steps):
        A = X @ X.T
        B = -1.25 * A + 0.375 * (A @ A)
        X = 1.875 * X + B @ X
    for _ in range(suppression_steps):
        A = X @ X.T
        B = 2.5 * A - 1.5 * (A @ A)
        X = B @ X
    if transposed:
        X = X.T
    return (X.to(update.dtype) * (scale_factor * max(rows, cols) ** 0.5)).view(orig_shape)


# ---------------------------------------------------------------------------
# MuonAdamW  (whole-matrix Muon + AdamW)
# ---------------------------------------------------------------------------

class MuonAdamW(optim.Optimizer):
    """MuonAdamW: Muon on ndim >= 2 + AdamW on ndim < 2.

    Args:
        use_muon:      ``None`` (default) picks Muon vs AdamW automatically
                       by parameter ``ndim``.  Force the whole optimizer
                       onto one branch by passing ``True`` / ``False``, or
                       set it per-group via
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
        scale_factor: float = 5.0,
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
            scale_factor  = group.get("scale_factor", 5.0)
            group_use_muon = group.get("use_muon", None)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad  = p.grad
                state = self.state[p]
                use_muon = (p.ndim >= 2) if group_use_muon is None else bool(group_use_muon)

                if len(state) == 0:
                    state["step"] = 0
                    if use_muon:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    else:
                        state["exp_avg"]    = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1

                if use_muon:
                    update = muon_update(
                        grad, state["momentum_buffer"],
                        beta=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, scale_factor=scale_factor,
                    )
                    if wd != 0:
                        p.mul_(1 - lr * wd)
                    p.add_(update, alpha=-lr)
                else:
                    _adamw_step(p, grad, state, lr, wd, beta1, beta2, eps)

        return loss


# ---------------------------------------------------------------------------
# DefaultPionAdamW  (whole-matrix Pion + AdamW)
# ---------------------------------------------------------------------------

class DefaultPionAdamW(optim.Optimizer):
    """DefaultPionAdamW: Pion (high-pass NS) on ndim >= 2 + AdamW on ndim < 2.

    Args:
        promotion_steps: Promotion iterations ``k_p`` (default ``0``).
                         The polynomial runs ``k_p`` Promotion steps then
                         ``ns_steps - k_p`` Suppression steps.  Pure-Suppression
                         (``k_p=0``) is recommended for RLVR / GRPO: Suppression
                         keeps strong gradient components while zeroing weak
                         ones, preserving the off-principal update geometry.
        ns_steps:        Total NS iterations.
        use_muon:        ``None`` (default) picks Pion vs AdamW automatically
                         by parameter ``ndim``.
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
        scale_factor: float = 5.0,
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
            scale_factor  = group.get("scale_factor", 5.0)
            group_use_muon = group.get("use_muon", None)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad  = p.grad
                state = self.state[p]
                use_muon = (p.ndim >= 2) if group_use_muon is None else bool(group_use_muon)

                if len(state) == 0:
                    state["step"] = 0
                    if use_muon:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    else:
                        state["exp_avg"]    = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1

                if use_muon:
                    update = pion_update(
                        grad=grad, momentum=state["momentum_buffer"],
                        beta=group["momentum"], nesterov=group["nesterov"],
                        promotion_steps=promotion_steps,
                        suppression_steps=suppression_steps,
                        scale_factor=scale_factor,
                    )
                    if wd != 0:
                        p.mul_(1 - lr * wd)
                    p.add_(update, alpha=-lr)
                else:
                    _adamw_step(p, grad, state, lr, wd, beta1, beta2, eps)

        return loss


# ---------------------------------------------------------------------------
# PerHeadMuonAdamW  (per-attention-head Muon + AdamW)
# ---------------------------------------------------------------------------

class PerHeadMuonAdamW(optim.Optimizer):
    """PerHeadMuonAdamW: per-attention-head Muon + AdamW.

    Reshapes each 2D weight into ``(num_heads, head_dim, in_dim)`` (or the
    permuted variant for O-proj, ``head_split_dim=1``) before Newton-Schulz
    orthogonalization.  This fixes the inconsistent per-step scale that
    whole-matrix Muon produces for GQA attention projections (e.g. Qwen3
    family).

    Per-param-group settings:
      ``num_heads``       (int | None):  ``None`` falls back to whole-matrix Muon.
      ``head_split_dim``  (int, 0 or 1): 0 = Q/K/V (output side); 1 = O (input side).
      ``use_muon``        (bool | None): ``None`` auto-routes by ``ndim``.
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
        num_heads: int | None = None,
        head_split_dim: int = 0,
        scale_factor: float = 5.0,
        use_muon: bool | None = None,
    ):
        defaults = dict(
            lr=lr, momentum=momentum, betas=betas, eps=eps,
            weight_decay=weight_decay, nesterov=nesterov,
            ns_steps=ns_steps, num_heads=num_heads,
            head_split_dim=head_split_dim,
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
            momentum    = group["momentum"]
            nesterov    = group["nesterov"]
            ns_steps    = group["ns_steps"]
            num_heads   = group.get("num_heads", None)
            split_dim   = group.get("head_split_dim", 0)
            scale_factor  = group.get("scale_factor", 5.0)
            group_use_muon = group.get("use_muon", None)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad  = p.grad
                state = self.state[p]
                use_muon = (p.ndim >= 2) if group_use_muon is None else bool(group_use_muon)

                if len(state) == 0:
                    state["step"] = 0
                    if use_muon:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    else:
                        state["exp_avg"]    = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1

                if use_muon:
                    if num_heads is not None:
                        update = perhead_muon_update(
                            grad, state["momentum_buffer"],
                            beta=momentum, nesterov=nesterov,
                            ns_steps=ns_steps,
                            num_heads=num_heads, head_split_dim=split_dim,
                            scale_factor=scale_factor,
                        )
                    else:
                        update = muon_update(
                            grad, state["momentum_buffer"],
                            beta=momentum, nesterov=nesterov, ns_steps=ns_steps,
                            scale_factor=scale_factor,
                        )
                    if wd != 0:
                        p.mul_(1 - lr * wd)
                    p.add_(update, alpha=-lr)
                else:
                    _adamw_step(p, grad, state, lr, wd, beta1, beta2, eps)

        return loss


# ---------------------------------------------------------------------------
# PerHeadPionAdamW  (per-attention-head Pion + AdamW)
# ---------------------------------------------------------------------------

class PerHeadPionAdamW(optim.Optimizer):
    """PerHeadPionAdamW: per-attention-head Pion (high-pass NS) + AdamW.

    Combines per-head reshaping from :class:`PerHeadMuonAdamW` with the
    Pion polynomial (Promotion + Suppression) from
    :class:`DefaultPionAdamW`.  For RLVR / GRPO use ``promotion_steps=0``
    (pure Suppression, default): Suppression retains the strong per-head
    gradient components, while Promotion would amplify weak components
    first and degrade the per-head update toward Muon.

    Per-param-group settings:
      ``num_heads``, ``head_split_dim``, ``use_muon`` -- see
      :class:`PerHeadMuonAdamW`.
      ``promotion_steps`` (int, default 0): Promotion iterations.
      ``ns_steps``        (int, default 5): Total NS steps.
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
        num_heads: int | None = None,
        head_split_dim: int = 0,
        scale_factor: float = 5.0,
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
            num_heads=num_heads, head_split_dim=head_split_dim,
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
            num_heads   = group.get("num_heads", None)
            split_dim   = group.get("head_split_dim", 0)
            scale_factor  = group.get("scale_factor", 5.0)
            group_use_muon = group.get("use_muon", None)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad  = p.grad
                state = self.state[p]
                use_muon = (p.ndim >= 2) if group_use_muon is None else bool(group_use_muon)

                if len(state) == 0:
                    state["step"] = 0
                    if use_muon:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    else:
                        state["exp_avg"]    = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1

                if use_muon:
                    if num_heads is not None:
                        update = multihead_pion_update(
                            grad, state["momentum_buffer"],
                            beta=group["momentum"], nesterov=group["nesterov"],
                            promotion_steps=promotion_steps,
                            suppression_steps=suppression_steps,
                            num_heads=num_heads, head_split_dim=split_dim,
                            scale_factor=scale_factor,
                        )
                    else:
                        update = pion_update(
                            grad=grad, momentum=state["momentum_buffer"],
                            beta=group["momentum"], nesterov=group["nesterov"],
                            promotion_steps=promotion_steps,
                            suppression_steps=suppression_steps,
                            scale_factor=scale_factor,
                        )
                    if wd != 0:
                        p.mul_(1 - lr * wd)
                    p.add_(update, alpha=-lr)
                else:
                    _adamw_step(p, grad, state, lr, wd, beta1, beta2, eps)

        return loss
