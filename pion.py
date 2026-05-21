"""Pion: sPectral hIgh-pass Optimization on momeNtum.

This module collects the four optimizers used in the paper

    Rethinking Muon Beyond Pretraining:
    Spectral Failures and High-Pass Remedies for VLA and RLVR

  * ``Muon``         -- the original Muon baseline (Newton-Schulz orthogonalization).
  * ``DefaultPion``  -- Pion with a two-stage *Promotion + Suppression*
                        (high-pass NS) iteration applied to the whole matrix.
  * ``PerHeadPion``  -- Pion applied independently to each attention head
                        through a reshape, preserving per-head specialization.
  * ``LowRankMuon``  -- Muon variant that uses exact SVD to project the update
                        onto the top-``k`` singular subspace before
                        orthogonalization.

All optimizers use the same distributed async ``all_gather`` skeleton as
``Muon``: every parameter group is sharded across ranks and updates are
collected with ``all_gather_into_tensor`` while the next shard is being
computed.  Pass ``rank=0, world_size=1`` for single-GPU use.
"""

import torch
import torch.distributed as dist
from torch import Tensor


# ---------------------------------------------------------------------------
#  Newton-Schulz orthogonalization (used by Muon and LowRankMuon)
# ---------------------------------------------------------------------------


def fast_newtonschulz(G: Tensor, steps: int = 5, epsilon: float = 1e-7) -> Tensor:
    """Standard Newton-Schulz iteration with the (3.4445, -4.7750, 2.0315)
    quintic polynomial.  Operates in bfloat16 and returns a bfloat16 tensor
    whose singular values are pushed toward 1.
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + epsilon)
    if X.size(-2) > X.size(-1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(-2) > G.size(-1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Muon -- MomentUm Orthogonalized by Newton-Schulz.

    Adapted from https://github.com/KellerJordan/Muon/blob/master/muon.py.

    Internally runs SGD-momentum, then replaces each 2D update with the
    nearest orthogonal matrix via :func:`fast_newtonschulz`.

    Notes:
      * Do not use this on embedding / final FC / 0-1D parameters; route
        those to AdamW instead.
      * For 4D convolutional filters, flatten the last 3 dimensions.

    Arguments:
        ns_steps:      Number of Newton-Schulz iterations.
        scale_factor:  Update-RMS scaling factor.  The per-step scale is
                       ``scale_factor * sqrt(max(rows, cols))``.
    """

    def __init__(
        self,
        params,
        lr=0.02,
        weight_decay=0.01,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        scale_factor=2.0,
        rank=None,
        world_size=None,
    ):
        if (rank is None) or (world_size is None):
            raise Exception(
                "world_size and rank params required. For single-GPU use, "
                "pass rank=0 and world_size=1."
            )
        self.rank = rank
        self.world_size = world_size
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            scale_factor=scale_factor,
        )
        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
            group = dict(
                params=[p for p in params if p.numel() == size],
                update_buffer=b,
                update_buffer_views=[b[i] for i in range(world_size)],
            )
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            update_buffer: Tensor = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            params: list[Tensor] = group["params"]
            handle = None
            params_world = None

            scale_factor = group["scale_factor"]

            def update_prev():
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.mul_(1 - group["lr"] * group["weight_decay"])
                    p_world.add_(
                        g_world.view_as(p_world),
                        alpha=-group["lr"] * scale_factor * max(p_world.size(-2), p_world.size(-1)) ** 0.5,
                    )

            for base_i in range(len(params))[:: self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    if g is None:
                        g = update_buffer_views[self.rank]
                        g.zero_()
                    else:
                        state = self.state[p]
                        if "momentum_buffer" not in state:
                            state["momentum_buffer"] = torch.zeros_like(g)
                        buf: Tensor = state["momentum_buffer"]
                        buf.lerp_(g, 1 - group["momentum"])
                        g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                        if g.ndim == 4:  # conv
                            g = g.view(len(g), -1)
                        g = fast_newtonschulz(g, steps=group["ns_steps"]).flatten()
                else:
                    g = update_buffer_views[self.rank]
                    g.zero_()

                if base_i > 0:
                    update_prev()
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]

            if params_world is not None:
                update_prev()


# ---------------------------------------------------------------------------
#  Pion: sPectral hIgh-pass Optimization on momeNtum
# ---------------------------------------------------------------------------


def high_pass_ns(
    X: Tensor,
    promotion_steps: int,
    suppression_steps: int,
    epsilon: float = 1e-7,
) -> Tensor:
    """High-pass Newton-Schulz iteration used by Pion.

    Two stages are applied to ``X`` (which may be 2D or batched 3D+):

      1. **Promotion** (``promotion_steps`` iters, 5th-degree optimal
         polynomial ``a=1.875, b=-1.25, c=0.375``) -- amplifies all singular
         directions so that dominant components are quickly pushed toward 1.
      2. **Suppression** (``suppression_steps`` iters, ``a=0, b=2.5,
         c=-1.5``) -- a sharp high-pass filter that anchors singular values
         near 1 and squashes noisy tail components toward 0.

    Each (rows, cols) sub-matrix is independently Frobenius-normalized.
    The smaller Gram matrix is always used in the polynomial loop by
    transposing when rows > cols.
    """
    X = X.bfloat16()
    if X.ndim > 2:
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + epsilon)
    else:
        X = X / (X.norm() + epsilon)

    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.transpose(-2, -1)

    for _ in range(promotion_steps):
        A = X @ X.transpose(-2, -1)
        B = -1.25 * A + 0.375 * (A @ A)
        X = 1.875 * X + B @ X

    for _ in range(suppression_steps):
        A = X @ X.transpose(-2, -1)
        B = 2.5 * A - 1.5 * (A @ A)
        X = B @ X

    if transposed:
        X = X.transpose(-2, -1)

    return X


class DefaultPion(torch.optim.Optimizer):
    """DefaultPion -- Pion applied to the whole 2D parameter matrix.

    Drop-in replacement for :class:`Muon`.  The Newton-Schulz iteration
    used inside the Muon step is replaced by the high-pass NS iteration
    (Promotion + Suppression), giving a sharp spectral high-pass effect.

    Arguments:
        promotion_steps:   Number of Promotion iterations ``k_p``.
        ns_steps:          Total NS iterations ``k_p + k_s``.  Suppression
                           iterations are ``max(0, ns_steps - promotion_steps)``.
        scale_factor:      Update-RMS scaling factor.  The per-step scale is
                           ``scale_factor * sqrt(max(rows, cols))``.
    """

    def __init__(
        self,
        params,
        lr=0.02,
        weight_decay=0.01,
        momentum=0.95,
        nesterov=True,
        promotion_steps=0,
        ns_steps=5,
        scale_factor=2.0,
        rank=None,
        world_size=None,
    ):
        if (rank is None) or (world_size is None):
            raise Exception(
                "world_size and rank params required. For single-GPU use, "
                "pass rank=0 and world_size=1."
            )
        self.rank = rank
        self.world_size = world_size
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            promotion_steps=promotion_steps,
            ns_steps=ns_steps,
            scale_factor=scale_factor,
        )
        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
            group = dict(
                params=[p for p in params if p.numel() == size],
                update_buffer=b,
                update_buffer_views=[b[i] for i in range(world_size)],
            )
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            update_buffer: Tensor = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            params: list[Tensor] = group["params"]
            handle = None
            params_world = None

            promotion_steps = group["promotion_steps"]
            suppression_steps = max(0, group["ns_steps"] - promotion_steps)
            scale_factor = group["scale_factor"]

            def update_prev():
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.mul_(1 - group["lr"] * group["weight_decay"])
                    p_world.add_(
                        g_world.view_as(p_world),
                        alpha=-group["lr"] * scale_factor * max(p_world.size(-2), p_world.size(-1)) ** 0.5,
                    )

            for base_i in range(len(params))[:: self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    if g is None:
                        g = update_buffer_views[self.rank]
                        g.zero_()
                    else:
                        state = self.state[p]
                        if "momentum_buffer" not in state:
                            state["momentum_buffer"] = torch.zeros_like(g)
                        buf: Tensor = state["momentum_buffer"]
                        buf.lerp_(g, 1 - group["momentum"])
                        g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                        if g.ndim == 4:  # conv
                            g = g.view(len(g), -1)
                        g = high_pass_ns(g, promotion_steps, suppression_steps).flatten()
                else:
                    g = update_buffer_views[self.rank]
                    g.zero_()

                if base_i > 0:
                    update_prev()
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]

            if params_world is not None:
                update_prev()


class PerHeadPion(torch.optim.Optimizer):
    """PerHeadPion -- Pion applied independently to each attention head.

    For 2D weights belonging to attention projections, the matrix is
    reshaped into ``(num_heads, head_dim_a, head_dim_b)`` and the
    high-pass NS iteration is applied per head.  This preserves the
    per-head heterogeneity inherited from prior training, which Muon's
    global whitening tends to destroy.

      * ``head_split_dim=0``  -> heads on the OUTPUT side (Q/K/V proj),
        ``(num_heads*head_dim, in_dim)`` viewed as
        ``(num_heads, head_dim, in_dim)``; per-head scale uses
        ``max(head_dim, in_dim)``.
      * ``head_split_dim=1``  -> heads on the INPUT side (O proj),
        ``(out_dim, num_heads*head_dim)`` reshaped+permuted to
        ``(num_heads, out_dim, head_dim)``; per-head scale uses
        ``max(out_dim, head_dim)``.

    If ``num_heads`` does not divide the target axis, this transparently
    falls back to the whole-matrix update.

    Arguments:
        promotion_steps:   Number of Promotion iterations ``k_p`` (default 0).
        ns_steps:          Total NS iterations ``k_p + k_s``. Suppression
                           iterations are ``max(0, ns_steps - promotion_steps)``.
        scale_factor:      Update-RMS scaling factor.  Per-head scale is
                           ``scale_factor * sqrt(max(head_rows, head_cols))``.
    """

    def __init__(
        self,
        params,
        lr=0.02,
        weight_decay=0.01,
        momentum=0.95,
        nesterov=True,
        promotion_steps=0,
        ns_steps=5,
        scale_factor=2.0,
        rank=None,
        world_size=None,
        num_heads=None,
        head_split_dim=0,
    ):
        if (rank is None) or (world_size is None):
            raise Exception(
                "world_size and rank params required. For single-GPU use, "
                "pass rank=0 and world_size=1."
            )
        if num_heads is None:
            raise ValueError("For PerHeadPion, num_heads must be provided.")
        if head_split_dim not in (0, 1):
            raise ValueError("head_split_dim must be 0 (Q/K/V) or 1 (O proj).")

        self.rank = rank
        self.world_size = world_size
        self.num_heads = int(num_heads)
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            promotion_steps=int(promotion_steps),
            ns_steps=int(ns_steps),
            scale_factor=scale_factor,
            head_split_dim=int(head_split_dim),
        )
        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
            group = dict(
                params=[p for p in params if p.numel() == size],
                update_buffer=b,
                update_buffer_views=[b[i] for i in range(world_size)],
            )
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            update_buffer: Tensor = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            params: list[Tensor] = group["params"]
            handle = None
            params_world = None

            promotion_steps = group["promotion_steps"]
            suppression_steps = max(0, group["ns_steps"] - promotion_steps)
            scale_factor = group["scale_factor"]
            head_split_dim = group["head_split_dim"]
            num_heads = self.num_heads

            def update_prev():
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.mul_(1 - group["lr"] * group["weight_decay"])
                    p_world.add_(g_world.view_as(p_world), alpha=-group["lr"])

            for base_i in range(len(params))[:: self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    if g is None:
                        g = update_buffer_views[self.rank]
                        g.zero_()
                    else:
                        state = self.state[p]
                        if "momentum_buffer" not in state:
                            state["momentum_buffer"] = torch.zeros_like(g)
                        buf: Tensor = state["momentum_buffer"]
                        buf.lerp_(g, 1 - group["momentum"])
                        g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf

                        if g.ndim == 4:
                            g_mat = g.view(len(g), -1)
                            g_ortho = high_pass_ns(g_mat, promotion_steps, suppression_steps)
                            g_ortho = g_ortho.to(g_mat.dtype)
                            g_ortho.mul_(scale_factor * max(g_ortho.size(-2), g_ortho.size(-1)) ** 0.5)
                            g = g_ortho.flatten()

                        elif g.ndim == 2:
                            out_dim, in_dim = g.shape
                            if head_split_dim == 0 and out_dim % num_heads == 0:
                                head_dim = out_dim // num_heads
                                g_view = g.view(num_heads, head_dim, in_dim)
                                g_ortho = high_pass_ns(g_view, promotion_steps, suppression_steps)
                                g_ortho = g_ortho.to(g.dtype)
                                g_ortho.mul_(scale_factor * max(head_dim, in_dim) ** 0.5)
                                g = g_ortho.reshape(out_dim, in_dim).flatten()
                            elif head_split_dim == 1 and in_dim % num_heads == 0:
                                head_dim = in_dim // num_heads
                                g_view = (
                                    g.view(out_dim, num_heads, head_dim)
                                     .permute(1, 0, 2)
                                     .contiguous()
                                )
                                g_ortho = high_pass_ns(g_view, promotion_steps, suppression_steps)
                                g_ortho = g_ortho.to(g.dtype)
                                g_ortho.mul_(scale_factor * max(out_dim, head_dim) ** 0.5)
                                g = (
                                    g_ortho.permute(1, 0, 2)
                                           .reshape(out_dim, in_dim)
                                           .flatten()
                                )
                            else:
                                g_ortho = high_pass_ns(g, promotion_steps, suppression_steps)
                                g_ortho = g_ortho.to(g.dtype)
                                g_ortho.mul_(scale_factor * max(out_dim, in_dim) ** 0.5)
                                g = g_ortho.flatten()

                        else:
                            g_mat = g.view(1, -1)
                            g_ortho = high_pass_ns(g_mat, promotion_steps, suppression_steps)
                            g_ortho = g_ortho.to(g.dtype)
                            g_ortho.mul_(scale_factor * max(g_mat.size(-2), g_mat.size(-1)) ** 0.5)
                            g = g_ortho.flatten()
                else:
                    g = update_buffer_views[self.rank]
                    g.zero_()

                if base_i > 0:
                    update_prev()
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]

            if params_world is not None:
                update_prev()


# ---------------------------------------------------------------------------
#  LowRankMuon -- Muon with exact-SVD low-rank truncation
# ---------------------------------------------------------------------------


class LowRankMuon(torch.optim.Optimizer):
    """LowRankMuon -- MomentUm Orthogonalized by exact SVD truncation.

    Runs standard SGD-momentum, then replaces the Newton-Schulz step with
    an exact SVD decomposition ``G = U S V^T``, keeps the top ``rank_k``
    singular directions, sets the kept singular values to 1, and uses
    ``U_k V_k^T`` as the update.  Useful as a clean reference for what
    "perfect low-rank orthogonalization" looks like; see the paper for
    discussion.

    Arguments:
        rank_k:        Number of top singular directions to keep.
        scale_factor:  Update-RMS scaling factor.  The per-step scale is
                       ``scale_factor * sqrt(max(rows, cols))``.
    """

    def __init__(
        self,
        params,
        lr=0.02,
        weight_decay=0.01,
        momentum=0.95,
        nesterov=True,
        rank_k=128,
        scale_factor=2.0,
        rank=None,
        world_size=None,
    ):
        if (rank is None) or (world_size is None):
            raise Exception(
                "world_size and rank params required. For single-GPU use, "
                "pass rank=0 and world_size=1."
            )
        self.rank = rank
        self.world_size = world_size

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            rank_k=rank_k,
            scale_factor=scale_factor,
        )

        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
            group = dict(
                params=[p for p in params if p.numel() == size],
                update_buffer=b,
                update_buffer_views=[b[i] for i in range(world_size)],
            )
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            update_buffer: Tensor = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            params: list[Tensor] = group["params"]
            handle = None
            params_world = None
            scale_factor = group["scale_factor"]

            def update_prev():
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.mul_(1 - group["lr"] * group["weight_decay"])
                    p_world.add_(
                        g_world.view_as(p_world),
                        alpha=-group["lr"] * scale_factor * max(p_world.size(-2), p_world.size(-1)) ** 0.5,
                    )

            for base_i in range(len(params))[:: self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    if g is None:
                        g = update_buffer_views[self.rank]
                        g.zero_()
                    else:
                        state = self.state[p]
                        if "momentum_buffer" not in state:
                            state["momentum_buffer"] = torch.zeros_like(g)
                        buf: Tensor = state["momentum_buffer"]
                        buf.lerp_(g, 1 - group["momentum"])
                        g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf

                        if g.ndim == 4:  # conv
                            g = g.view(len(g), -1)

                        rows, cols = g.shape
                        k_eff = int(max(1, min(group["rank_k"], rows, cols)))

                        # fp32 SVD for numerical stability.
                        U, S, Vh = torch.linalg.svd(g.to(torch.float32), full_matrices=False)
                        U_k = U[:, :k_eff]
                        Vh_k = Vh[:k_eff, :]
                        g_ortho = U_k @ Vh_k
                        g = g_ortho.to(g.dtype).flatten()
                else:
                    g = update_buffer_views[self.rank]
                    g.zero_()

                if base_i > 0:
                    update_prev()
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]

            if params_world is not None:
                update_prev()
