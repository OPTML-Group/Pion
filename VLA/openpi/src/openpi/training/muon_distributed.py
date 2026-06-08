"""Flash-Muon-style distributed sharding for Muon / Pion Newton-Schulz iterations.

The vanilla MuonAdamW / DefaultPionAdamW.step() runs Newton-Schulz on EVERY
parameter on EVERY DDP rank — wasting (world_size - 1) / world_size of the
optimizer work, because after DDP all-reduce every rank already has identical
gradients.

This module reimplements the optimizer step in the spirit of
KellerJordan/Muon and nil0x9/flash-muon: each rank only computes NS on
~len(params) / world_size of the parameters; results are then exchanged via an
**async** ``all_gather_into_tensor`` that overlaps with the next rank's NS
compute. Net effect at world_size=8: ~7× speedup of the orthogonalization
portion of the optimizer step.

We also ship a vendored Triton ``matmul_transpose`` kernel (computes
``X @ X.T`` by skipping the strictly-lower-triangular blocks). The kernel
becomes the inner ``A = X @ X.T`` of the NS / Pion polynomials whenever
Triton + CUDA + bf16 are available; otherwise we transparently fall back to
the native ``X @ X.mT``.

References
----------
* https://github.com/KellerJordan/Muon (original Muon optimizer)
* https://github.com/nil0x9/flash-muon (Triton matmul_transpose kernel)

State / checkpoint semantics
----------------------------
Every rank still runs the momentum ``lerp_`` for every Muon parameter (cheap,
single fused op). Only the **owning** rank then runs the Newton-Schulz /
Pion polynomial. This keeps ``state["momentum_buffer"]`` identical across
ranks so rank-0 ``optimizer.state_dict()`` checkpoints remain correct.
"""

from __future__ import annotations

import collections
import math
from typing import Callable

import torch
import torch.distributed as dist

# ──────────────────────────────────────────────────────────────────────────────
# Vendored flash-muon Triton kernel for `X @ X.T`
#
# Source: https://github.com/nil0x9/flash-muon/blob/master/flash_muon/matmul_transpose_triton.py
# MIT-licensed. We vendor it here to avoid an extra pip dependency.
# Falls back to native `X @ X.mT` if Triton is unavailable or input is not
# 2D / bf16 / fp16 / cuda.
# ──────────────────────────────────────────────────────────────────────────────

try:  # pragma: no cover - import-time guard, exercised by smoke test
    import triton
    import triton.language as tl

    def _mmt_autotune_configs():
        return [
            triton.Config(
                {"BLOCK_SIZE_M": blk_m, "BLOCK_SIZE_K": blk_k, "GROUP_SIZE_M": grp},
                num_stages=ns,
                num_warps=nw,
            )
            for blk_m in (32, 64, 128)
            for blk_k in (32, 64)
            for grp in (8,)
            for ns in (3, 4, 5)
            for nw in (4, 8)
        ]

    @triton.autotune(configs=_mmt_autotune_configs(), key=["M", "K"])
    @triton.jit
    def _mmt_kernel(
        x, y,
        M, K,
        stride_xm, stride_xk,
        stride_ym, stride_yn,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        """Computes ``y = x @ x.T`` for 2D bf16 / fp16 input.

        Each program owns one (pid_m, pid_n) block of the output. Programs with
        ``pid_m > pid_n`` exit early; programs with ``pid_m < pid_n`` also write
        the transposed result into the lower-triangular slot — saving ~half
        the GEMM work versus a generic matmul.
        """
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        if pid_m > pid_n:
            return

        offs_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_xn = (pid_n * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = x + (offs_xm[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        b_ptrs = x + (offs_xn[:, None] * stride_xm + offs_k[None, :] * stride_xk)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_M), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, tl.permute(b, (1, 0)), accumulator)
            a_ptrs += BLOCK_SIZE_K * stride_xk
            b_ptrs += BLOCK_SIZE_K * stride_xk
        c = accumulator.to(x.dtype.element_ty)

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        c_ptrs = y + stride_ym * offs_cm[:, None] + stride_yn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
        tl.store(c_ptrs, c, mask=c_mask)

        if pid_m < pid_n:
            ct_ptrs = y + stride_ym * offs_cn[:, None] + stride_yn * offs_cm[None, :]
            ct_mask = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
            tl.store(ct_ptrs, tl.permute(c, (1, 0)), mask=ct_mask)

    _TRITON_AVAILABLE = True

except Exception:  # pragma: no cover
    _TRITON_AVAILABLE = False


# Minimum M for which we bother with the Triton kernel. Below this size the
# native matmul + cuBLAS dispatch wins (kernel overhead dominates). The
# flash-muon benchmark confirms the crossover is ~1024–2048 on A100/H100.
_TRITON_MIN_M = 256
# Triton requires K (inner dim) >= BLOCK_SIZE_K (smallest config is 32).
_TRITON_MIN_K = 32


def _triton_eligible(x: torch.Tensor) -> bool:
    """Return True if ``x @ x.T`` should use the vendored Triton kernel."""
    if not _TRITON_AVAILABLE:
        return False
    if not (x.is_cuda and x.ndim == 2):
        return False
    if x.dtype not in (torch.bfloat16, torch.float16):
        return False
    M, K = x.shape
    return M >= _TRITON_MIN_M and K >= _TRITON_MIN_K


def matmul_xxT(x: torch.Tensor) -> torch.Tensor:
    """Fast ``x @ x.T`` via vendored flash-muon Triton kernel (with fallback).

    For 3D+ tensors (batched per-head matmul) we just dispatch to the native
    ``x @ x.mT`` — the kernel only handles a single 2D matrix.
    """
    if x.ndim > 2 or not _triton_eligible(x):
        return x @ x.mT
    x_c = x.contiguous()
    M, K = x_c.shape
    out = torch.empty((M, M), device=x_c.device, dtype=x_c.dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) ** 2,)  # noqa: E731
    with torch.cuda.device(x_c.device.index):
        _mmt_kernel[grid](
            x_c, out, M, K,
            x_c.stride(0), x_c.stride(1),
            out.stride(0), out.stride(1),
        )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Polynomial kernels (Newton-Schulz, Pion) using fast matmul_xxT
# ──────────────────────────────────────────────────────────────────────────────


def fast_newton_schulz(X: torch.Tensor, ns_steps: int) -> torch.Tensor:
    """5-term Newton-Schulz polynomial, accelerated via matmul_xxT.

    Assumes ``X`` is already bf16, Frobenius-normalized, and oriented so
    ``rows <= cols``. Mirrors flash_muon.fast_newtonschulz exactly.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(ns_steps):
        A = matmul_xxT(X)
        B = A @ X
        X = a * X + b * B + c * (A @ B)
    return X


def fast_pion(X: torch.Tensor, promotion_steps: int, suppression_steps: int) -> torch.Tensor:
    """Pion high-pass NS polynomial (Promotion → Suppression),
    accelerated via matmul_xxT.

    Same shape contract as :func:`fast_newton_schulz`.
    """
    for _ in range(promotion_steps):
        A = matmul_xxT(X)
        B = -1.25 * A + 0.375 * (A @ A)
        X = 1.875 * X + B @ X
    for _ in range(suppression_steps):
        A = matmul_xxT(X)
        B = 2.5 * A - 1.5 * (A @ A)
        X = B @ X
    return X


# ──────────────────────────────────────────────────────────────────────────────
# Distributed sharded orthogonalization step
# ──────────────────────────────────────────────────────────────────────────────


def _flat_2d_shape(p: torch.Tensor) -> tuple[int, int]:
    """Match the (rows, cols) view used by `muon_update`/`pion_update`.

    ndim==2 → (rows, cols); ndim>=3 → (p.shape[0], numel/p.shape[0]).
    """
    if p.ndim >= 3:
        return p.shape[0], p.numel() // p.shape[0]
    return p.shape[-2], p.shape[-1]


def _orth_buffer_apply(
    p_w: torch.Tensor,
    g_flat: torch.Tensor,
    lr: float,
    wd: float,
    scale_factor: float,
) -> None:
    """Apply one (orthogonalized) update tensor to a single parameter."""
    if wd != 0:
        p_w.mul_(1 - lr * wd)
    rows, cols = _flat_2d_shape(p_w)
    scale = scale_factor * math.sqrt(max(rows, cols))
    p_w.add_(g_flat.view_as(p_w).to(p_w.dtype), alpha=-lr * scale)


def distributed_orth_step(
    state: dict,
    plist: list[torch.Tensor],
    *,
    lr: float,
    wd: float,
    momentum: float,
    nesterov: bool,
    scale_factor: float,
    poly_fn: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    """Flash-Muon-style distributed orthogonalization step.

    Steps:
      1. Every rank lerp-updates ``state[p]["momentum_buffer"]`` for every
         param in ``plist``. Cheap; keeps optimizer state in sync for
         checkpointing.
      2. Bucket params by ``numel`` so a fixed-size ``update_buffer`` can be
         reused for the all-gather.
      3. Within each bucket: rank ``r`` processes params at positions
         ``base_i + r``; an async ``all_gather_into_tensor`` overlaps with the
         next rank's polynomial compute; updates are applied to the local
         weights one bucket-row behind.

    Parameters
    ----------
    state : dict
        ``self.state`` of the calling Optimizer.
    plist : list[torch.Tensor]
        Params (with ``grad != None``) to be orthogonalized. The list must be
        identical across ranks; DDP guarantees this if all params receive
        gradients on all ranks.
    poly_fn : (X_bf16) -> X_bf16
        Polynomial applied to the (Frobenius-normalized, transposed-if-tall)
        gradient. Typically ``lambda X: fast_newton_schulz(X, ns_steps)`` or
        ``lambda X: fast_pion(X, promotion_steps, suppression_steps)``.
    """
    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError("distributed_orth_step requires an initialized process group")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = plist[0].device

    # Step 1: momentum lerp on every rank for every param. Cheap and keeps
    # `state["momentum_buffer"]` in sync across ranks (matters for checkpoints).
    for p in plist:
        st = state[p]
        if "momentum_buffer" not in st:
            st["momentum_buffer"] = torch.zeros_like(p)
        st["momentum_buffer"].lerp_(p.grad, 1 - momentum)
        st["step"] = st.get("step", 0) + 1

    # Step 2: bucket by numel so the all_gather buffer has a single shape.
    by_size: dict[int, list[torch.Tensor]] = collections.defaultdict(list)
    for p in plist:
        by_size[p.numel()].append(p)

    # Step 3: per-bucket sharded NS / Pion with async all_gather overlap.
    for numel, ps in by_size.items():
        update_buffer = torch.empty(world_size, numel, dtype=torch.bfloat16, device=device)
        views = [update_buffer[i] for i in range(world_size)]

        handle = None
        params_world: list[torch.Tensor] | None = None

        def _apply_prev() -> None:
            handle.wait()
            for p_w, g_w in zip(params_world, views):
                _orth_buffer_apply(p_w, g_w, lr=lr, wd=wd, scale_factor=scale_factor)

        for base_i in range(0, len(ps), world_size):
            if base_i + rank < len(ps):
                p = ps[base_i + rank]
                buf = state[p]["momentum_buffer"]
                # Build the post-momentum update; do NOT mutate p.grad / buf.
                u = p.grad.lerp(buf, momentum) if nesterov else buf.clone()
                if u.ndim >= 3:
                    u = u.reshape(u.shape[0], -1)
                X = u.to(torch.bfloat16)
                X = X / (X.norm() + 1e-7)
                transposed = X.size(-2) > X.size(-1)
                if transposed:
                    X = X.T
                X = poly_fn(X)
                if transposed:
                    X = X.T
                upd = X.reshape(-1).contiguous()
            else:
                upd = views[rank]
                upd.zero_()

            if base_i > 0:
                _apply_prev()
            handle = dist.all_gather_into_tensor(update_buffer, upd, async_op=True)
            params_world = ps[base_i : base_i + world_size]

        if params_world is not None:
            _apply_prev()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers callers use to decide which path to take
# ──────────────────────────────────────────────────────────────────────────────


def ddp_world_size() -> int:
    """Return DDP world size, or 1 if not initialized."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def should_use_distributed_orth(plist: list[torch.Tensor], min_params_per_rank: int = 1) -> bool:
    """Cheap gate: only use the sharded path if it actually helps.

    Below ``world_size * min_params_per_rank`` total params the all_gather
    overhead outweighs the compute savings.
    """
    ws = ddp_world_size()
    return ws > 1 and len(plist) >= ws * min_params_per_rank
