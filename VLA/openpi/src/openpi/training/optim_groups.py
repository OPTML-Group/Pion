"""Parameter-group routing for PyTorch fine-tuning of pi0/pi05.

Given an optimizer name in {"adamw", "muon", "pion"}, this module returns a
LIST of torch.optim.Optimizer instances that together cover *all* trainable
parameters exactly once. Callers iterate the list and call `.step()` /
`.zero_grad()` / `.state_dict()` on each — no wrapper class is needed.

Routing rules:
  - adamw : everything → one AdamW.
  - muon  : ndim>=2 weights that are not embeddings/output-heads → Muon;
            the rest (1-D params, embeddings, output heads) → AdamW.
  - pion : VL tower (vision + language) ndim>=2 non-embed/output → Muon;
            action-expert tower ndim>=2 non-embed/output → Pion;
            everything else → AdamW.

Use `classify_parameters(model, name)` for the pure routing dict that
backs `build_optimizers`.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

import torch
from torch import nn

from openpi.training import muon_optim as _muon_optim


OptimizerName = Literal["adamw", "muon", "pion"]


# ──────────────────────────────────────────────────────────────────────────────
# Parameter classification
# ──────────────────────────────────────────────────────────────────────────────


_EMBED_HINTS = ("embed_tokens", "embeddings.weight", "position_embedding", "patch_embedding")
_OUTPUT_HINTS = ("lm_head", "action_out_proj")


def _is_embed_or_output(name: str) -> bool:
    return any(h in name for h in _EMBED_HINTS) or any(h in name for h in _OUTPUT_HINTS)


def _is_action_tower(name: str) -> bool:
    """True if this parameter belongs to the action expert tower of pi0/pi05.

    Action tower parameters live either inside `paligemma_with_expert.gemma_expert.*`
    or at the PI0Pytorch root as `action_in_proj`, `action_out_proj`, `state_proj`,
    `action_time_mlp_*`, `time_mlp_*`.
    """
    if "paligemma_with_expert.gemma_expert" in name:
        return True
    n = name.removeprefix("module.")  # strip DDP wrapper prefix if present
    return n.startswith(
        (
            "action_in_proj",
            "action_out_proj",
            "state_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
            "time_mlp_in",
            "time_mlp_out",
        )
    )


def _is_vl_tower(name: str) -> bool:
    """True if this parameter belongs to the VL (vision + language) tower."""
    return "paligemma_with_expert.paligemma" in name


def _named_trainable(model: nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    for name, p in model.named_parameters():
        if p.requires_grad:
            yield name, p


# ──────────────────────────────────────────────────────────────────────────────
# Public classification API (used by build_optimizers)
# ──────────────────────────────────────────────────────────────────────────────


# Possible bucket labels. Each label maps directly to one sub-optimizer in
# build_optimizers (see the routing table below).
BucketLabel = Literal["muon", "pion", "adamw"]


def classify_parameters(
    model: nn.Module, name: OptimizerName
) -> dict[BucketLabel, list[tuple[str, torch.Tensor]]]:
    """Return per-bucket lists of (param_name, param) covering all trainable params.

    The returned dict always has the three keys 'muon', 'pion', 'adamw' (some
    of them empty depending on the scheme). Buckets are exclusive: every
    trainable parameter appears in exactly one bucket.
    """
    name_l = name.lower()
    buckets: dict[BucketLabel, list[tuple[str, torch.Tensor]]] = {
        "muon": [],
        "pion": [],
        "adamw": [],
    }

    if name_l == "adamw":
        for n, p in _named_trainable(model):
            buckets["adamw"].append((n, p))
        return buckets

    if name_l == "muon":
        for n, p in _named_trainable(model):
            if p.ndim >= 2 and not _is_embed_or_output(n):
                buckets["muon"].append((n, p))
            else:
                buckets["adamw"].append((n, p))
        return buckets

    if name_l == "pion":
        for n, p in _named_trainable(model):
            if p.ndim < 2 or _is_embed_or_output(n):
                buckets["adamw"].append((n, p))
            elif _is_action_tower(n):
                buckets["pion"].append((n, p))
            elif _is_vl_tower(n):
                buckets["muon"].append((n, p))
            else:
                buckets["adamw"].append((n, p))
        return buckets

    raise ValueError(f"Unknown optimizer name '{name}'. Expected adamw / muon / pion.")


def _bucket_human_label(scheme: str, bucket: BucketLabel) -> str:
    """Human-readable description of what a bucket contains under a scheme."""
    s = scheme.lower()
    if s == "adamw":
        return "AdamW (all trainable)"
    if s == "muon":
        return {
            "muon": "Muon (ndim>=2 non-embed/output)",
            "adamw": "AdamW (1-D, embeddings, output heads)",
            "pion": "(unused)",
        }[bucket]
    if s == "pion":
        return {
            "muon": "Muon (VL tower ndim>=2 non-embed/output)",
            "pion": "Pion (Action tower ndim>=2 non-embed/output)",
            "adamw": "AdamW (1-D, embeddings, output heads, residual)",
        }[bucket]
    return bucket


# ──────────────────────────────────────────────────────────────────────────────
# Optimizer factory
# ──────────────────────────────────────────────────────────────────────────────


def _summarise(label: str, named_list: list[tuple[str, torch.Tensor]]) -> str:
    n = len(named_list)
    total = sum(int(p.numel()) for _, p in named_list)
    return f"  {label}: {n} tensors, {total/1e6:.1f}M params"


def build_optimizers(
    model: nn.Module,
    *,
    name: OptimizerName,
    peak_lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    muon_momentum: float = 0.95,
    muon_ns_steps: int = 5,
    muon_scale_factor: float = 2.0,
    pion_promotion_steps: int = 0,
    log_fn=print,
) -> list[torch.optim.Optimizer]:
    """Build the optimizer list for the given routing scheme."""
    buckets = classify_parameters(model, name)
    name_l = name.lower()

    log_fn(f"Optimizer routing for '{name_l}':")
    for b in ("muon", "pion", "adamw"):
        log_fn(_summarise(_bucket_human_label(name_l, b), buckets[b]))

    muon_params = [p for _, p in buckets["muon"]]
    pion_params = [p for _, p in buckets["pion"]]
    adamw_params = [p for _, p in buckets["adamw"]]

    optimizers: list[torch.optim.Optimizer] = []

    if muon_params:
        optimizers.append(
            _muon_optim.MuonAdamW(
                muon_params,
                lr=peak_lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                momentum=muon_momentum,
                ns_steps=muon_ns_steps,
                scale_factor=muon_scale_factor,
                nesterov=False,
                use_muon=True,
            )
        )
    if pion_params:
        optimizers.append(
            _muon_optim.DefaultPionAdamW(
                pion_params,
                lr=peak_lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                momentum=muon_momentum,
                ns_steps=muon_ns_steps,
                promotion_steps=pion_promotion_steps,
                scale_factor=muon_scale_factor,
                nesterov=False,
                use_muon=True,
            )
        )
    if adamw_params:
        optimizers.append(
            torch.optim.AdamW(
                adamw_params,
                lr=peak_lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
            )
        )
    if not optimizers:
        raise RuntimeError("No trainable parameters were assigned to any optimizer.")
    return optimizers
