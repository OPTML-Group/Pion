# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import warnings
from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING
import torch

from verl.base_config import BaseConfig

__all__ = [
    "OptimizerConfig",
    "FSDPOptimizerConfig",
    "McoreOptimizerConfig",
    "build_optimizer",
    "VeOmniOptimizerConfig",
    "TorchtitanOptimizerConfig",
]


def _normalize_parameters(parameters):
    """Return both plain params and optional (name, param) tuples."""
    parameter_items = list(parameters)
    if parameter_items and isinstance(parameter_items[0], tuple):
        named_params = parameter_items
        plain_params = [p for _, p in named_params]
        return plain_params, named_params
    return parameter_items, None


def _split_multihead_param_groups(
    named_params,
    num_q_heads: int | None,
    num_kv_heads: int | None,
    exclude_names=None,
    print_groups: bool = False,
    print_max_names: int = -1,
):
    """Split named parameters into per-head attention groups + fallback groups.

    For attention projections that can be decomposed per-head:
      - q_proj / k_proj / v_proj → head_split_dim=0 (heads on OUTPUT side)
      - o_proj                   → head_split_dim=1 (heads on INPUT  side)

    All remaining 2D+ params (FFN weights) fall back to whole-matrix Muon
    (num_heads=None in their group).  1D params and excluded params use AdamW.

    Args:
        named_params:  iterable of (name, param) pairs
        num_q_heads:   number of Q heads (e.g. 32 for Qwen3-4B)
        num_kv_heads:  number of KV heads (e.g.  8 for Qwen3-4B GQA)
        exclude_names: list of substrings; matching params fall back to AdamW
        print_groups:  whether to print group statistics on rank-0
        print_max_names: max param names to print per group (-1 = all)

    Returns:
        list of param group dicts for PerHeadMuonAdamW / PerHeadPionAdamW
    """
    exclude_names = exclude_names or []

    q_params, q_names   = [], []
    k_params, k_names   = [], []
    v_params, v_names   = [], []
    o_params, o_names   = [], []
    muon_params, muon_names = [], []
    adam_params, adam_names = [], []

    for name, param in named_params:
        if not param.requires_grad:
            continue
        has_exclude = any(key in name for key in exclude_names)

        if has_exclude or param.ndim < 2:
            adam_params.append(param)
            adam_names.append(name)
        elif "q_proj.weight" in name and num_q_heads is not None:
            q_params.append(param)
            q_names.append(name)
        elif "k_proj.weight" in name and num_kv_heads is not None:
            k_params.append(param)
            k_names.append(name)
        elif "v_proj.weight" in name and num_kv_heads is not None:
            v_params.append(param)
            v_names.append(name)
        elif "o_proj.weight" in name and num_q_heads is not None:
            o_params.append(param)
            o_names.append(name)
        else:
            muon_params.append(param)
            muon_names.append(name)

    if print_groups and _is_primary_rank():
        def _head(tag, names, total):
            print(f"[MultiHeadParamGroups][{tag}] count={total}")
            limit = total if print_max_names < 0 else print_max_names
            for n in names[:limit]:
                print(f"  - {n}")
            if 0 <= print_max_names < total:
                print(f"  ... ({total - print_max_names} more)")

        print(
            f"[MultiHeadParamGroups] "
            f"q={len(q_params)} k={len(k_params)} v={len(v_params)} "
            f"o={len(o_params)} muon={len(muon_params)} adamw={len(adam_params)} "
            f"num_q_heads={num_q_heads} num_kv_heads={num_kv_heads} "
            f"exclude={exclude_names}"
        )
        _head("Q",     q_names,    len(q_params))
        _head("K",     k_names,    len(k_params))
        _head("V",     v_names,    len(v_params))
        _head("O",     o_names,    len(o_params))
        _head("Muon",  muon_names, len(muon_params))
        _head("AdamW", adam_names, len(adam_params))

    param_groups = []
    if q_params:
        param_groups.append({"params": q_params, "num_heads": num_q_heads,  "head_split_dim": 0, "use_muon": True})
    if k_params:
        param_groups.append({"params": k_params, "num_heads": num_kv_heads, "head_split_dim": 0, "use_muon": True})
    if v_params:
        param_groups.append({"params": v_params, "num_heads": num_kv_heads, "head_split_dim": 0, "use_muon": True})
    if o_params:
        param_groups.append({"params": o_params, "num_heads": num_q_heads,  "head_split_dim": 1, "use_muon": True})
    if muon_params:
        # num_heads defaults to None in the optimizer → whole-matrix Muon
        param_groups.append({"params": muon_params, "use_muon": True})
    if adam_params:
        param_groups.append({"params": adam_params, "use_muon": False})
    return param_groups


def _split_muon_param_groups(named_params, ndim_min=2, include_names=None, exclude_names=None):
    include_names = include_names or []
    exclude_names = exclude_names or []

    muon_params = []
    adam_params = []
    muon_names = []
    adam_names = []

    for name, param in named_params:
        if not param.requires_grad:
            continue

        has_include = any(key in name for key in include_names)
        has_exclude = any(key in name for key in exclude_names)
        use_muon = param.ndim >= ndim_min and (has_include or not has_exclude)

        if use_muon:
            muon_params.append(param)
            muon_names.append(name)
        else:
            adam_params.append(param)
            adam_names.append(name)

    return muon_params, adam_params, muon_names, adam_names


def _is_primary_rank():
    if not torch.distributed.is_available():
        return True
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


@dataclass
class OptimizerConfig(BaseConfig):
    """Base optimizer configuration.

    Args:
        lr (float): learning rate. Must be specified.
        lr_warmup_steps_ratio (float): Warmup steps ratio; total steps will be injected at runtime.
        total_training_steps (int): Total training steps (must be overridden at runtime).
        weight_decay (float): Weight decay factor.
        lr_warmup_steps (Optional[int]): Number of warmup steps; None delegates to lr_warmup_steps_ratio.
    """

    _mutable_fields = {"clip_grad", "total_training_steps", "lr_warmup_steps"}

    lr: float = 1e-3
    lr_warmup_steps_ratio: float = 0.0
    total_training_steps: int = -1
    weight_decay: float = 0.01
    lr_warmup_steps: Optional[int] = -1
    betas: tuple[float, float] = (0.9, 0.999)
    clip_grad: float = 1.0
    # deprecate grad_clip
    grad_clip: Optional[float] = None

    def __post_init__(self):
        assert self.lr != MISSING
        if self.grad_clip is not None:
            warnings.warn("`grad_clip` is deprecated, use `clip_grad` instead.", DeprecationWarning, stacklevel=2)
            self.clip_grad = self.grad_clip


@dataclass
class VeOmniOptimizerConfig(OptimizerConfig):
    """VeOmni optimizer configuration extending base OptimizerConfig.

    Args:
        optimizer (str): Optimizer name; default is "adamw".
        lr (float): Learning rate.
        lr_min (float): Minimum learning rate.
        lr_start (float): Starting learning rate for warmup.
        lr_decay_ratio (float): LR decay ratio.
        lr_scheduler_type (str): LR scheduler type: "constant" or "cosine".
    """

    _mutable_fields = OptimizerConfig._mutable_fields.copy()

    optimizer: str = "adamw"
    lr_min: float = 0.0
    lr_start: float = 0.0
    lr_decay_ratio: float = 1.0
    lr_scheduler_type: str = "constant"
    override_optimizer_config: Optional[dict] = None


@dataclass
class FSDPOptimizerConfig(OptimizerConfig):
    """FSDP optimizer configuration extending base OptimizerConfig.

    Args:
        optimizer (str): Optimizer class name (e.g., "AdamW", "AdamW8bit", "_AdamW").
        optimizer_impl (str): Module path to import optimizer from (e.g., "torch.optim", "torchao.optim",
            "bitsandbytes.optim").
        lr (float): Learning rate.
        min_lr_ratio (Optional[float]): Minimum LR ratio for cosine schedule.
        lr_scheduler_type (str): LR scheduler type: "constant" or "cosine".
        num_cycles (float): Number of cosine cycles in LR schedule.
        zero_indexed_step (bool): Whether the LR schedule uses 0-indexed steps. If True (default),
            step counting starts at 0. If False, step counting starts at 1.
    """

    _mutable_fields = OptimizerConfig._mutable_fields.copy()
    _mutable_fields.add("lr_scheduler_type")

    optimizer: str = "AdamW"
    optimizer_impl: str = "torch.optim"
    min_lr_ratio: Optional[float] = None
    # deprecate warmup_style
    warmup_style: Optional[str] = None
    lr_scheduler_type: str = "constant"
    num_cycles: float = 0.5
    override_optimizer_config: Optional[dict] = None
    zero_indexed_step: bool = True

    def __post_init__(self):
        if self.warmup_style is not None:
            assert self.warmup_style in ["constant", "cosine"]
            warnings.warn(
                "`warmup_style` is deprecated, use `lr_scheduler_type` instead.", DeprecationWarning, stacklevel=2
            )
            self.lr_scheduler_type = self.warmup_style
        assert self.lr_scheduler_type in ["constant", "cosine"]
        return super().__post_init__()


@dataclass
class McoreOptimizerConfig(OptimizerConfig):
    """Mcore optimizer configuration extending base OptimizerConfig.

    Args:
        optimizer (str): Optimizer name; default is "adam".
        lr (float): Learning rate.
        clip_grad (float): Gradient clipping norm.
        lr_warmup_init (float): Initial learning rate for warmup; defaults to 0.0.
        lr_decay_steps (Optional[int]): Number of decay steps.
        lr_decay_style (str): LR decay style: "constant", "linear", "cosine", or "inverse_square_root".
        min_lr (float): Minimum learning rate.
        weight_decay_incr_style (str): Weight decay increment style: "constant" or "cosine".
        lr_wsd_decay_style (str): Weight-standard-deviation decay style: "constant", "exponential", or "cosine".
        lr_wsd_decay_steps (Optional[int]): Number of steps for weight-standard-deviation decay.
        use_checkpoint_opt_param_scheduler (bool): Whether to use checkpoint optimizer parameter scheduler.
    """

    optimizer: str = "adam"
    lr_warmup_init: float = 0.0
    lr_decay_steps: Optional[int] = None
    lr_decay_style: str = "linear"
    min_lr: float = 0.0
    weight_decay_incr_style: str = "constant"
    lr_wsd_decay_style: str = "exponential"
    lr_wsd_decay_steps: Optional[int] = None
    use_checkpoint_opt_param_scheduler: bool = False
    override_optimizer_config: Optional[dict] = None


@dataclass
class TorchtitanOptimizerConfig(OptimizerConfig):
    """Torchtitan optimizer configuration extending base OptimizerConfig.

    Args:
        name (str): Optimizer name; default is "AdamW".
        eps (float): Epsilon value for AdamW optimizer, default 1e-8.
        decay_type (str): Weight decay type: "linear", "sqrt", or "cosine".
        min_lr_factor (float): Minimum learning rate factor.
    """

    name: str = "AdamW"
    eps: float = 1e-8
    decay_type: str = "linear"
    min_lr_factor: float = 0.0


def build_optimizer(parameters, config: FSDPOptimizerConfig):
    """Build an optimizer based on the configuration.

    Dynamically imports and instantiates an optimizer class from the specified module.

    Args:
        parameters: Model parameters to optimize
        config: FSDPOptimizerConfig with optimizer settings

    Returns:
        Optimizer instance

    Examples:
        # PyTorch AdamW
        config.optimizer_impl = "torch.optim"
        config.optimizer = "AdamW"

        # TorchAO AdamW with bf16 stochastic rounding
        config.optimizer_impl = "torchao.optim"
        config.optimizer = "_AdamW"
        config.override_optimizer_config = {"bf16_stochastic_round": True}

        # BitsAndBytes AdamW 8bit
        config.optimizer_impl = "bitsandbytes.optim"
        config.optimizer = "AdamW8bit"
    """
    import importlib

    plain_parameters, named_parameters = _normalize_parameters(parameters)

    optimizer_args = {
        "lr": config.lr,
        "weight_decay": config.weight_decay,
    }

    optimizer_name_lower = config.optimizer.lower()
    if "adam" in optimizer_name_lower or "ademamix" in optimizer_name_lower:
        optimizer_args["betas"] = config.betas

    if config.override_optimizer_config is not None:
        optimizer_args.update(config.override_optimizer_config)

    try:
        module = importlib.import_module(config.optimizer_impl)
        optimizer_cls = getattr(module, config.optimizer)
    except ImportError as e:
        raise ImportError(
            f"Failed to import module '{config.optimizer_impl}'. Make sure the package is installed. Error: {e}"
        ) from e
    except AttributeError as e:
        raise AttributeError(
            f"Optimizer '{config.optimizer}' not found in module '{config.optimizer_impl}'. "
            f"Available optimizers: {dir(module)}"
        ) from e

    if config.optimizer_impl == "verl.utils.muon" and named_parameters is not None:
        # ── Whole-matrix MuonAdamW / DefaultPionAdamW ───────────────────────
        if config.optimizer in {"MuonAdamW", "DefaultPionAdamW"}:
            muon_ndim_min = optimizer_args.pop("muon_ndim_min", 2)
            muon_include_names = optimizer_args.pop("muon_include_names", [])
            muon_exclude_names = optimizer_args.pop("muon_exclude_names", [])
            muon_print_param_groups = optimizer_args.pop("muon_print_param_groups", False)
            muon_print_max_names = int(optimizer_args.pop("muon_print_max_names", -1))

            muon_params, adam_params, muon_names, adam_names = _split_muon_param_groups(
                named_params=named_parameters,
                ndim_min=muon_ndim_min,
                include_names=muon_include_names,
                exclude_names=muon_exclude_names,
            )

            param_groups = []
            if muon_params:
                param_groups.append({"params": muon_params, "use_muon": True})
            if adam_params:
                param_groups.append({"params": adam_params, "use_muon": False})

            if muon_print_param_groups and _is_primary_rank():
                if muon_print_max_names >= 0:
                    muon_names_to_print = muon_names[:muon_print_max_names]
                    adam_names_to_print = adam_names[:muon_print_max_names]
                else:
                    muon_names_to_print = muon_names
                    adam_names_to_print = adam_names

                print(
                    "[MuonParamGroups] "
                    f"muon={len(muon_params)} adamw={len(adam_params)} "
                    f"ndim_min={muon_ndim_min} include={muon_include_names} exclude={muon_exclude_names}"
                )
                print("[MuonParamGroups][Muon]")
                for n in muon_names_to_print:
                    print(f"  - {n}")
                if 0 <= muon_print_max_names < len(muon_names):
                    print(f"  ... ({len(muon_names) - muon_print_max_names} more)")

                print("[MuonParamGroups][AdamW]")
                for n in adam_names_to_print:
                    print(f"  - {n}")
                if 0 <= muon_print_max_names < len(adam_names):
                    print(f"  ... ({len(adam_names) - muon_print_max_names} more)")

            if param_groups:
                return optimizer_cls(param_groups, **optimizer_args)

        # ── Per-head PerHeadMuonAdamW / PerHeadPionAdamW ──────────────────
        elif config.optimizer in {"PerHeadMuonAdamW", "PerHeadPionAdamW"}:
            num_q_heads = optimizer_args.pop("num_q_heads", None)
            num_kv_heads = optimizer_args.pop("num_kv_heads", None)
            muon_exclude_names = optimizer_args.pop("muon_exclude_names", [])
            muon_print_param_groups = optimizer_args.pop("muon_print_param_groups", False)
            muon_print_max_names = int(optimizer_args.pop("muon_print_max_names", -1))

            param_groups = _split_multihead_param_groups(
                named_params=named_parameters,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                exclude_names=muon_exclude_names,
                print_groups=muon_print_param_groups,
                print_max_names=muon_print_max_names,
            )

            if param_groups:
                return optimizer_cls(param_groups, **optimizer_args)

    return optimizer_cls(plain_parameters, **optimizer_args)
