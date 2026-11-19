"""CPU regression test: fine-tuning keeps the trainable weights in fp32 (the master copy) and computes in bf16.

PyTorch's AdamW keeps no fp32 copy of a bf16 parameter: it writes each update straight into bf16, which rounds away any
update smaller than about 1/512 of the weight (https://arxiv.org/abs/2610.31859).
"""

import ast
from pathlib import Path

import torch

FINETUNE = Path(__file__).resolve().parents[1] / "vla-scripts" / "finetune_pion.py"


def _adamw_step(weight_dtype):
    torch.manual_seed(0)
    layer = torch.nn.Linear(256, 256, bias=False)
    with torch.no_grad():
        layer.weight.uniform_(0.5, 1.5)  # weights near 1.0
    layer = layer.to(weight_dtype)
    before = layer.weight.detach().clone()
    with torch.autocast("cpu", dtype=torch.bfloat16):  # bf16 compute, as finetune_pion.py does on CUDA
        loss = layer(torch.randn(16, 256)).float().square().mean()
    loss.backward()
    torch.optim.AdamW(layer.parameters(), lr=1e-5).step()
    return before, layer.weight.detach()


def test_bf16_weights_lose_small_adamw_updates():
    before, after = _adamw_step(torch.bfloat16)
    assert torch.equal(after, before)  # no master copy: every update was rounded away

    before, after = _adamw_step(torch.float32)
    assert (after != before).all()  # fp32 master copy: every update was kept


def _function(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _keyword_values(function, keyword):
    """(line, value) of every `keyword=value` argument passed inside `function`."""
    return [
        (call.lineno, kw.value)
        for call in ast.walk(function)
        if isinstance(call, ast.Call)
        for kw in call.keywords
        if kw.arg == keyword
    ]


def _is_torch(node, attr):
    return isinstance(node, ast.Attribute) and node.attr == attr and getattr(node.value, "id", None) == "torch"


def _is_autocast(with_item):
    expr = with_item.context_expr
    return isinstance(expr, ast.Call) and _is_torch(expr.func, "autocast")


def test_finetune_pion_keeps_trainable_weights_in_fp32():
    # The script needs GPUs, the RLDS data and the VLM checkpoint to run, so check its source.
    tree = ast.parse(FINETUNE.read_text())
    finetune = _function(tree, "finetune")

    # The VLA, the action head and the proprio projector are built in fp32 ...
    dtypes = _keyword_values(finetune, "torch_dtype")
    not_fp32 = [line for line, value in dtypes if not _is_torch(value, "float32")]
    assert dtypes and not not_fp32, f"finetune() builds the VLA in a dtype other than fp32 at lines {not_fp32}"
    to_bf16 = _keyword_values(finetune, "to_bf16")
    cast = [line for line, value in to_bf16 if not (isinstance(value, ast.Constant) and value.value is False)]
    assert not cast, f"finetune() casts a trainable module to bf16 at lines {cast}"

    # ... and the action head computes in bf16 under autocast, like the VLA.
    autocast_blocks = [
        node
        for node in ast.walk(_function(tree, "run_forward_pass"))
        if isinstance(node, ast.With) and any(_is_autocast(item) for item in node.items)
    ]
    assert any("predict_action" in ast.dump(block) for block in autocast_blocks), "action head runs outside autocast"
