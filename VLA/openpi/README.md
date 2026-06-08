# Pion on openpi (π0.5 DROID fine-tuning)

This folder reproduces the **openpi / π0.5** fine-tuning experiments
from the Pion paper ([arXiv:2605.19282](https://arxiv.org/abs/2605.19282),
§VLA, real-robot Franka FR3 / DROID).  It is a vendored, pruned copy
of [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)
with three drop-in scripts for **AdamW**, **Muon**, and **Pion**.

The AdamW-fused optimizer implementations live in
[`src/openpi/training/muon_optim.py`](src/openpi/training/muon_optim.py)
(`MuonAdamW`, `DefaultPionAdamW`).

## Environment

openpi uses [`uv`](https://docs.astral.sh/uv/) for env management.

```bash
# 1. Install uv (if not already)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Sync env (creates .venv with PyTorch + JAX + LeRobot + flash-attn)
cd Final/Pion/VLA/openpi
uv sync

# 3. Patch HuggingFace transformers (openpi ships its own gemma_expert head)
cp -r src/openpi/models_pytorch/transformers_replace/* \
      .venv/lib/python3.11/site-packages/transformers/

# 4. JAX must stay on CPU so it does not allocate on every torchrun rank
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"
```

## Data preparation

### 1) Raw DROID episodes → LeRobot format (run once per task)

Place raw demonstrations under `<DATA_ROOT>/<task>/`. The paper uses
three Franka FR3 grasp-and-place tasks. Convert each one to LeRobot:

```bash
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}

for task in cubic_to_bowl cubic_to_plate cucumber_to_plate; do
    uv run examples/droid/convert_collected_data_to_lerobot.py \
        --data_dir "<DATA_ROOT>" --task_name "${task}"
done
```

Output: `$HF_LEROBOT_HOME/collected_data/<task>/`.

The language instruction for each task is auto-mapped from the folder
name:

| Task               | Prompt                                       |
|--------------------|----------------------------------------------|
| `cubic_to_bowl`    | "put the rubik's cube into the bowl"         |
| `cubic_to_plate`   | "put the rubik's cube on the plate"          |
| `cucumber_to_plate`| "put the cucumber on the plate"              |

### 2) π0.5-DROID JAX → PyTorch checkpoint (run once, ~10 min)

```bash
# 2a. Download the JAX π0.5-DROID checkpoint (~11.6 GiB)
mkdir -p ~/.cache/openpi/openpi-assets/checkpoints
gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_droid \
    ~/.cache/openpi/openpi-assets/checkpoints/

# 2b. Convert to PyTorch safetensors
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid \
    --output_path   ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch \
    --config_name   pi05_droid
```

The three task configs (`pi05_<task>_finetune` in
`src/openpi/training/config.py`) automatically load this PyTorch
checkpoint.

## Training

All three runs share `scripts/train_pytorch.py` and select the
optimizer via `--pytorch-optimizer-name={adamw,muon,pion}`.

Routing scheme (built into `src/openpi/training/optim_groups.py`):

| `--pytorch-optimizer-name` | VL tower (PaliGemma) | Action tower (Gemma-expert)       | 1-D / embed / output |
|----------------------------|----------------------|-----------------------------------|----------------------|
| `adamw`                    | AdamW                | AdamW                             | AdamW                |
| `muon`                     | `MuonAdamW`          | `MuonAdamW`                       | AdamW                |
| `pion`                     | `MuonAdamW`          | **`DefaultPionAdamW`** (high-pass)| AdamW                |

Default training hyperparameters (locked into the three run scripts):

| knob                  | value                                          |
|-----------------------|------------------------------------------------|
| GPUs                  | 8                                              |
| global batch          | 32                                             |
| steps                 | 20000, save every 5000                         |
| LR schedule           | warmup 1000 → peak 2.5e-5 → cosine to 2.5e-6   |

Sequentially runs all 3 tasks:

```bash
cd VLA/openpi
bash scripts/run_adamw.sh    # AdamW on all 3 tasks
bash scripts/run_muon.sh     # Muon  on all 3 tasks
bash scripts/run_pion.sh     # Pion  on all 3 tasks
```

To run a single task, edit the `TASKS=( … )` array at the top of the
script.

### Resume / skip behavior (built into the run scripts)

- If `checkpoints/.../1999/` exists → task is **skipped** (already done).
- If an earlier checkpoint exists → run is **resumed** from latest step.
- Otherwise → fresh start with `--overwrite`.

## Code layout

```
openpi/
├── scripts/
│   ├── train_pytorch.py             # the only training entrypoint
│   ├── compute_norm_stats.py        # one-time data-stats pass
│   ├── serve_policy.py              # real-robot inference server
│   ├── run_adamw.sh
│   ├── run_muon.sh
│   └── run_pion.sh
├── src/openpi/training/
│   ├── config.py                    # task configs (pi05_<task>_finetune)
│   ├── optim_groups.py              # adamw / muon / pion routing
│   ├── muon_optim.py                # MuonAdamW + DefaultPionAdamW
│   └── muon_distributed.py          # DDP-sharded NS / Pion polynomials
├── src/openpi/models_pytorch/       # PyTorch port of π0.5
├── packages/                        # openpi-client utilities
└── examples/                        # data converters, JAX→PT, etc.
```

## Citation

```bibtex
@article{fan2026rethinking,
  title={Rethinking Muon Beyond Pretraining: Spectral Failures and High-Pass Remedies for VLA and RLVR},
  author={Fan, Chongyu and Liu, Gaowen and Hong, Mingyi and Kompella, Ramana Rao and Liu, Sijia},
  journal={arXiv preprint arXiv:2605.19282},
  year={2026}
}

@misc{physicalintelligence2024openpi,
  title  = {openpi},
  author = {{Physical Intelligence}},
  year   = {2024},
  howpublished = {\url{https://github.com/Physical-Intelligence/openpi}},
}
```
