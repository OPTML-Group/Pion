# Pion on VLANeXt (LIBERO + LIBERO-Plus)

This folder reproduces the **VLANeXt** experiments from the Pion paper
([arXiv:2605.19282](https://arxiv.org/abs/2605.19282), §VLA).
It is a vendored, pruned copy of
[DravenALG/VLANeXt](https://github.com/DravenALG/VLANeXt) (a
flow-matching VLA built on Qwen3-VL-2B) with three drop-in scripts for
**AdamW**, **Muon**, and **Pion**.  The optimizer implementations live
in [`pion_optim/`](pion_optim/) (vendored, pure-PyTorch, no extra build
step).

## Environment

```bash
# Training env
conda create -n VLANeXt python=3.10 -y
conda activate VLANeXt

# 1. PyTorch 2.6 + CUDA 12.4 (torch 2.4 lacks
#    `torch.distributed.tensor.device_mesh` that transformers >=5.x imports).
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# 2. Repo deps. transformers is pinned to 5.2.0 (the version this code was
#    written against). Newer 5.x adds a required `mm_token_type_ids` arg to
#    `Qwen3VLModel.get_rope_index` and will break the forward pass.
pip install -r requirements.txt

# 3. Flash-Attention 2 (prebuilt wheel for CUDA 12 + torch 2.6 + cp310)
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# If your CUDA / torch / Python combo differs, pick the matching wheel from
# https://github.com/Dao-AILab/flash-attention/releases

conda install -c conda-forge ffmpeg -y
```

> The Pion optimizers in [`pion_optim/`](pion_optim/) are pure PyTorch
> and require no Triton / CUDA build.

LIBERO simulator (needed for evaluation; in the training env):

```bash
mkdir -p third_party
[ -d third_party/LIBERO ] || git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO

# Use the legacy (egg-link) editable install. PEP 660 — the default since
# pip 23 — generates a finder with an EMPTY MAPPING for LIBERO's old-style
# `setup.py`, so `import libero` silently fails with ModuleNotFoundError.
pip install -e third_party/LIBERO --config-settings editable_mode=compat

# Re-pin NumPy and OpenCV AFTER LIBERO install: robosuite / mujoco need
# numpy<2, but recent opencv-python (>=4.11) hard-requires numpy>=2 and
# will pull it back in via LIBERO's dependencies.
pip install "numpy<2" "opencv-python<4.11"

# torch 2.6 flipped the default of `torch.load(weights_only=...)` from False
# to True, which breaks LIBERO's init-states load (uses numpy globals not
# in the secure-load allowlist). Patch the one offending call site.
sed -i 's/torch\.load(init_states_path)$/torch.load(init_states_path, weights_only=False)/' \
    third_party/LIBERO/libero/libero/benchmark/__init__.py

# Headless GL/EGL runtime — only needed if mujoco crashes with EGL errors
# (e.g. `'NoneType' object has no attribute 'eglQueryString'`). Most
# cluster nodes already have these system-wide; skip if so.
#   - No sudo: install into the active conda env from conda-forge:
conda install -c conda-forge -y mesalib glew libglu
#   - With sudo (system-wide alternative):
#     sudo apt-get install -y libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev libglew-dev
```

LIBERO-Plus (separate env, only needed if you run the LIBERO-Plus eval —
its older `robosuite` pin would otherwise drag the training stack back):

```bash
conda create -n libero_plus_vlanext python=3.10 -y
conda activate libero_plus_vlanext

mkdir -p third_party
[ -d third_party/LIBERO-plus ] || git clone https://github.com/sylvestf/LIBERO-plus.git third_party/LIBERO-plus

# Same PEP 660 compat workaround as the LIBERO block above.
pip install -e third_party/LIBERO-plus --config-settings editable_mode=compat

# Upstream LIBERO-Plus extra system libs — only install if eval crashes
# for missing fontconfig / imagemagick. Most cluster nodes already have
# these; skip if so.
#   - No sudo: install into the active conda env from conda-forge:
conda install -c conda-forge -y expat fontconfig imagemagick
#   - With sudo (system-wide alternative):
#     sudo apt-get install -y libexpat1 libfontconfig1-dev libpython3-stdlib libmagickwand-dev
pip install -r third_party/LIBERO-plus/extra_requirements.txt

# Re-pin NumPy and OpenCV (same reason as above).
pip install "numpy<2" "opencv-python<4.11"

# LIBERO-Plus reads its own config; default location is ~/.libero_plus/config.yaml
# (see third_party/LIBERO-plus/README.md for the exact paths to set).
conda env config vars set LIBERO_CONFIG_PATH=~/.libero_plus
```

## Data preparation

LIBERO training data (RLDS, ~10 GB):

```bash
hf download openvla/modified_libero_rlds \
    --repo-type dataset \
    --local-dir data/LIBERO_modified
```

Path to the dataset is configured in
`config/libero_train_*_config.yaml`. By default the configs look for
`data/LIBERO_modified`. Edit the YAML if you store it elsewhere.

VLM backbone (Qwen3-VL-2B): downloaded automatically from HuggingFace on
first run; you can pre-fetch with `huggingface-cli download
Qwen/Qwen3-VL-2B-Instruct`.

## Training

All three runs share the configurable entrypoint
[`scripts/train_pion.py`](scripts/train_pion.py).  Configs live under
[`config/`](config/):

| Config                            | Used by      |
|-----------------------------------|--------------|
| `libero_train_config.yaml`        | `run_adamw.sh` |
| `libero_train_muon_config.yaml`   | `run_muon.sh`  |
| `libero_train_pion_config.yaml`  | `run_pion.sh`  |

The run scripts pin all important flags and accept overrides via env
vars (`TASK`, `MAX_STEPS`, `BATCH_SIZE`, `GRAD_ACC`, `GPUS`, `NPROC`,
`PORT`, plus Pion-specific `PROMOTION_STEPS`, `NS_STEPS`).

Default knobs match the paper:

| knob                | value                                                |
|---------------------|------------------------------------------------------|
| task suite          | `libero_object_no_noops`                             |
| max steps           | `10000` (all suites), save every `2000`              |
| global batch        | `256`                                                |
| LR                  | `1e-4` (in the YAML)                                 |
| GPUs                | `0..7` (8 × ~80 GB)                                  |

> All four LIBERO suites train for the same `10000` steps; only the
> evaluation checkpoint differs (see [Evaluation](#evaluation)).

Switch suite via env:

```bash
cd VLA/VLANeXt
TASK=libero_spatial_no_noops bash scripts/run_pion.sh
```

### 1. AdamW baseline

```bash
cd VLA/VLANeXt
bash scripts/run_adamw.sh
```

All parameters use AdamW.

### 2. Muon baseline

```bash
cd VLA/VLANeXt
bash scripts/run_muon.sh
```

All `ndim ≥ 2` parameters use Muon; 1-D parameters (LN / biases /
embeddings) fall back to AdamW.

### 3. Pion (recommended)

```bash
cd VLA/VLANeXt
bash scripts/run_pion.sh
```

Vision and language backbones use **Muon** (preserves pretraining); the
action head uses whole-matrix **Pion** (high-pass NS).

## Evaluation

### 1. Where the checkpoints land

Training writes to

```
checkpoints/VLANeXt/<config_name>/<run_name>/
├── checkpoint_2000.pt           # every `SAVE_INTERVAL=2000` steps
├── checkpoint_4000.pt
├── checkpoint_6000.pt
├── checkpoint_8000.pt
└── checkpoint_10000.pt
```

`<run_name>` encodes the optimizer routing (e.g.
`V_Muon_L_Muon_A_DefaultPion_O_Muon_PionPromo0_PionSupp5_PionNs5`), so
AdamW / Muon / Pion runs do not overwrite each other.

### 2. Which checkpoint to evaluate

All four LIBERO suites train for `10000` steps with a checkpoint saved
every `2000`. The paper-reported checkpoint differs by suite (`object`
converges much earlier):

| training task suite         | `--task_suite`    | checkpoint to evaluate |
|-----------------------------|-------------------|------------------------|
| `libero_object_no_noops`    | `libero_object`   | `checkpoint_6000.pt`   |
| `libero_spatial_no_noops`   | `libero_spatial`  | `checkpoint_10000.pt`  |
| `libero_goal_no_noops`      | `libero_goal`     | `checkpoint_10000.pt`  |
| `libero_10_no_noops`        | `libero_10`       | `checkpoint_10000.pt`  |

### 3. Run eval

The entrypoints
[`scripts.libero_bench_eval`](scripts/libero_bench_eval.py) and
[`scripts.libero_plus_bench_eval`](scripts/libero_plus_bench_eval.py)
both take a YAML config plus two CLI overrides (`--checkpoint`,
`--task_suite`):

```bash
cd VLA/VLANeXt

# --- LIBERO (training env) ---
conda activate VLANeXt
unset PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/third_party/LIBERO

CKPT=checkpoints/VLANeXt/.../checkpoint_6000.pt   # object example
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
python -m scripts.libero_bench_eval \
    --config config/libero_bench_config.yaml \
    --checkpoint "${CKPT}" \
    --task_suite libero_object

# --- LIBERO-Plus (separate env) ---
conda activate libero_plus_vlanext
unset PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/third_party/LIBERO-plus

CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
python -m scripts.libero_plus_bench_eval \
    --config config/libero_plus_bench_config.yaml \
    --checkpoint "${CKPT}" \
    --task_suite libero_object
```

For the other three suites swap `libero_object` ↔
{`libero_spatial`, `libero_goal`, `libero_10`} and point `CKPT` at the
matching `checkpoint_10000.pt`.

Per-rollout knobs (trials per task, diffusion steps, action-chunk
execution length, etc.) live in
[`config/libero_bench_config.yaml`](config/libero_bench_config.yaml) and
[`config/libero_plus_bench_config.yaml`](config/libero_plus_bench_config.yaml).
Defaults match the paper recipe (50 trials × 10 tasks = 500 rollouts per
suite for LIBERO; LIBERO-Plus uses its own task list).

### 4. Kill stuck eval

LIBERO eval forks many simulator processes; if Ctrl-C does not clean up:

```bash
pkill -f "python -m scripts.libero_bench_eval"
pkill -f "python -m scripts.libero_plus_bench_eval"
```

## Code layout

```
VLANeXt/
├── scripts/
│   ├── train_pion.py               # the only training entrypoint
│   ├── libero_bench_eval.py
│   ├── libero_plus_bench_eval.py
│   ├── run_adamw.sh
│   ├── run_muon.sh
│   └── run_pion.sh
├── config/                          # YAML configs (lr, model, data)
├── pion_optim/                      # Muon / DefaultPion (pure PyTorch)
├── src/{models,evaluation}/         # framework code
└── third_party/                     # LIBERO / LIBERO-plus (you clone)
```

## Citation

```bibtex
@article{fan2026rethinking,
  title={Rethinking Muon Beyond Pretraining: Spectral Failures and High-Pass Remedies for VLA and RLVR},
  author={Fan, Chongyu and Liu, Gaowen and Hong, Mingyi and Kompella, Ramana Rao and Liu, Sijia},
  journal={arXiv preprint arXiv:2605.19282},
  year={2026}
}

@article{wu2026vlanext,
  title   = {VLANeXt: Recipes for Building Strong VLA Models},
  author  = {Wu, Xiao-Ming and Fan, Bin and Liao, Kang and
             Jiang, Jian-Jian and Yang, Runze and Luo, Yihang and
             Wu, Zhonghua and Zheng, Wei-Shi and Loy, Chen Change},
  journal = {arXiv preprint arXiv:2602.18532},
  year    = {2026},
}
```
