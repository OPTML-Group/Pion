# Pion on VLA-Adapter (LIBERO)

This folder reproduces the **VLA-Adapter** experiments from the Pion
paper ([arXiv:2605.19282](https://arxiv.org/abs/2605.19282), §VLA).
It is a vendored, pruned copy of
[OpenHelix-Team/VLA-Adapter](https://github.com/OpenHelix-Team/VLA-Adapter)
with three drop-in scripts comparing **AdamW**, **Muon**, and **Pion**.
The optimizer implementations live in
[`pion_optim/`](pion_optim/) (vendored, pure-PyTorch, no extra build
step).

## Environment

```bash
# 1. Conda env
conda create -n vla-adapter python=3.10.16 -y
conda activate vla-adapter

# 2. PyTorch (CUDA 12.x)
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

# 3. Repo deps (run from this folder)
pip install -e .
pip install packaging ninja
ninja --version            # should print a version and exit 0

# 4. Pin NumPy < 2  (torch 2.2 was compiled against NumPy 1.x)
pip install "numpy<2"

# 5. Flash-Attention 2 (prebuilt wheel for CUDA 12.x + torch 2.2 + cp310)
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# If your CUDA / torch / Python combo differs, pick the matching wheel from
# https://github.com/Dao-AILab/flash-attention/releases/tag/v2.5.5
```

> The Pion optimizers in [`pion_optim/`](pion_optim/) are pure PyTorch
> and require no Triton / CUDA build.

> A frozen pip-freeze for the exact env we tested on is in
> [`our_envs.txt`](our_envs.txt).

## Data preparation

### LIBERO simulator

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git

# Use the legacy (egg-link) editable install. PEP 660 — the default since
# pip 23 — generates a finder with an EMPTY MAPPING for LIBERO's old-style
# `setup.py`, so `import libero` silently fails with ModuleNotFoundError.
pip install -e LIBERO --config-settings editable_mode=compat

pip install -r experiments/robot/libero/libero_requirements.txt

# Re-pin NumPy and OpenCV AFTER `libero_requirements.txt` is installed:
# robosuite / mujoco need numpy<2, but recent opencv-python (>=4.11) hard-
# requires numpy>=2 and will pull it back in via the requirements step.
pip install "numpy<2" "opencv-python<4.11"

# Headless GL/EGL runtime — only needed if mujoco crashes with EGL errors
# (e.g. `'NoneType' object has no attribute 'eglQueryString'`). Most
# cluster nodes already have these system-wide; skip if so.
#   - No sudo: install into the active conda env from conda-forge:
conda install -c conda-forge -y mesalib glew libglu
#   - With sudo (system-wide alternative):
#     sudo apt-get install -y libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev libglew-dev
```

### LIBERO datasets (RLDS)

```bash
git clone git@hf.co:datasets/openvla/modified_libero_rlds data/libero
# rename so the directory names lose the leading "modified_"
cd data/libero && for d in modified_*; do mv "$d" "${d#modified_}"; done && cd -
```

Final layout used by the run scripts:

```
data/libero/
├── libero_object_no_noops/1.0.0/        (32 tfrecords)
├── libero_spatial_no_noops/1.0.0/       (16 tfrecords)
├── libero_goal_no_noops/1.0.0/          (16 tfrecords)
└── libero_10_no_noops/1.0.0/            (32 tfrecords)
```

### VLM backbone

Download Prismatic-VLM (Qwen2.5-0.5B + DINO/SigLIP) and place under
`pretrained_models/`:

```bash
huggingface-cli download Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b \
    --local-dir pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b
```

`pretrained_models/configs/` is **already vendored** in this repo (the
9 tokenizer / processor JSON files + the 3 custom `*_prismatic.py`
modules that `AutoProcessor` / `AutoConfig` need at fine-tune time), so
you only need to download the VLM weights above.

Expected layout:

```
pretrained_models/
├── configs/                                         # vendored, no download
└── prism-qwen25-extra-dinosiglip-224px-0_5b/        # downloaded above
```

## Training

All three runs use a single configurable entrypoint,
[`vla-scripts/finetune_pion.py`](vla-scripts/finetune_pion.py), and
share identical hyper-parameters; they differ only in which optimizer
the vision / language backbones and the action head are routed to.

Default in every run script:

| knob                | value                                                |
|---------------------|------------------------------------------------------|
| dataset             | `libero_object_no_noops`                             |
| max steps           | `1500`, save every `500`                             |
| batch size / GPU    | `8`                                                  |
| GPUs                | `0,1,...,7` (8 × A100/H100, ~80GB)                   |
| LR / WD             | `1e-4` / `1e-2`                                      |

> The scripts default to `1500` steps, which is the LIBERO-Object recipe
> from the paper (where Pion reaches 100% success). The other three suites
> (`spatial` / `goal` / `10`) train for `15000` steps — override with
> `MAX_STEPS=15000 bash scripts/run_pion.sh`.

Switch suite via env:

```bash
cd VLA/VLAAdapter
# other suites train for 15000 steps
MAX_STEPS=15000 DATA_NAME=libero_spatial_no_noops bash scripts/run_pion.sh
```

### 1. AdamW baseline

```bash
cd VLA/VLAAdapter
bash scripts/run_adamw.sh
```

All parameters use AdamW.

### 2. Muon baseline

```bash
cd VLA/VLAAdapter
bash scripts/run_muon.sh
```

All `ndim ≥ 2` parameters use Muon; 1-D parameters (LN / biases /
embeddings) fall back to AdamW through an auxiliary bucket.

### 3. Pion (recommended)

```bash
cd VLA/VLAAdapter
bash scripts/run_pion.sh
```

Vision and language backbones use **Muon** (preserves pretraining); the
action head uses **Pion** (high-pass NS).

## Evaluation

### 1. Where the checkpoints land

The three run scripts write each saved checkpoint as a **sibling
directory** of the live run (one self-contained HF-style snapshot per
save), not as a `step_N/` subdirectory:

```
outputs/<optimizer>/<run_dir>/                       # live training dir
outputs/<optimizer>/<run_dir>--500_chkpt/            # every `SAVE_FREQ=500` steps
outputs/<optimizer>/<run_dir>--1000_chkpt/
outputs/<optimizer>/<run_dir>--1500_chkpt/           # last save for object (MAX_STEPS=1500)
...                                                  # up to --15000_chkpt/ for other suites
```

Each `--<N>_chkpt/` is a full HuggingFace snapshot
(`model.safetensors`, `action_head--<N>_checkpoint.pt`,
`proprio_projector--<N>_checkpoint.pt`, processor / tokenizer configs,
etc.) — point `--pretrained_checkpoint` at the directory itself.

`<run_dir>` is auto-generated; it concatenates the hyperparam string with
a `RUN_ID_NOTE` suffix that defaults to
`<Optimizer>-<data_name>-kp<k_p>-<timestamp>`, e.g.
`configs+libero_object_no_noops+b8+lr-0.0001+...--image_aug--Pion-libero_object_no_noops-kp0-0530-004242`.
Override with `RUN_ID_NOTE=…` if you want a stable name across re-runs.

### 2. Which checkpoint to evaluate

A checkpoint is saved every `500` steps. The paper-reported checkpoint
differs by suite: `libero_object` trains for `1500` steps (the script
default), while `spatial` / `goal` / `10` train for `15000` steps
(`MAX_STEPS=15000`):

| suite                       | `--task_suite_name` | steps   | checkpoint to evaluate     |
|-----------------------------|---------------------|---------|----------------------------|
| `libero_object_no_noops`    | `libero_object`     | `1500`  | `<run_dir>--1500_chkpt/`   |
| `libero_spatial_no_noops`   | `libero_spatial`    | `15000` | `<run_dir>--15000_chkpt/`  |
| `libero_goal_no_noops`      | `libero_goal`       | `15000` | `<run_dir>--15000_chkpt/`  |
| `libero_10_no_noops`        | `libero_10`         | `15000` | `<run_dir>--15000_chkpt/`  |

### 3. Run eval

Eval entrypoint is
[`experiments/robot/libero/run_libero_eval.py`](experiments/robot/libero/run_libero_eval.py)
(draccus-based; all flags are `--field value`).

```bash
cd VLA/VLAAdapter

# Pick the latest libero_object run's 1500-step checkpoint.
# (Glob over the long auto-generated run_dir prefix.)
CKPT=$(ls -d outputs/pion/*libero_object*--1500_chkpt 2>/dev/null | tail -1)
echo "Eval ckpt: ${CKPT}"

CUDA_VISIBLE_DEVICES=0 \
python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint "${CKPT}" \
    --task_suite_name libero_object \
    --use_proprio True \
    --num_images_in_input 2 \
    --use_film False \
    --use_pro_version True \
    --num_trials_per_task 50
```

- `--use_pro_version True` **must match the training setting**. All three
  `run_{adamw,muon,pion}.sh` scripts pass `--use_pro_version True`, so the
  action head saved in your checkpoint uses the Pro architecture
  (`k_self / v_self / k_adapter / v_adapter / k_task / v_task / film_gen`).
  Passing `False` here instantiates the simpler `k_proj / v_proj` head
  and produces a `Missing/Unexpected key(s) in state_dict` error.
- (The flag does NOT trigger any HuggingFace download. The official
  `VLA-Adapter/LIBERO-*-Pro` weights are only fetched when
  `--pretrained_checkpoint` itself is an HF repo id, not a local path.)
- Defaults match the paper recipe (`50` trials × `10` tasks per suite =
  `500` rollouts, ~hour on a single H100 / A100).

For the other 3 suites (swap suite name + step, all use `--15000_chkpt`):

```bash
python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint "$(ls -d outputs/pion/*libero_spatial*--15000_chkpt | tail -1)" \
    --task_suite_name libero_spatial \
    --use_proprio True --num_images_in_input 2 --use_film False --use_pro_version True
# … and similarly for libero_goal / libero_10
```

Outputs:
- per-suite success-rate log → `eval_logs/`
- rollout videos                → `rollouts/`

For LIBERO-Plus, swap the entrypoint to
[`experiments/robot/libero/run_libero_plus_eval.py`](experiments/robot/libero/run_libero_plus_eval.py)
(same args).

## Code layout

```
VLAAdapter/
├── vla-scripts/
│   ├── finetune_pion.py            # the only training entrypoint (AdamW / Muon / Pion)
│   ├── merge_lora_weights_and_save.py
│   └── vla_evaluation.py
├── scripts/
│   ├── run_adamw.sh
│   ├── run_muon.sh
│   └── run_pion.sh
├── pion_optim/                      # Muon / DefaultPion / LowRankMuon (pure PyTorch)
├── prismatic/                       # VLA-Adapter model code
├── experiments/                     # eval utilities (LIBERO/CALVIN wrappers)
└── pretrained_models/               # YOU put the VLM here
```

## Citation

If you use this code, please cite both Pion and VLA-Adapter:

```bibtex
@article{fan2026rethinking,
  title={Rethinking Muon Beyond Pretraining: Spectral Failures and High-Pass Remedies for VLA and RLVR},
  author={Fan, Chongyu and Liu, Gaowen and Hong, Mingyi and Kompella, Ramana Rao and Liu, Sijia},
  journal={arXiv preprint arXiv:2605.19282},
  year={2026}
}

@article{wang2025vlaadapter,
  title   = {VLA-Adapter: An Effective Paradigm for Tiny-Scale
             Vision-Language-Action Model},
  author  = {OpenHelix-Team},
  journal = {arXiv preprint arXiv:2509.09372},
  year    = {2025},
}
```
