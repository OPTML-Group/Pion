# Pion on verl (RLVR: GRPO / GMPO on Qwen3-1.7B & 4B, MATH / GSM8K)

This folder reproduces the **RLVR** experiments from the Pion paper
([arXiv:2605.19282](https://arxiv.org/abs/2605.19282), §RLVR). It is a
vendored, pruned copy of [volcengine/verl](https://github.com/volcengine/verl)
restricted to:

| Algorithm  | Models                | Datasets           | Optimizers |
|------------|-----------------------|--------------------|------------|
| GRPO, GMPO | Qwen3-1.7B, Qwen3-4B  | GSM8K, MATH (3-5)  | AdamW, Muon, Pion (per-head + whole-matrix) |

The four families of optimizers live in
[`verl/utils/muon.py`](verl/utils/muon.py):
`MuonAdamW`, `DefaultPionAdamW` (= DefaultPion fused with AdamW for excluded
params), `PerHeadMuonAdamW`, `PerHeadPionAdamW` (= PerHeadPion).

## Environment

```bash
conda create -n verl-pion python=3.10 -y
conda activate verl-pion

# 1. PyTorch + CUDA 12 + matched flash-attn
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install flash-attn==2.7.4.post1 --no-build-isolation

# 2. verl runtime (vLLM rollout, FSDP2 training)
pip install -r requirements-cuda.txt
pip install -e .

# 3. (optional) Hugging Face login for gated models / wandb
huggingface-cli login
wandb login
```

> If you target NPU instead of CUDA, use `requirements-npu.txt` and follow
> upstream verl docs.

## Data preparation

Both GSM8K and MATH need to be turned into verl's parquet format once:

```bash
# GSM8K -> $HOME/data/gsm8k/{train,test}.parquet
python examples/data_preprocess/gsm8k.py --local_dir $HOME/data/gsm8k

# MATH (full 7.5k) -> $HOME/data/math/{train,test}.parquet
python examples/data_preprocess/math_dataset.py --local_dir $HOME/data/math

# MATH-3-to-5 (difficulty 3-5 split used in the paper)
# The math3-5 example scripts read $HOME/data/math/train_3to5.parquet by
# default; if you used examples/data_preprocess/math_dataset.py above,
# you can keep the original parquet and pass
#     data.train_files=$HOME/data/math/train.parquet
# data.val_files=$HOME/data/math/test.parquet  on the CLI.
```

Pre-fetch the base models (saves time on the first run):

```bash
huggingface-cli download Qwen/Qwen3-1.7B
huggingface-cli download Qwen/Qwen3-4B
```

## Training

We provide three top-level wrappers under [`scripts/`](scripts/) that
dispatch to the right recipe in
[`examples/{grpo,gmpo}_trainer/`](examples/):

```bash
cd RL/verl
# GRPO, Qwen3-1.7B, GSM8K, default optimizer family:
bash scripts/run_adamw.sh                # AdamW baseline
bash scripts/run_muon.sh                 # whole-matrix Muon (baseline)
bash scripts/run_pion.sh                 # per-head Pion (recommended)

# All three wrappers accept the same env overrides:
TRAINER=gmpo FLAVOR=perhead bash scripts/run_muon.sh        # GMPO (per-head Muon; GMPO has no whole-matrix recipe)
TRAINER=gmpo bash scripts/run_pion.sh                       # GMPO instead of GRPO
MODEL=qwen3-4b bash scripts/run_pion.sh                     # 4B instead of 1.7B
DATASET=math bash scripts/run_pion.sh                       # MATH-3-5 instead of GSM8K
PROMOTION_STEPS=1 bash scripts/run_pion.sh                  # override Pion default
FLAVOR=perhead bash scripts/run_muon.sh                     # per-head Muon (Muon defaults to whole-matrix)
FLAVOR=whole bash scripts/run_pion.sh                       # whole-matrix Pion (GRPO only)
```

Each wrapper just `exec`s the matching recipe under
`examples/{grpo,gmpo}_trainer/`, so you can also call those directly if
you want full control. The naming convention there is:

```
run_<model>_fsdp2_<optim>[<_math3-5>].sh

<model> ∈ {qwen3-1.7b, qwen3-4b}
<optim> ∈ {adamw, muon, perhead_muon, pion, perhead_pion}
```

(`muon` = whole-matrix Muon (default for `run_muon.sh`), `perhead_muon`
= per-head Muon.  `pion` = whole-matrix Pion, `perhead_pion` = per-head
Pion (default for `run_pion.sh`).)

## Recipe defaults (Qwen3-1.7B example)

Inside each `run_qwen3-1.7b_fsdp2_*.sh`:

| knob                       | value                                  |
|----------------------------|----------------------------------------|
| algorithm                  | `grpo` / `gmpo`                        |
| rollout backend            | vLLM, tp=2                             |
| GPUs                       | 2 (per recipe; scale via verl config)  |
| train batch (prompts)      | 1024, 5 rollouts/prompt                |
| micro-batch / GPU          | 16                                     |
| max prompt / response      | 512 / 1024                             |
| LR                         | 1e-6                                   |
| KL loss                    | low-var, coef 0.001                    |
| FSDP                       | FSDP2                                  |
| optimizer impl             | `verl.utils.muon`                      |
| optimizer class            | `MuonAdamW` / `PerHeadMuonAdamW` /  `DefaultPionAdamW` / `PerHeadPionAdamW` |
| Pion knobs                 | `ns_steps=5`, `promotion_steps=$PROMOTION_STEPS` |
| GQA topology               | num_q_heads=16, num_kv_heads=8         |
| excluded params (AdamW)    | `embed`, `lm_head`                     |

## Code layout

```
verl/
├── verl/
│   ├── utils/
│   │   └── muon.py                  # Muon + Pion + per-head variants (the actual optimizer code)
│   └── trainer/main_ppo.py          # the entrypoint all recipes call
├── examples/
│   ├── grpo_trainer/
│   │   ├── README.md
│   │   ├── run_qwen3-1.7b_fsdp2_adamw[.|_math3-5].sh
│   │   ├── run_qwen3-1.7b_fsdp2_muon[.|_math3-5].sh
│   │   ├── run_qwen3-1.7b_fsdp2_perhead_muon[.|_math3-5].sh
│   │   ├── run_qwen3-1.7b_fsdp2_pion[.|_math3-5].sh
│   │   ├── run_qwen3-1.7b_fsdp2_perhead_pion[.|_math3-5].sh
│   │   ├── run_qwen3-4b_fsdp2_*  (analogous set)
│   ├── gmpo_trainer/
│   │   └── (same set, per-head variants only)
│   └── data_preprocess/             # GSM8K / MATH / etc. converters
├── scripts/
│   ├── run_adamw.sh                 # NEW thin wrapper
│   ├── run_muon.sh                  # NEW thin wrapper
│   └── run_pion.sh                  # NEW thin wrapper
└── requirements*.txt                # CUDA / NPU / sglang variants
```

## Citation

```bibtex
@article{fan2026rethinking,
  title={Rethinking Muon Beyond Pretraining: Spectral Failures and High-Pass Remedies for VLA and RLVR},
  author={Fan, Chongyu and Liu, Gaowen and Hong, Mingyi and Kompella, Ramana Rao and Liu, Sijia},
  journal={arXiv preprint arXiv:2605.19282},
  year={2026}
}

@article{sheng2025verl,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Sheng, Guangming and Zhang, Chi and Ye, Zilingfeng and
             Wu, Xibin and Zhang, Wang and Zhang, Ru and Peng, Yanghua
             and Lin, Haibin and Wu, Chuan},
  journal = {arXiv preprint arXiv:2409.19256},
  year    = {2025},
}
```
