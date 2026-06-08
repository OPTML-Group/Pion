# RL: Reinforcement Learning with Verifiable Rewards (RLVR)

This part of the Pion release reproduces the **RLVR** experiments from
the Pion paper ([arXiv:2605.19282](https://arxiv.org/abs/2605.19282),
§RLVR): GRPO and GMPO post-training of **Qwen3-1.7B** and **Qwen3-4B**
on **GSM8K** and **MATH**, with **AdamW**, **Muon**, and **Pion**
optimizers.

| Subfolder | Source | What it provides |
|-----------|--------|------------------|
| [`verl/`](verl/) | [volcengine/verl](https://github.com/volcengine/verl) | GRPO + GMPO trainers, the per-head Pion (`PerHeadPionAdamW`) and per-head Muon implementations, and 3 + many recipes for Qwen3-1.7B/4B on GSM8K + MATH-3-5 |

The verl subfolder ships its own FSDP2-aware optimizer module
[`verl/utils/muon.py`](verl/verl/utils/muon.py) with the four
AdamW-fused classes used by GRPO / GMPO:
`MuonAdamW`, `DefaultPionAdamW`, `PerHeadMuonAdamW`, `PerHeadPionAdamW`.

## Quick start (Qwen3-1.7B GRPO on GSM8K, 2 GPUs)

```bash
cd RL/verl
# one-time env + data prep (see verl/README.md for full details)
pip install -r requirements-cuda.txt && pip install -e .
python examples/data_preprocess/gsm8k.py --local_dir $HOME/data/gsm8k

# 3 runs (back-to-back)
bash scripts/run_adamw.sh    # AdamW baseline
bash scripts/run_muon.sh     # per-head Muon
bash scripts/run_pion.sh     # per-head Pion (recommended)
```

Switch model / dataset / trainer via env vars:

```bash
cd RL/verl
TRAINER=gmpo MODEL=qwen3-4b DATASET=math bash scripts/run_pion.sh
```

Full recipe documentation: [`verl/README.md`](verl/README.md).
