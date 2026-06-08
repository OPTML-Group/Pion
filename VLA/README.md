# VLA: Vision-Language-Action training

This part of the Pion release reproduces the **VLA** experiments from
the Pion paper ([arXiv:2605.19282](https://arxiv.org/abs/2605.19282),
§VLA), spanning three complementary VLA architectures and two action
parameterizations (ℓ1-regression and flow-matching), all with
**AdamW**, **Muon**, and **Pion** optimizers.

| Subfolder | Source | Architecture / action head | Benchmarks |
|-----------|--------|----------------------------|------------|
| [`VLAAdapter/`](VLAAdapter/) | [OpenHelix-Team/VLA-Adapter](https://github.com/OpenHelix-Team/VLA-Adapter) | Prismatic-VLM (Qwen2.5-0.5B) + LoRA + ℓ1-regression action head | LIBERO (Object / Spatial / Goal / Long) |
| [`VLANeXt/`](VLANeXt/)       | [DravenALG/VLANeXt](https://github.com/DravenALG/VLANeXt)            | Qwen3-VL-2B + flow-matching action head | LIBERO + LIBERO-Plus |
| [`openpi/`](openpi/)         | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | π0.5 + flow matching                            | Real-robot Franka FR3 (DROID schema) |

Each subfolder contains:
- the vendored source tree (heavy datasets / checkpoints / logs stripped);
- one configurable training entrypoint (`finetune_pion.py` /
  `train_pion.py` / `train_pytorch.py`);
- three drop-in run scripts: `run_adamw.sh`, `run_muon.sh`, `run_pion.sh`;
- a `README.md` with environment setup, data preparation and the
  exact commands used in the paper.

Each subfolder ships its own self-contained optimizer module so the
recipes run out-of-the-box:

| Subfolder    | Optimizer module                                   | Classes provided                                                             |
|--------------|----------------------------------------------------|------------------------------------------------------------------------------|
| `VLA-Adapter` | `pion_optim/muon.py` (pure PyTorch)                | `Muon`, `DefaultPion`, `LowRankMuon`                                         |
| `VLANeXt`    | `pion_optim/muon.py` (pure PyTorch)                | `Muon`, `DefaultPion`                                                        |
| `openpi`     | `src/openpi/training/muon_optim.py` (FSDP-aware)   | `MuonAdamW`, `DefaultPionAdamW`                                              |

`*AdamW` variants are AdamW-fused (Muon/Pion on 2-D parameters, AdamW on
the rest); the bare classes use only the Muon / Pion update.

## Quick start (per sub-repo)

```bash
cd VLA

# VLA-Adapter on LIBERO Object (8 × ~80 GB GPUs)
cd VLAAdapter
bash scripts/run_adamw.sh
bash scripts/run_muon.sh
bash scripts/run_pion.sh

# VLANeXt on LIBERO Object
cd ../VLANeXt
bash scripts/run_adamw.sh
bash scripts/run_muon.sh
bash scripts/run_pion.sh

# openpi π0.5 fine-tune on all 3 Franka tasks
cd ../openpi
bash scripts/run_adamw.sh
bash scripts/run_muon.sh
bash scripts/run_pion.sh
```

Switch suites / GPUs / steps via env vars — see the per-subfolder
`README.md` for the full set.
