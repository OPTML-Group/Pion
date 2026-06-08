#!/usr/bin/env bash
# Train pi05 finetune with Muon on all 3 tasks, sequentially.
#
# Usage:
#   bash scripts/run_muon.sh
#
# Routing scheme for 'muon':
#   - ndim>=2 non-embed/output params -> Muon
#   - everything else (1-D, embed, output head, residual) -> AdamW
#
# Tasks are hardcoded below; edit the TASKS array if you want to run a
# subset. All other hyperparams (20k steps, save every 5000 -> ckpts at
# {5000, 10000, 15000, 19999}, 8 GPUs, batch=32, lr/scheduler from config)
# are fixed.

set -euo pipefail

# ===== Tasks =====
TASKS=(
    "cubic_to_bowl"
    "cubic_to_plate"
    "cucumber_to_plate"
)

OPTIM="muon"
EXP_NAME="${OPTIM}"

cd "$(dirname "$0")/.."
REPO_DIR="$(pwd)"

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}"
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"

# Fixed training hyperparams.
# 20k steps, save every 5000 -> ckpts at {5000, 10000, 15000, 19999}, ALL
# retained (PyTorch save_checkpoint never deletes old ones, so keep-period
# is a no-op).
NUM_TRAIN_STEPS=20000
SAVE_INTERVAL=5000
KEEP_PERIOD=5000
FINAL_STEP=$((NUM_TRAIN_STEPS - 1))  # train_pytorch.py saves at step == num_train_steps-1

for TASK in "${TASKS[@]}"; do
    CONFIG_NAME="pi05_${TASK}_finetune"
    PROJECT_NAME="pi05_${TASK}_finetune"
    LOG_DIR="torchrun_logs/${CONFIG_NAME}_${EXP_NAME}"
    CKPT_DIR="${REPO_DIR}/checkpoints/${CONFIG_NAME}/${EXP_NAME}"
    mkdir -p "${LOG_DIR}"

    # ===== Smart resume / skip logic =====
    # 1) final-step ckpt exists      -> skip (already finished)
    # 2) some intermediate ckpt      -> --resume from latest
    # 3) no ckpt                     -> --overwrite (fresh start; harmless if dir doesn't exist)
    RUN_FLAG="--overwrite"
    LATEST_CKPT=""
    SKIP_TASK=0
    if [[ -d "${CKPT_DIR}/${FINAL_STEP}" ]]; then
        SKIP_TASK=1
    elif [[ -d "${CKPT_DIR}" ]]; then
        # NOTE: trailing `|| true` is mandatory. bash 5.0.x triggers `set -e`
        # exit on `var=$(failing_pipe)` (the `grep` returns 1 when no numeric
        # ckpt subdir exists, e.g. a dir holding only `wandb_id.txt` from an
        # earlier run that crashed before the first save).
        LATEST_CKPT=$(ls -1 "${CKPT_DIR}" 2>/dev/null | grep -E "^[0-9]+$" | sort -n | tail -1 || true)
        if [[ -n "${LATEST_CKPT}" ]]; then
            RUN_FLAG="--resume"
            # Clean leftover tmp_* dirs from interrupted saves so they don't confuse anything.
            find "${CKPT_DIR}" -maxdepth 1 -name "tmp_*" -type d -exec rm -rf {} + 2>/dev/null || true
        fi
    fi

    echo
    echo "################################################################"
    if [[ ${SKIP_TASK} -eq 1 ]]; then
        echo "#  [${OPTIM}]  task='${TASK}'  -> SKIP (already completed: ${CKPT_DIR}/${FINAL_STEP} exists)"
        echo "################################################################"
        continue
    fi
    echo "#  [${OPTIM}]  task='${TASK}'"
    echo "#  config        : ${CONFIG_NAME}"
    echo "#  exp_name      : ${EXP_NAME}"
    echo "#  wandb project : ${PROJECT_NAME}"
    echo "#  wandb run name: ${EXP_NAME}"
    echo "#  ckpt dir      : ${CKPT_DIR}"
    echo "#  log  dir      : ${REPO_DIR}/${LOG_DIR}"
    echo "#  resume mode   : ${RUN_FLAG}${LATEST_CKPT:+ (latest step=${LATEST_CKPT})}"
    echo "################################################################"

    # LR schedule: warmup-cosine-decay (openpi default, identical to
    # pi05_droid_finetune). Schedule decay_steps=30000 vs our 20k train steps,
    # so by step 20k the cosine decay has driven lr down to ~30% of peak.
    .venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=8 \
        --log-dir="${LOG_DIR}" --redirects=3 --tee=3 \
        scripts/train_pytorch.py "${CONFIG_NAME}" \
        --exp-name="${EXP_NAME}" \
        --pytorch-optimizer-name="${OPTIM}" \
        --project-name="${PROJECT_NAME}" \
        --num-train-steps=${NUM_TRAIN_STEPS} \
        --save-interval=${SAVE_INTERVAL} \
        --keep-period=${KEEP_PERIOD} \
        --log-interval=20 \
        --lr-schedule.warmup-steps=1000 \
        --lr-schedule.peak-lr=2.5e-5 \
        --lr-schedule.decay-steps=30000 \
        --lr-schedule.decay-lr=2.5e-6 \
        ${RUN_FLAG}

    echo "[${OPTIM}/${TASK}] done."
done

echo
echo "All Muon runs finished: ${TASKS[*]}"
