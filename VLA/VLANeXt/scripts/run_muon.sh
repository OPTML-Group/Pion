#!/usr/bin/env bash
# Muon baseline for VLANeXt on LIBERO.
#
# Routes V / L / A / O all to Muon (1-D params auto-fallback to AdamW).

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${CONFIG:-config/libero_train_muon_config.yaml}
TASK=${TASK:-libero_object_no_noops}
MAX_STEPS=${MAX_STEPS:-10000}
SAVE_INTERVAL=${SAVE_INTERVAL:-2000}
BATCH_SIZE=${BATCH_SIZE:-256}
GRAD_ACC=${GRAD_ACC:-1}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
NPROC=${NPROC:-8}
PORT=${PORT:-29501}

mkdir -p logs
LOG=logs/run_muon_${TASK}_$(date +%m%d-%H%M%S).log

CUDA_VISIBLE_DEVICES="${GPUS}" \
torchrun --nproc_per_node="${NPROC}" --master_port="${PORT}" \
    -m scripts.train_pion \
    --config "${CONFIG}" \
    --task_suite_name "${TASK}" \
    --max_steps "${MAX_STEPS}" \
    --save_interval "${SAVE_INTERVAL}" \
    --batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACC}" \
    --optimizer_V Muon \
    --optimizer_L Muon \
    --optimizer_A Muon \
    --optimizer_O Muon \
    2>&1 | tee "${LOG}"
