#!/usr/bin/env bash
# Pion (high-pass NS) for VLANeXt on LIBERO.
#
# Routing (paper-recommended):
#   - Vision / Language backbones  -> Muon         (preserve pretraining)
#   - Action head                  -> DefaultPion  (high-pass NS)
#
# High-pass polynomial knobs:
#   --pion_promotion_steps   k_p (Promotion iterations,    default 0)
#   --pion_ns_steps          k_p + k_s (total NS steps,    default 5)
# Suppression iterations  k_s = pion_ns_steps - pion_promotion_steps.
# Pure-Suppression (k_p=0) is the recommended VLA setting.

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${CONFIG:-config/libero_train_pion_config.yaml}
TASK=${TASK:-libero_object_no_noops}
MAX_STEPS=${MAX_STEPS:-10000}
SAVE_INTERVAL=${SAVE_INTERVAL:-2000}
BATCH_SIZE=${BATCH_SIZE:-256}
GRAD_ACC=${GRAD_ACC:-1}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
NPROC=${NPROC:-8}
PORT=${PORT:-29501}

PROMOTION_STEPS=${PROMOTION_STEPS:-0}
NS_STEPS=${NS_STEPS:-5}

mkdir -p logs
LOG=logs/run_pion_${TASK}_kp${PROMOTION_STEPS}_$(date +%m%d-%H%M%S).log

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
    --optimizer_A DefaultPion \
    --optimizer_O Muon \
    --pion_promotion_steps "${PROMOTION_STEPS}" \
    --pion_ns_steps "${NS_STEPS}" \
    2>&1 | tee "${LOG}"
