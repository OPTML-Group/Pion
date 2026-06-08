#!/usr/bin/env bash
# AdamW baseline for RLVR (GRPO / GMPO) on Qwen3-1.7B or Qwen3-4B,
# evaluated on GSM8K (default) or MATH.
#
# Usage:
#   bash scripts/run_adamw.sh              # GRPO + Qwen3-1.7B + GSM8K
#
#   TRAINER=gmpo bash scripts/run_adamw.sh # GMPO + Qwen3-1.7B + GSM8K
#   MODEL=qwen3-4b DATASET=math TRAINER=gmpo bash scripts/run_adamw.sh
#
# Supported envs:
#   TRAINER  ∈ {grpo, gmpo}                 default grpo
#   MODEL    ∈ {qwen3-1.7b, qwen3-4b}       default qwen3-1.7b
#   DATASET  ∈ {gsm8k, math}                default gsm8k

set -euo pipefail
cd "$(dirname "$0")/.."

TRAINER=${TRAINER:-grpo}
MODEL=${MODEL:-qwen3-1.7b}
DATASET=${DATASET:-gsm8k}

case "${DATASET}" in
    gsm8k) suffix="" ;;
    math)  suffix="_math3-5" ;;
    *) echo "DATASET must be gsm8k or math (got ${DATASET})"; exit 1 ;;
esac

case "${TRAINER}" in
    grpo|gmpo) ;;
    *) echo "TRAINER must be grpo or gmpo (got ${TRAINER})"; exit 1 ;;
esac

case "${MODEL}" in
    qwen3-1.7b|qwen3-4b) ;;
    *) echo "MODEL must be qwen3-1.7b or qwen3-4b (got ${MODEL})"; exit 1 ;;
esac

script="examples/${TRAINER}_trainer/run_${MODEL}_fsdp2_adamw${suffix}.sh"
if [[ ! -f "${script}" ]]; then
    echo "Recipe not found: ${script}"; exit 1
fi

echo "[run_adamw] ${TRAINER} | ${MODEL} | ${DATASET} -> ${script}"
exec bash "${script}" "$@"
