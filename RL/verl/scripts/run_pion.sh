#!/usr/bin/env bash
# Pion (high-pass NS) for RLVR (GRPO / GMPO) on Qwen3-1.7B or Qwen3-4B,
# evaluated on GSM8K (default) or MATH.
#
# The paper's RLVR setup uses *per-head Pion* (PerHeadPionAdamW).
# Set FLAVOR=whole to use the whole-matrix variant (DefaultPionAdamW).
#
# High-pass NS knobs:
#   PROMOTION_STEPS  k_p (Promotion iterations), default 0 -> pure Suppression
#   NS_STEPS      k_p + k_s (total),          set inside the upstream scripts
#
# Usage:
#   bash scripts/run_pion.sh                              # GRPO + Qwen3-1.7B + GSM8K + per-head Pion
#   TRAINER=gmpo MODEL=qwen3-4b DATASET=math \
#       bash scripts/run_pion.sh                          # GMPO + Qwen3-4B + MATH + per-head Pion
#   PROMOTION_STEPS=1 bash scripts/run_pion.sh               # 1 Promotion + 4 Suppression
#   FLAVOR=whole bash scripts/run_pion.sh                 # whole-matrix Pion (GRPO only)
#
# Supported envs:
#   TRAINER       ∈ {grpo, gmpo}                 default grpo
#   MODEL         ∈ {qwen3-1.7b, qwen3-4b}       default qwen3-1.7b
#   DATASET       ∈ {gsm8k, math}                default gsm8k
#   FLAVOR        ∈ {perhead, whole}             default perhead
#   PROMOTION_STEPS  k_p, propagated to the upstream script (default 0)

set -euo pipefail
cd "$(dirname "$0")/.."

TRAINER=${TRAINER:-grpo}
MODEL=${MODEL:-qwen3-1.7b}
DATASET=${DATASET:-gsm8k}
FLAVOR=${FLAVOR:-perhead}
export PROMOTION_STEPS=${PROMOTION_STEPS:-0}

case "${DATASET}" in
    gsm8k) suffix="" ;;
    math)  suffix="_math3-5" ;;
    *) echo "DATASET must be gsm8k or math (got ${DATASET})"; exit 1 ;;
esac

case "${FLAVOR}" in
    perhead) tag="perhead_pion" ;;
    whole)   tag="pion" ;;
    *) echo "FLAVOR must be perhead or whole (got ${FLAVOR})"; exit 1 ;;
esac

script="examples/${TRAINER}_trainer/run_${MODEL}_fsdp2_${tag}${suffix}.sh"
if [[ ! -f "${script}" ]]; then
    echo "Recipe not found: ${script} (whole-matrix Pion is only provided for GRPO)"; exit 1
fi

echo "[run_pion] ${TRAINER} | ${MODEL} | ${DATASET} | ${FLAVOR} | k_p=${PROMOTION_STEPS} -> ${script}"
exec bash "${script}" "$@"
