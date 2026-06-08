#!/usr/bin/env bash
# Muon baseline for RLVR (GRPO / GMPO) on Qwen3-1.7B or Qwen3-4B,
# evaluated on GSM8K (default) or MATH.
#
# Defaults to *whole-matrix Muon* (MuonAdamW), the standard Muon baseline
# from the paper.  Set FLAVOR=perhead to use the per-head variant
# (PerHeadMuonAdamW).
#
# Usage:
#   bash scripts/run_muon.sh                            # GRPO + Qwen3-1.7B + GSM8K + whole-matrix Muon
#   MODEL=qwen3-4b DATASET=math bash scripts/run_muon.sh
#                                                      # GRPO + Qwen3-4B + MATH + whole-matrix Muon
#   FLAVOR=perhead bash scripts/run_muon.sh             # per-head Muon
#   TRAINER=gmpo FLAVOR=perhead bash scripts/run_muon.sh
#                                                      # GMPO (per-head only)
#
# Supported envs:
#   TRAINER  ∈ {grpo, gmpo}                 default grpo
#   MODEL    ∈ {qwen3-1.7b, qwen3-4b}       default qwen3-1.7b
#   DATASET  ∈ {gsm8k, math}                default gsm8k
#   FLAVOR   ∈ {whole, perhead}             default whole

set -euo pipefail
cd "$(dirname "$0")/.."

TRAINER=${TRAINER:-grpo}
MODEL=${MODEL:-qwen3-1.7b}
DATASET=${DATASET:-gsm8k}
FLAVOR=${FLAVOR:-whole}

case "${DATASET}" in
    gsm8k) suffix="" ;;
    math)  suffix="_math3-5" ;;
    *) echo "DATASET must be gsm8k or math (got ${DATASET})"; exit 1 ;;
esac

case "${FLAVOR}" in
    whole)   tag="muon" ;;
    perhead) tag="perhead_muon" ;;
    *) echo "FLAVOR must be whole or perhead (got ${FLAVOR})"; exit 1 ;;
esac

script="examples/${TRAINER}_trainer/run_${MODEL}_fsdp2_${tag}${suffix}.sh"
if [[ ! -f "${script}" ]]; then
    if [[ "${TRAINER}" == "gmpo" && "${FLAVOR}" == "whole" ]]; then
        echo "Recipe not found: ${script} (GMPO has no whole-matrix Muon recipe; use FLAVOR=perhead)"
    else
        echo "Recipe not found: ${script}"
    fi
    exit 1
fi

echo "[run_muon] ${TRAINER} | ${MODEL} | ${DATASET} | ${FLAVOR} -> ${script}"
exec bash "${script}" "$@"
