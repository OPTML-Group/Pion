#!/usr/bin/env bash
# AdamW baseline for VLA-Adapter on LIBERO.
#
# Uses the same training entrypoint as run_muon.sh / run_pion.sh but routes
# every module (Vision, Language, Action head, Other) to AdamW.

set -euo pipefail

# Ensure `pion_optim` (a top-level package under VLAAdapter/) is importable
# when finetune_pion.py is launched via `python vla-scripts/finetune_pion.py`
# (in which case sys.path[0] = vla-scripts/, NOT the project root).
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

current_time=$(date +%m%d-%H%M%S)
data_name=${DATA_NAME:-libero_object_no_noops}   # e.g. libero_object/spatial/goal/10
max_steps=${MAX_STEPS:-1500}
save_freq=${SAVE_FREQ:-500}
batch_size=${BATCH_SIZE:-8}
lr=${LR:-1e-4}
wd=${WD:-1e-2}
nproc=${NPROC:-8}
out_dir=${OUT_DIR:-outputs/adamw}
run_id_note=${RUN_ID_NOTE:-AdamW-${data_name}-${current_time}}
log_file=logs/${run_id_note}.log
mkdir -p "${out_dir}" logs

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} \
torchrun --standalone --nnodes 1 --nproc-per-node "${nproc}" vla-scripts/finetune_pion.py \
  --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --config_file_path pretrained_models/configs \
  --data_root_dir data/libero \
  --dataset_name "${data_name}" \
  --run_root_dir "${out_dir}" \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --use_lora False \
  --use_fz True \
  --use_minivlm True \
  --image_aug True \
  --num_steps_before_decay 100000 \
  --max_steps "${max_steps}" \
  --save_freq "${save_freq}" \
  --save_latest_checkpoint_only False \
  --merge_lora_during_training True \
  --batch_size "${batch_size}" \
  --grad_accumulation_steps 1 \
  --learning_rate "${lr}" \
  --weight_decay "${wd}" \
  --lr_ratio 1.0 \
  --lora_rank 64 \
  --use_pro_version True \
  --wandb_project "${data_name}" \
  --run_id_note "${run_id_note}" \
  --optimizer_V AdamW \
  --optimizer_L AdamW \
  --optimizer_A AdamW \
  --optimizer_O AdamW \
  2>&1 | tee "${log_file}"
