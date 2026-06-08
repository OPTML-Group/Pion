# Per-head PerHeadMuonAdamW on Qwen3-1.7B with GRPO / MATH Level 3-5.
# Training data from arxiv:2503.20783 (MATH Level 3-5), using standard GRPO.
#
# Qwen3-1.7B GQA topology: 16 Q-heads, 8 KV-heads, head_dim=128, hidden=2048
#   Q/K/V (head_split_dim=0): each head orthogonalized as (head_dim, in_dim)
#   O     (head_split_dim=1): each head orthogonalized as (out_dim, head_dim)
# All four projections receive the same scale  0.2 * sqrt(max(head_dim, in_dim)).
# FFN weights fall back to whole-matrix Muon; embed/lm_head use AdamW.
# Configured for 2 GPUs (TP=2 for vLLM).
#
# Prerequisite:
#   MATH_MIN_LEVEL=3 MATH_MAX_LEVEL=5 python examples/data_preprocess/math_dataset.py \
#       --local_save_dir ~/data/math_level3-5
#   python examples/data_preprocess/math500.py --local_save_dir ~/data/math500
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1 bash examples/grpo_trainer/run_qwen3-1.7b_fsdp2_perhead_muon_math3-5.sh

set -x

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=$HOME/data/math_level3-5/train.parquet \
    data.val_files=$HOME/data/math500/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=3000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_math3-5' \
    trainer.experiment_name='qwen3_1.7b_grpo_fsdp2_perhead_muon' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=20 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
    actor_rollout_ref.ref.fsdp_config.use_orig_params=True \
    actor_rollout_ref.actor.optim.optimizer_impl='verl.utils.muon' \
    actor_rollout_ref.actor.optim.optimizer='PerHeadMuonAdamW' \
    +actor_rollout_ref.actor.optim.override_optimizer_config.ns_steps=5 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.num_q_heads=16 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.num_kv_heads=8 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.muon_print_param_groups=true \
    +actor_rollout_ref.actor.optim.override_optimizer_config.muon_exclude_names='["embed","lm_head"]' $@
