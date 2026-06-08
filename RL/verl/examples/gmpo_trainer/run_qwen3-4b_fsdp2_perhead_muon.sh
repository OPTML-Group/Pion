# Per-head PerHeadMuonAdamW on Qwen3-4B with GMPO / GSM8K.
#
# Qwen3-4B GQA topology: 32 Q-heads, 8 KV-heads, head_dim=128, hidden=2560
#   Q/K/V (head_split_dim=0): each head orthogonalized as (head_dim, in_dim)
#   O     (head_split_dim=1): each head orthogonalized as (out_dim, head_dim)
# All four projections receive the same scale  0.2 * sqrt(max(head_dim, in_dim)).
# FFN weights fall back to whole-matrix Muon; embed/lm_head use AdamW.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1 bash examples/gmpo_trainer/run_qwen3-4b_fsdp2_perhead_muon.sh

set -x

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.policy_loss.loss_mode=geo_mean \
    actor_rollout_ref.actor.clip_ratio_low=0.4 \
    actor_rollout_ref.actor.clip_ratio_high=0.4 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_gmpo_example_gsm8k' \
    trainer.experiment_name='qwen3_4b_function_rm_fsdp2_perhead_muon' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
    actor_rollout_ref.ref.fsdp_config.use_orig_params=True \
    actor_rollout_ref.actor.optim.optimizer_impl='verl.utils.muon' \
    actor_rollout_ref.actor.optim.optimizer='PerHeadMuonAdamW' \
    +actor_rollout_ref.actor.optim.override_optimizer_config.ns_steps=5 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.scale_factor=5.0 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.num_q_heads=32 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.num_kv_heads=8 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.muon_print_param_groups=true \
    +actor_rollout_ref.actor.optim.override_optimizer_config.muon_exclude_names='["embed","lm_head"]' $@
