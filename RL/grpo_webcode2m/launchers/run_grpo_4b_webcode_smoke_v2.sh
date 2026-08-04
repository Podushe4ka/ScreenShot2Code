#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="${PROJECT:-$HOME/mla_project}"
VERL_ROOT="${VERL_ROOT:-$HOME/verl_workspace/verl}"
VENV="${VENV:-$HOME/verl_workspace/.venv-qwen35}"
STORAGE_ROOT="${STORAGE_ROOT:-/mnt/storage-1/$USER}"

MODEL="$STORAGE_ROOT/models/Qwen3.5-4B"
TRAIN_DATA="$STORAGE_ROOT/data/verl_webcode2m_test2/train.parquet"
VAL_DATA="$STORAGE_ROOT/data/verl_webcode2m_test2/validation.parquet"
REWARD_FILE="$PROJECT/screenshot_reward.py"

cd "$PROJECT"
source "$VENV/bin/activate"

export PYTHONPATH="$PROJECT:$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=1
export PLAYWRIGHT_WS_ENDPOINT="ws://127.0.0.1:3001/"

echo "=== Проверяем необходимые файлы ==="

for path in \
    "$MODEL/config.json" \
    "$TRAIN_DATA" \
    "$VAL_DATA" \
    "$REWARD_FILE"
do
    if [[ ! -e "$path" ]]; then
        echo "Не найден файл: $path"
        exit 1
    fi

    echo "Найден: $path"
done

echo
echo "=== Проверяем GPU 0 ==="

GPU_MEMORY=$(
    nvidia-smi \
        --id=1 \
        --query-gpu=memory.used \
        --format=csv,noheader,nounits |
    tr -d ' '
)

GPU_UTILIZATION=$(
    nvidia-smi \
        --id=1 \
        --query-gpu=utilization.gpu \
        --format=csv,noheader,nounits |
    tr -d ' '
)

echo "Занято памяти: ${GPU_MEMORY} MiB"
echo "Загрузка: ${GPU_UTILIZATION}%"

if [[ "$GPU_MEMORY" -gt 1000 ]]; then
    echo "GPU 2 уже занята — запуск остановлен"
    exit 1
fi

echo "GPU 2 свободна"

DATA=(
    data.train_files="$TRAIN_DATA"
    data.val_files="$VAL_DATA"
    data.train_batch_size=1
    data.max_prompt_length=4096
    data.max_response_length=4096
    data.image_key=images
    data.image_patch_size=16
    data.shuffle=False
    data.truncation=error
    +data.apply_chat_template_kwargs.enable_thinking=False \
)

REWARD=(
    reward.custom_reward_function.path="$REWARD_FILE"
    reward.custom_reward_function.name=compute_score
)

MODEL_CONFIG=(
    actor_rollout_ref.model.path="$MODEL"
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.enable_gradient_checkpointing=True

    actor_rollout_ref.model.lora_rank=16
    actor_rollout_ref.model.lora_alpha=32
    actor_rollout_ref.model.target_modules=all-linear
    actor_rollout_ref.model.exclude_modules='.*visual.*'

    actor_rollout_ref.model.lora.merge=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=2e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=1
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_dynamic_bsz=False

    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0.0

    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.fsdp_config.fsdp_size=1
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.use_torch_compile=False
)

REF=(
    actor_rollout_ref.ref.strategy=fsdp2
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.use_torch_compile=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.prompt_length=4096
    actor_rollout_ref.rollout.response_length=4096

    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35
    actor_rollout_ref.rollout.n=2

    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.max_num_batched_tokens=8192

    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.layered_summon=True
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console"]'
    trainer.project_name=screenshot2code_grpo

    trainer.n_gpus_per_node=1
    trainer.nnodes=1
    trainer.balance_batch=False

    trainer.val_before_train=False
    trainer.test_freq=999999
    trainer.save_freq=1
    trainer.total_epochs=1


    ++ray_kwargs.ray_init._temp_dir=$STORAGE_ROOT/r4bw2
    trainer.experiment_name=qwen35_4b_webcode_smoke_4096_v2
    trainer.default_local_dir="$STORAGE_ROOT/checkpoints/grpo_4b_webcode_smoke_4096_v2"
    trainer.rollout_data_dir="$PROJECT/outputs/grpo_4b_webcode_smoke_4096_v2/rollouts"
    trainer.validation_data_dir="$PROJECT/outputs/grpo_4b_webcode_smoke_4096_v2/validation"
)

mkdir -p \
    "$PROJECT/logs/grpo_4b_webcode_smoke_4096_v2" \
    "$PROJECT/checkpoints/grpo_4b_webcode_smoke_4096_v2"

echo
echo "=== Запускаем GRPO smoke test ==="

python -m verl.trainer.main_ppo \
    "${ALGORITHM[@]}" \
    "${DATA[@]}" \
    "${MODEL_CONFIG[@]}" \
    "${REWARD[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    2>&1 | tee "$PROJECT/logs/grpo_4b_webcode_smoke_4096_v2/run.log"