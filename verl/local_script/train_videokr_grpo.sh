#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

if [[ -z "${MODEL_PATH:-}" ]]; then
  echo "MODEL_PATH is required, e.g. MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct" >&2
  exit 1
fi

DATA_DIR="${DATA_DIR:-data/videokr_rl}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train.parquet}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/test.parquet}"

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "Missing train parquet: ${TRAIN_FILE}" >&2
  exit 1
fi
if [[ ! -f "${VAL_FILE}" ]]; then
  echo "Missing validation parquet: ${VAL_FILE}" >&2
  exit 1
fi

PROJECT_NAME="${PROJECT_NAME:-videokr_grpo}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-videokr_qwen_vl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${PROJECT_NAME}/${EXPERIMENT_NAME}}"

N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-${N_GPUS:-8}}"
NNODES="${NNODES:-1}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-16384}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-True}"
USE_FUSED_KERNELS="${USE_FUSED_KERNELS:-True}"

ROLLOUT_N="${ROLLOUT_N:-8}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.6}"
AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-8}"

PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}"
LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-8}"

SAVE_FREQ="${SAVE_FREQ:-20}"
TEST_FREQ="${TEST_FREQ:-20}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TRAINER_LOGGER="${TRAINER_LOGGER:-[\"console\"]}"

QWEN3_PATCH_MODE="${VERL_PATCH_QWEN3_VIDEO:-auto}"
MODEL_PATH_LOWER="$(printf '%s' "${MODEL_PATH}" | tr '[:upper:]' '[:lower:]')"
if [[ "${QWEN3_PATCH_MODE}" == "auto" ]]; then
  if [[ "${MODEL_PATH_LOWER}" == *"qwen3"* && "${MODEL_PATH_LOWER}" == *"vl"* ]]; then
    export VERL_PATCH_QWEN3_VIDEO=1
  else
    unset VERL_PATCH_QWEN3_VIDEO || true
  fi
else
  export VERL_PATCH_QWEN3_VIDEO="${QWEN3_PATCH_MODE}"
fi

MODEL_OVERRIDES=()
if [[ -n "${ATTN_IMPLEMENTATION}" ]]; then
  MODEL_OVERRIDES+=(+actor_rollout_ref.model.override_config.attn_implementation="${ATTN_IMPLEMENTATION}")
fi

mkdir -p "${OUTPUT_DIR}/rollout_data"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.video_key=videos \
  data.return_multi_modal_inputs=True \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.actor.optim.lr="${LR:-5e-6}" \
  "${MODEL_OVERRIDES[@]}" \
  actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}" \
  actor_rollout_ref.model.use_fused_kernels="${USE_FUSED_KERNELS}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF:-0.01}" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload="${PARAM_OFFLOAD:-False}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${OPTIMIZER_OFFLOAD:-False}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOGPROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}" \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager="${ENFORCE_EAGER:-False}" \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_NUM_WORKERS}" \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOGPROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.logger="${TRAINER_LOGGER}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  trainer.nnodes="${NNODES}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.default_local_dir="${OUTPUT_DIR}/checkpoints" \
  trainer.rollout_data_dir="${OUTPUT_DIR}/rollout_data" \
  "$@"
