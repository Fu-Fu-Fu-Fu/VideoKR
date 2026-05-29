#!/usr/bin/env bash
set -euo pipefail

if [[ -f .env ]]; then
    while IFS='=' read -r key value; do
        [[ -z "${key}" || "${key}" =~ ^[[:space:]]*# ]] && continue
        key="${key%%[[:space:]]*}"
        [[ "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
        if [[ -z "${!key+x}" ]]; then
            value="${value%$'\r'}"
            value="${value%\"}"
            value="${value#\"}"
            value="${value%\'}"
            value="${value#\'}"
            export "${key}=${value}"
        fi
    done < .env
fi

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-40960}"

# Judge settings. If no usable API credentials are provided, tasks fall back to
# rule-based scoring for multiple-choice questions and skip open-ended questions.
export API_TYPE="${API_TYPE:-azure}"
export MODEL_VERSION="${MODEL_VERSION:-gpt-4o}"
export API_VERSION="${API_VERSION:-2024-02-15-preview}"
export AZURE_API_KEY=${AZURE_API_KEY:-}
export AZURE_ENDPOINT="${AZURE_ENDPOINT:-}"
export OPENAI_API_KEY=${OPENAI_API_KEY:-}
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
export OPENAI_API_URL="${OPENAI_API_URL:-${OPENAI_BASE_URL%/}/chat/completions}"

if [[ -z "${VIDEOKR_USE_VLM_JUDGE:-}" ]]; then
    if [[ "${API_TYPE}" == "azure" ]]; then
        if [[ -n "${AZURE_API_KEY}" && -n "${AZURE_ENDPOINT}" ]]; then
            export VIDEOKR_USE_VLM_JUDGE=1
        else
            export VIDEOKR_USE_VLM_JUDGE=0
        fi
    elif [[ "${API_TYPE}" == "openai" ]]; then
        if [[ -n "${OPENAI_API_KEY}" ]]; then
            export VIDEOKR_USE_VLM_JUDGE=1
        else
            export VIDEOKR_USE_VLM_JUDGE=0
        fi
    else
        export VIDEOKR_USE_VLM_JUDGE=0
    fi
fi

MODEL="${VIDEOKR_MODEL:?Set VIDEOKR_MODEL to your fine-tuned VideoKR checkpoint or Hugging Face repo.}"
TASKS="${TASKS:-videokr_eval}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_FRAME_NUM="${MAX_FRAME_NUM:-128}"
MAX_PIXELS="${MAX_PIXELS:-200704}"
FPS="${FPS:-2}"
LIMIT="${LIMIT:-}"
GEN_KWARGS="${GEN_KWARGS:-}"

MODEL_NAME="${RUN_NAME:-$(basename "${MODEL}")}"
MODEL_NAME="${MODEL_NAME//[^A-Za-z0-9._-]/_}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/videokr/${MODEL_NAME}_vllm}"
LOG_SUFFIX="${LOG_SUFFIX:-videokr_${MODEL_NAME}_vllm}"

LIMIT_ARG=()
if [[ -n "${LIMIT}" ]]; then
    LIMIT_ARG=(--limit "${LIMIT}")
fi

GEN_KWARGS_ARG=()
if [[ -n "${GEN_KWARGS}" ]]; then
    GEN_KWARGS_ARG=(--gen_kwargs "${GEN_KWARGS}")
fi

python -m lmms_eval \
    --model vllm_generate \
    --model_args "model=${MODEL},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},data_parallel_size=${DATA_PARALLEL_SIZE},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},max_model_len=${MAX_MODEL_LEN},fps=${FPS},max_pixels=${MAX_PIXELS},max_frame_num=${MAX_FRAME_NUM},disable_log_stats=True" \
    --tasks "${TASKS}" \
    --batch_size "${BATCH_SIZE}" \
    "${LIMIT_ARG[@]}" \
    "${GEN_KWARGS_ARG[@]}" \
    --log_samples \
    --log_samples_suffix "${LOG_SUFFIX}" \
    --output_path "${OUTPUT_PATH}"
