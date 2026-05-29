#!/usr/bin/env bash
set -euo pipefail

MODEL_FAMILY="${1:-qwen2_5vl}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${MODEL_FAMILY}" in
  qwen2_5vl|qwen2.5vl)
    CONFIG="examples/train_full/videokr_qwen2_5vl_sft.yaml"
    DEFAULT_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
    DEFAULT_OUTPUT="saves/videokr/qwen2_5vl_sft"
    ;;
  qwen3vl|qwen3_vl)
    CONFIG="examples/train_full/videokr_qwen3vl_sft.yaml"
    DEFAULT_MODEL="Qwen/Qwen3-VL-8B-Instruct"
    DEFAULT_OUTPUT="saves/videokr/qwen3vl_sft"
    ;;
  *)
    echo "Usage: $0 {qwen2_5vl|qwen3vl}" >&2
    exit 2
    ;;
esac

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${DEFAULT_MODEL}}"
DATASET_DIR="${DATASET_DIR:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT}}"
REPORT_TO="${REPORT_TO:-none}"

export FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-2048001}"

llamafactory-cli train "${CONFIG}" \
  model_name_or_path="${MODEL_NAME_OR_PATH}" \
  dataset_dir="${DATASET_DIR}" \
  output_dir="${OUTPUT_DIR}" \
  report_to="${REPORT_TO}" \
  "$@"
