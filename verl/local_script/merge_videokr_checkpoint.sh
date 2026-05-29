#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

usage() {
  cat <<'EOF'
Usage:
  bash local_script/merge_videokr_checkpoint.sh CHECKPOINT_DIR [TARGET_DIR]

Examples:
  bash local_script/merge_videokr_checkpoint.sh outputs/videokr_grpo/videokr_qwen_vl/checkpoints/global_step_20/actor
  DRY_RUN=1 bash local_script/merge_videokr_checkpoint.sh outputs/videokr_grpo/videokr_qwen_vl/checkpoints/global_step_20/actor

Environment variables:
  BACKEND=fsdp|megatron              Checkpoint backend. Default: fsdp
  TRUST_REMOTE_CODE=0|1              Pass --trust-remote-code. Default: 0
  USE_CPU_INITIALIZATION=0|1         Pass --use_cpu_initialization. Default: 0
  TIE_WORD_EMBEDDING=0|1             Megatron-only --tie-word-embedding. Default: 0
  IS_VALUE_MODEL=0|1                 Megatron-only --is-value-model. Default: 0
  HF_UPLOAD_PATH=org/model           Optional Hugging Face model repo to upload
  PRIVATE=0|1                        Upload model as private when HF_UPLOAD_PATH is set. Default: 0
  DRY_RUN=0|1                        Print the command without running it. Default: 0
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

CHECKPOINT_DIR="${1:-${CHECKPOINT_DIR:-}}"
if [[ -z "${CHECKPOINT_DIR}" ]]; then
  usage >&2
  echo "CHECKPOINT_DIR is required." >&2
  exit 1
fi

BACKEND="${BACKEND:-fsdp}"
if [[ "${BACKEND}" != "fsdp" && "${BACKEND}" != "megatron" ]]; then
  echo "BACKEND must be fsdp or megatron, got: ${BACKEND}" >&2
  exit 1
fi

CHECKPOINT_DIR="${CHECKPOINT_DIR%/}"
TARGET_DIR="${2:-${TARGET_DIR:-${CHECKPOINT_DIR}/huggingface}}"

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "Checkpoint directory does not exist: ${CHECKPOINT_DIR}" >&2
  exit 1
fi

if [[ ! -d "${CHECKPOINT_DIR}/huggingface" ]]; then
  echo "Missing Hugging Face config directory: ${CHECKPOINT_DIR}/huggingface" >&2
  echo "The expected CHECKPOINT_DIR is usually .../checkpoints/global_step_x/actor." >&2
  exit 1
fi

if [[ "${BACKEND}" == "fsdp" ]]; then
  if [[ ! -f "${CHECKPOINT_DIR}/fsdp_config.json" ]]; then
    echo "Missing FSDP config: ${CHECKPOINT_DIR}/fsdp_config.json" >&2
    exit 1
  fi
  if ! compgen -G "${CHECKPOINT_DIR}/model_world_size_*_rank_0.pt" >/dev/null; then
    echo "Missing FSDP model shard: ${CHECKPOINT_DIR}/model_world_size_*_rank_0.pt" >&2
    exit 1
  fi
fi

cmd=(
  python3 -m verl.model_merger merge
  --backend "${BACKEND}"
  --local_dir "${CHECKPOINT_DIR}"
  --target_dir "${TARGET_DIR}"
)

if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  cmd+=(--trust-remote-code)
fi
if [[ "${USE_CPU_INITIALIZATION:-0}" == "1" ]]; then
  cmd+=(--use_cpu_initialization)
fi
if [[ "${TIE_WORD_EMBEDDING:-0}" == "1" ]]; then
  cmd+=(--tie-word-embedding)
fi
if [[ "${IS_VALUE_MODEL:-0}" == "1" ]]; then
  cmd+=(--is-value-model)
fi
if [[ -n "${HF_UPLOAD_PATH:-}" ]]; then
  cmd+=(--hf_upload_path "${HF_UPLOAD_PATH}")
  if [[ "${PRIVATE:-0}" == "1" ]]; then
    cmd+=(--private)
  fi
fi

echo "Merging checkpoint:"
echo "  checkpoint: ${CHECKPOINT_DIR}"
echo "  target:     ${TARGET_DIR}"
printf '  command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

"${cmd[@]}"
