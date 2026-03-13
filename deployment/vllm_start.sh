#!/bin/bash

set -euo pipefail

select_model_and_alias() {
  local choice="${LLM_CHOICE:-}"
  case "${choice}" in
    GEMMA3)
      echo "${VLLM_MODEL:-google/gemma-3-12b-it}|${VLLM_SERVED_MODEL_NAME:-ebdm/gemma3-enhanced:12b}"
      ;;
    LLAMA3.2)
      echo "${VLLM_MODEL:-meta-llama/Llama-3.2-3B-Instruct}|${VLLM_SERVED_MODEL_NAME:-llama3.2}"
      ;;
    GPT-OSS)
      echo "${VLLM_MODEL:-openai/gpt-oss-120b}|${VLLM_SERVED_MODEL_NAME:-gpt-oss:120b}"
      ;;
    QWEN3_30B)
      echo "${VLLM_MODEL:-Qwen/Qwen2.5-32B-Instruct}|${VLLM_SERVED_MODEL_NAME:-qwen3.5:27b-q8_0}"
      ;;
    *)
      echo "${VLLM_MODEL:-Qwen/Qwen2.5-32B-Instruct}|${VLLM_SERVED_MODEL_NAME:-qwen3.5:27b-q8_0}"
      ;;
  esac
}

IFS='|' read -r MODEL_TO_SERVE SERVED_MODEL_NAME <<< "$(select_model_and_alias)"

HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"

cmd=(python -m vllm.entrypoints.openai.api_server
  --host "${HOST}"
  --port "${PORT}"
  --model "${MODEL_TO_SERVE}"
  --served-model-name "${SERVED_MODEL_NAME}"
)

if [ -n "${VLLM_API_KEY:-}" ]; then
  cmd+=(--api-key "${VLLM_API_KEY}")
fi

if [ -n "${VLLM_TENSOR_PARALLEL_SIZE:-}" ]; then
  cmd+=(--tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}")
fi

if [ -n "${VLLM_GPU_MEMORY_UTILIZATION:-}" ]; then
  cmd+=(--gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}")
fi

if [ -n "${VLLM_MAX_MODEL_LEN:-}" ]; then
  cmd+=(--max-model-len "${VLLM_MAX_MODEL_LEN}")
fi

if [ -n "${VLLM_DTYPE:-}" ]; then
  cmd+=(--dtype "${VLLM_DTYPE}")
fi

if [ -n "${VLLM_TASK:-}" ]; then
  cmd+=(--task "${VLLM_TASK}")
fi

if [ -n "${VLLM_DOWNLOAD_DIR:-}" ]; then
  cmd+=(--download-dir "${VLLM_DOWNLOAD_DIR}")
fi

if [ -n "${VLLM_EXTRA_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  extra_args=( ${VLLM_EXTRA_ARGS} )
  cmd+=("${extra_args[@]}")
fi

echo "Starting vLLM with model='${MODEL_TO_SERVE}' served_as='${SERVED_MODEL_NAME}' on ${HOST}:${PORT}"
exec "${cmd[@]}"
