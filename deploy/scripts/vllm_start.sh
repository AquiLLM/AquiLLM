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
      echo "${VLLM_MODEL:-Qwen/Qwen2.5-32B-Instruct}|${VLLM_SERVED_MODEL_NAME:-qwen3.5:27b}"
      ;;
    *)
      echo "${VLLM_MODEL:-Qwen/Qwen2.5-32B-Instruct}|${VLLM_SERVED_MODEL_NAME:-qwen3.5:27b}"
      ;;
  esac
}

IFS='|' read -r MODEL_TO_SERVE SERVED_MODEL_NAME <<< "$(select_model_and_alias)"

HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"

# If .env omits OCR_VLLM_EXTRA_ARGS, full BF16/BFloat16 weights (~15+ GiB for 7B-VL) often exhaust KV budget at low gpu_memory_utilization.
# Same flags as .env.example / typical OCR_VLLM_EXTRA_ARGS (bitsandbytes 4-bit).
_DEFAULT_OCR_VLLM_EXTRA_ARGS="--quantization bitsandbytes --load-format bitsandbytes --dtype float16 --model-loader-extra-config '{\"load_in_4bit\":true,\"bnb_4bit_compute_dtype\":\"float16\",\"bnb_4bit_quant_type\":\"nf4\",\"bnb_4bit_use_double_quant\":true}'"

# Compose sometimes injects VLLM_EXTRA_ARGS="" when ${VAR:-} interpolation is empty on the host,
# which overrides env_file. Recover from the service-specific *VLLM_EXTRA_ARGS in the same .env.
if [ -z "${VLLM_EXTRA_ARGS// }" ]; then
  case "${VLLM_TASK:-}" in
    score) export VLLM_EXTRA_ARGS="${APP_RERANK_VLLM_EXTRA_ARGS:-}" ;;
  esac
fi
if [ -z "${VLLM_EXTRA_ARGS// }" ] && [ "${VLLM_RUNNER:-}" = "pooling" ] && [ -z "${VLLM_TASK:-}" ]; then
  case "${VLLM_MODEL:-}" in
    *Embedding*|*embedding*) export VLLM_EXTRA_ARGS="${MEM0_EMBED_VLLM_EXTRA_ARGS:-}" ;;
  esac
fi
if [ -z "${VLLM_EXTRA_ARGS// }" ]; then
  case "${VLLM_MODEL:-}" in
    *whisper*|*Whisper*) export VLLM_EXTRA_ARGS="${TRANSCRIBE_VLLM_EXTRA_ARGS:-}" ;;
    *Qwen2.5-VL*|*Qwen/Qwen2.5-VL*)
      if [ -n "${OCR_VLLM_EXTRA_ARGS// }" ]; then
        export VLLM_EXTRA_ARGS="${OCR_VLLM_EXTRA_ARGS}"
      else
        export VLLM_EXTRA_ARGS="${_DEFAULT_OCR_VLLM_EXTRA_ARGS}"
      fi
      ;;
  esac
fi

detect_python_bin() {
  if [ -n "${VLLM_PYTHON_BIN:-}" ] && command -v "${VLLM_PYTHON_BIN}" >/dev/null 2>&1; then
    echo "${VLLM_PYTHON_BIN}"
    return 0
  fi
  for candidate in python3 python; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

if ! PYTHON_BIN="$(detect_python_bin)"; then
  echo "ERROR: No python interpreter found in container PATH (tried python3/python)." >&2
  exit 127
fi

supports_arg() {
  local arg_name="$1"
  "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server --help 2>&1 | grep -q -- "${arg_name}"
}

resolve_gguf_model_path() {
  local spec="$1"
  if [[ "${spec}" == */*:* && "${spec}" != /* ]]; then
    local repo_id="${spec%%:*}"
    local selector="${spec#*:}"
    if [ -z "${selector}" ]; then
      echo "ERROR: Invalid GGUF model spec '${spec}' (missing filename or selector after ':')." >&2
      return 1
    fi
    local dl_dir="${VLLM_DOWNLOAD_DIR:-/root/.cache/huggingface/gguf}"
    mkdir -p "${dl_dir}"
    echo "Resolving GGUF selector '${selector}' from '${repo_id}'..." >&2
    local local_path
    if ! local_path="$("${PYTHON_BIN}" - "${repo_id}" "${selector}" "${dl_dir}" <<'PY'
from huggingface_hub import hf_hub_download, list_repo_files
import os
import sys

repo_id, selector, cache_dir = sys.argv[1], sys.argv[2], sys.argv[3]
selector_lc = selector.lower().strip()
selector_no_ext = selector_lc[:-5] if selector_lc.endswith(".gguf") else selector_lc

def normalize(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())

selector_norm = normalize(selector_no_ext)

repo_files = list_repo_files(repo_id=repo_id, repo_type="model")
gguf_files = [f for f in repo_files if f.lower().endswith(".gguf")]
if not gguf_files:
    raise RuntimeError(f"No .gguf files found in repo '{repo_id}'")

target = None

# 1) Exact path match (case-insensitive), with and without .gguf suffix.
exact_candidates = [selector_lc]
if not selector_lc.endswith(".gguf"):
    exact_candidates.insert(0, f"{selector_lc}.gguf")
for candidate in exact_candidates:
    for f in gguf_files:
        if f.lower() == candidate:
            target = f
            break
    if target:
        break

# 2) Exact basename match.
if target is None:
    wanted = f"{selector_no_ext}.gguf"
    base_matches = [f for f in gguf_files if os.path.basename(f).lower() == wanted]
    if len(base_matches) == 1:
        target = base_matches[0]

# 3) Token match anywhere in basename/path.
if target is None:
    token_matches = [
        f for f in gguf_files
        if (
            selector_no_ext in os.path.basename(f).lower()
            or selector_no_ext in f.lower()
            or selector_norm in normalize(os.path.basename(f))
            or selector_norm in normalize(f)
        )
    ]
    if len(token_matches) == 1:
        target = token_matches[0]
    elif len(token_matches) > 1:
        # Prefer shortest filename as a deterministic tie-breaker.
        token_matches = sorted(token_matches, key=lambda x: (len(os.path.basename(x)), len(x), x))
        target = token_matches[0]
        print(
            f"WARNING: selector '{selector}' matched multiple GGUF files; using '{target}'.",
            file=sys.stderr,
        )

if target is None:
    sample = ", ".join(sorted(os.path.basename(f) for f in gguf_files)[:12])
    raise RuntimeError(
        f"Could not resolve GGUF selector '{selector}' in repo '{repo_id}'. "
        f"Example available files: {sample}"
    )

print(f"Downloading GGUF file '{target}' from '{repo_id}' into '{cache_dir}'...", file=sys.stderr)
local_path = hf_hub_download(repo_id=repo_id, filename=target, cache_dir=cache_dir)
print(local_path)
PY
    )"; then
      echo "ERROR: Failed to resolve/download GGUF selector '${selector}' from '${repo_id}'." >&2
      return 1
    fi
    if [ -z "${local_path}" ]; then
      echo "ERROR: GGUF download returned empty path for '${repo_id}:${selector}'." >&2
      return 1
    fi
    echo "${local_path}"
    return 0
  fi
  echo "${spec}"
}

if ! MODEL_TO_SERVE="$(resolve_gguf_model_path "${MODEL_TO_SERVE}")"; then
  exit 1
fi

cmd=("${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server
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

if [ -n "${VLLM_RUNNER:-}" ] && supports_arg "--runner"; then
  cmd+=(--runner "${VLLM_RUNNER}")
fi

if [ -n "${VLLM_TASK:-}" ] && supports_arg "--task"; then
  cmd+=(--task "${VLLM_TASK}")
fi

if [ -n "${VLLM_DOWNLOAD_DIR:-}" ]; then
  cmd+=(--download-dir "${VLLM_DOWNLOAD_DIR}")
fi

if [ -n "${VLLM_TOKENIZER:-}" ]; then
  cmd+=(--tokenizer "${VLLM_TOKENIZER}")
fi

if [ "${VLLM_TRUST_REMOTE_CODE:-0}" = "1" ] || [ "${VLLM_TRUST_REMOTE_CODE:-}" = "true" ] || [ "${VLLM_TRUST_REMOTE_CODE:-}" = "TRUE" ]; then
  if supports_arg "--trust-remote-code"; then
    cmd+=(--trust-remote-code)
  fi
fi

if [ -n "${VLLM_EXTRA_ARGS:-}" ]; then
  parser_script="/parse_vllm_extra_args.py"
  if [ -f "${parser_script}" ]; then
    mapfile -d '' -t extra_args < <("${PYTHON_BIN}" "${parser_script}" "${VLLM_EXTRA_ARGS}")
    if [ "${#extra_args[@]}" -gt 0 ]; then
      cmd+=("${extra_args[@]}")
    fi
  else
    # Fallback for unexpected image/script skew.
    # shellcheck disable=SC2206,SC2294
    eval "extra_args=( ${VLLM_EXTRA_ARGS} )"
    cmd+=("${extra_args[@]}")
  fi
fi

# Optional LMCache / KV connector flags (see .env.example: LMCACHE_*).
if [ "${LMCACHE_ENABLED:-0}" = "1" ] || [ "${LMCACHE_ENABLED:-}" = "true" ] || [ "${LMCACHE_ENABLED:-}" = "TRUE" ]; then
  if [ -n "${LMCACHE_EXTRA_ARGS:-}" ]; then
    parser_script="/parse_vllm_extra_args.py"
    if [ -f "${parser_script}" ]; then
      mapfile -d '' -t lmc_args < <("${PYTHON_BIN}" "${parser_script}" "${LMCACHE_EXTRA_ARGS}")
      if [ "${#lmc_args[@]}" -gt 0 ]; then
        cmd+=("${lmc_args[@]}")
      fi
    else
      # shellcheck disable=SC2206,SC2294
      eval "lmc_args=( ${LMCACHE_EXTRA_ARGS} )"
      cmd+=("${lmc_args[@]}")
    fi
  fi
fi

# vLLM's offloading connector requires hybrid KV cache manager to be disabled.
# Auto-append the flag when KV offloading is enabled so startup doesn't crash.
if printf '%s\n' "${cmd[@]}" | grep -q -- '--kv-offloading-'; then
  if ! printf '%s\n' "${cmd[@]}" | grep -q -- '--disable-hybrid-kv-cache-manager'; then
    echo "Detected KV offloading args; adding --disable-hybrid-kv-cache-manager"
    cmd+=(--disable-hybrid-kv-cache-manager)
  fi
fi

# Avoid vLLM env validation warnings for wrapper-only variables.
unset \
  VLLM_HOST \
  VLLM_PORT \
  VLLM_MODEL \
  VLLM_SERVED_MODEL_NAME \
  VLLM_TENSOR_PARALLEL_SIZE \
  VLLM_GPU_MEMORY_UTILIZATION \
  VLLM_MAX_MODEL_LEN \
  VLLM_DTYPE \
  VLLM_RUNNER \
  VLLM_TASK \
  VLLM_DOWNLOAD_DIR \
  VLLM_TOKENIZER \
  VLLM_TRUST_REMOTE_CODE \
  VLLM_EXTRA_ARGS \
  VLLM_PYTHON_BIN \
  VLLM_BASE_URL \
  LMCACHE_ENABLED \
  LMCACHE_EXTRA_ARGS || true

echo "Starting vLLM with model='${MODEL_TO_SERVE}' served_as='${SERVED_MODEL_NAME}' on ${HOST}:${PORT}"
exec "${cmd[@]}"
