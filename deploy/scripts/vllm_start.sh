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
      echo "${VLLM_MODEL:-hampsonw/Qwen3.6-27B-AWQ-BF16-INT4-mtp-bf16}|${VLLM_SERVED_MODEL_NAME:-qwen3.6:27b-mtp-awq}"
      ;;
    *)
      echo "${VLLM_MODEL:-hampsonw/Qwen3.6-27B-AWQ-BF16-INT4-mtp-bf16}|${VLLM_SERVED_MODEL_NAME:-qwen3.6:27b-mtp-awq}"
      ;;
  esac
}

IFS='|' read -r MODEL_TO_SERVE SERVED_MODEL_NAME <<< "$(select_model_and_alias)"

HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"

# If OCR_VLLM_EXTRA_ARGS is omitted, keep OCR startup VRAM-friendly with fp8 KV + 4-bit weights.
# Same baseline as .env.example.
_DEFAULT_OCR_VLLM_EXTRA_ARGS="--kv-cache-dtype fp8 --compilation-config '{\"cudagraph_mode\":\"PIECEWISE\"}' --quantization bitsandbytes --load-format bitsandbytes --model-loader-extra-config '{\"load_in_4bit\":true,\"bnb_4bit_compute_dtype\":\"float16\",\"bnb_4bit_quant_type\":\"nf4\",\"bnb_4bit_use_double_quant\":true}'"
_DEFAULT_TRANSCRIBE_VLLM_EXTRA_ARGS="--enforce-eager --max-num-seqs 1 --max-num-batched-tokens 50000 --generation-config /opt/aquillm/nemotron-generation-config"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
OCR_VLLM_EXTRA_ARGS="${OCR_VLLM_EXTRA_ARGS:-}"

# Compose sometimes injects VLLM_EXTRA_ARGS="" when ${VAR:-} interpolation is empty on the host,
# which overrides env_file. Recover from the service-specific *VLLM_EXTRA_ARGS in the same .env.
if [ -z "${VLLM_EXTRA_ARGS// }" ] && [ "${VLLM_SERVICE_KIND:-}" = "transcribe" ]; then
  _transcribe_extra_args="${TRANSCRIBE_VLLM_EXTRA_ARGS:-}"
  if [ -n "${_transcribe_extra_args// }" ]; then
    export VLLM_EXTRA_ARGS="${TRANSCRIBE_VLLM_EXTRA_ARGS}"
  else
    export VLLM_EXTRA_ARGS="${_DEFAULT_TRANSCRIBE_VLLM_EXTRA_ARGS}"
  fi
fi
if [ -z "${VLLM_EXTRA_ARGS// }" ]; then
  case "${VLLM_TASK:-}" in
    score) export VLLM_EXTRA_ARGS="${APP_RERANK_VLLM_EXTRA_ARGS:-}" ;;
  esac
fi
if [ -z "${VLLM_EXTRA_ARGS// }" ] && [ "${VLLM_RUNNER:-}" = "pooling" ] && [ -z "${VLLM_TASK:-}" ]; then
  case "${VLLM_MODEL:-}" in
    *Reranker*|*reranker*) export VLLM_EXTRA_ARGS="${APP_RERANK_VLLM_EXTRA_ARGS:-}" ;;
    *Embedding*|*embedding*) export VLLM_EXTRA_ARGS="${MEM0_EMBED_VLLM_EXTRA_ARGS:-}" ;;
  esac
fi
if [ -z "${VLLM_EXTRA_ARGS// }" ]; then
  case "${VLLM_MODEL:-}" in
    *Qwen2.5-VL*|*Qwen/Qwen2.5-VL*|*Qwen3.5-4B*|*Qwen/Qwen3.5-4B*)
      if [ -z "${VLLM_DTYPE:-}" ]; then
        export VLLM_DTYPE="float16"
      fi
      if [ -n "${OCR_VLLM_EXTRA_ARGS// }" ]; then
        export VLLM_EXTRA_ARGS="${OCR_VLLM_EXTRA_ARGS}"
      else
        export VLLM_EXTRA_ARGS="${_DEFAULT_OCR_VLLM_EXTRA_ARGS}"
      fi
      ;;
  esac
fi

# Strict sidecars must reject an alternate interpreter before invoking it for
# help, parsing, downloads, or model startup.
case "${VLLM_STRICT_PROTECTED_ARGS:-0}" in
  1|true|TRUE)
    if [ "${VLLM_PYTHON_BIN:-}" != "python3" ]; then
      echo "ERROR: invalid strict vLLM service contract: VLLM_PYTHON_BIN must be python3" >&2
      exit 64
    fi
    ;;
esac

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

_VLLM_HELP_STATE=0
_VLLM_HELP_TEXT=""

load_vllm_help() {
  if [ "${_VLLM_HELP_STATE}" = "1" ]; then
    return 0
  fi
  if [ "${_VLLM_HELP_STATE}" = "-1" ]; then
    return 1
  fi
  if ! _VLLM_HELP_TEXT="$(
    "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server --help 2>&1
  )"; then
    _VLLM_HELP_STATE=-1
    return 1
  fi
  _VLLM_HELP_STATE=1
}

supports_arg() {
  local arg_name="$1"
  load_vllm_help || return 1
  grep -Fq -- "${arg_name}" <<< "${_VLLM_HELP_TEXT}"
}

reject_protected_extra_args() {
  local arg
  for arg in "$@"; do
    case "${arg}" in
      --model|--model=*|\
      --served-model-name|--served-model-name=*|\
      --tokenizer|--tokenizer=*|\
      --revision|--revision=*|\
      --tokenizer-revision|--tokenizer-revision=*|\
      --code-revision|--code-revision=*|\
      --trust-remote-code|--trust-remote-code=*|\
      --no-trust-remote-code|--no-trust-remote-code=*|\
      --runner|--runner=*|\
      --dtype|--dtype=*|\
      --tensor-parallel-size|--tensor-parallel-size=*|\
      --gpu-memory-utilization|--gpu-memory-utilization=*|\
      --max-model-len|--max-model-len=*|\
      --api-key|--api-key=*|\
      --download-dir|--download-dir=*|\
      --task|--task=*)
        echo "ERROR: protected vLLM option is not allowed in extra args: ${arg}" >&2
        return 64
        ;;
    esac
  done
}

strict_protected_args_enabled() {
  case "${VLLM_STRICT_PROTECTED_ARGS:-0}" in
    1|true|TRUE) return 0 ;;
    *) return 1 ;;
  esac
}

parse_extra_args_into() {
  local raw="$1"
  local output_name="$2"
  local parser_script="/parse_vllm_extra_args.py"
  local parser_output
  local -n parsed_output="${output_name}"
  parsed_output=()

  if [ -f "${parser_script}" ] && [ -r "${parser_script}" ]; then
    parser_output="$(mktemp)"
    if ! "${PYTHON_BIN}" "${parser_script}" "${raw}" > "${parser_output}"; then
      rm -f -- "${parser_output}"
      echo "ERROR: vLLM extra-argument parser rejected its input." >&2
      return 65
    fi
    mapfile -d '' -t parsed_output < "${parser_output}"
    rm -f -- "${parser_output}"
    if strict_protected_args_enabled && [ "${#parsed_output[@]}" -eq 0 ]; then
      echo "ERROR: strict vLLM extra-argument parser returned no arguments." >&2
      return 65
    fi
    return 0
  fi

  if strict_protected_args_enabled; then
    echo "ERROR: strict vLLM service requires /parse_vllm_extra_args.py." >&2
    return 66
  fi
  # Compatibility fallback for legacy images. Strict sidecars never evaluate it.
  eval "${output_name}=( ${raw} )"
}

strict_contract_error() {
  echo "ERROR: invalid strict vLLM service contract: $1" >&2
  return 64
}

is_safe_hf_repo_id() {
  local value="$1"
  [ "${#value}" -le 256 ] \
    && [[ "${value}" =~ ^[A-Za-z0-9]([A-Za-z0-9._-]{0,126}[A-Za-z0-9])?/[A-Za-z0-9]([A-Za-z0-9._-]{0,126}[A-Za-z0-9])?$ ]]
}

validate_strict_service_contract() {
  local model="${VLLM_MODEL:-}"
  local served="${VLLM_SERVED_MODEL_NAME:-}"
  local tokenizer="${VLLM_TOKENIZER:-}"
  local revision="${VLLM_REVISION:-}"
  local tokenizer_revision="${VLLM_TOKENIZER_REVISION:-}"
  local code_revision="${VLLM_CODE_REVISION:-}"
  local gpu="${VLLM_GPU_MEMORY_UTILIZATION:-}"
  local required_arg

  if ! is_safe_hf_repo_id "${model}"; then
    strict_contract_error "model must be a bounded Hugging Face repository ID"
    return
  fi
  if [ "${served}" != "${model}" ] || [ "${tokenizer}" != "${model}" ]; then
    strict_contract_error "model, served model, and tokenizer must be identical"
    return
  fi
  if ! [[ "${revision}" =~ ^[0-9a-f]{40}$ ]] \
    || [ "${tokenizer_revision}" != "${revision}" ] \
    || [ "${code_revision}" != "${revision}" ]; then
    strict_contract_error "model, tokenizer, and code revisions must be one lowercase commit"
    return
  fi
  if [ "${VLLM_RUNNER:-}" != "pooling" ] || [ "${VLLM_DTYPE:-}" != "float16" ]; then
    strict_contract_error "runner and dtype must be the canonical pooling/float16 pair"
    return
  fi
  if [ "${VLLM_TRUST_REMOTE_CODE:-}" != "1" ]; then
    strict_contract_error "trust-remote-code must be explicitly enabled"
    return
  fi
  if [ "${VLLM_API_KEY:-}" != "EMPTY" ]; then
    strict_contract_error "API key must be the canonical EMPTY token"
    return
  fi
  if [ "${VLLM_DOWNLOAD_DIR:-}" != "/root/.cache/huggingface/hub" ]; then
    strict_contract_error "download directory must be the canonical model cache"
    return
  fi
  if [ "${LMCACHE_ENABLED:-}" != "0" ] || [ -n "${LMCACHE_EXTRA_ARGS:-}" ]; then
    strict_contract_error "LMCache must be explicitly disabled without extra arguments"
    return
  fi
  if [ -z "${VLLM_EXTRA_ARGS// }" ]; then
    strict_contract_error "extra arguments must be the canonical service payload"
    return
  fi
  if [ -n "${VLLM_TASK:-}" ]; then
    strict_contract_error "strict vLLM 0.21 services must not set the removed task option"
    return
  fi
  if ! [[ "${VLLM_TENSOR_PARALLEL_SIZE:-}" =~ ^[1-9][0-9]*$ ]] \
    || ! [[ "${VLLM_MAX_MODEL_LEN:-}" =~ ^[1-9][0-9]*$ ]]; then
    strict_contract_error "tensor parallel size and max model length must be positive integers"
    return
  fi
  if ! [[ "${gpu}" =~ ^(0\.[0-9]+|1(\.0+)?)$ ]] || [[ "${gpu}" =~ ^0\.0+$ ]]; then
    strict_contract_error "GPU memory utilization must be in (0, 1]"
    return
  fi

  local required_args=(
    --model
    --served-model-name
    --tokenizer
    --revision
    --tokenizer-revision
    --code-revision
    --runner
    --dtype
    --trust-remote-code
    --tensor-parallel-size
    --gpu-memory-utilization
    --max-model-len
    --api-key
    --download-dir
  )
  for required_arg in "${required_args[@]}"; do
    if ! supports_arg "${required_arg}"; then
      strict_contract_error "required option ${required_arg} is unsupported"
      return
    fi
  done
}

validate_strict_extra_args() {
  local -a expected=()
  local index
  if [[ "${VLLM_MODEL:-}" == *[Rr]eranker* ]]; then
    expected=(
      --chat-template
      /templates/qwen3_vl_reranker.jinja
      --hf-overrides
      '{"architectures":["Qwen3VLForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}'
    )
  elif [[ "${VLLM_MODEL:-}" == *Embedding* ]] \
    || [[ "${VLLM_MODEL:-}" == *embedding* ]]; then
    expected=(
      --quantization
      bitsandbytes
      --load-format
      bitsandbytes
      --model-loader-extra-config
      '{"load_in_4bit":true,"bnb_4bit_compute_dtype":"float16","bnb_4bit_quant_type":"nf4","bnb_4bit_use_double_quant":true}'
      --hf-overrides
      '{"matryoshka_dimensions":[1024]}'
    )
  else
    strict_contract_error "strict service must be the pinned embedding or reranker role"
    return
  fi
  if [ "${#extra_args[@]}" -ne "${#expected[@]}" ]; then
    strict_contract_error "extra arguments differ from the canonical service payload"
    return
  fi
  for index in "${!expected[@]}"; do
    if [ "${extra_args[$index]}" != "${expected[$index]}" ]; then
      strict_contract_error "extra arguments differ from the canonical service payload"
      return
    fi
  done
}

_strict_extra_args_preparsed=0
if strict_protected_args_enabled; then
  parse_extra_args_into "${VLLM_EXTRA_ARGS:-}" extra_args
  reject_protected_extra_args "${extra_args[@]}"
  validate_strict_extra_args
  _strict_extra_args_preparsed=1
  validate_strict_service_contract
fi

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
  --disable-log-requests
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

if [ -n "${VLLM_REVISION:-}" ] && supports_arg "--revision"; then
  cmd+=(--revision "${VLLM_REVISION}")
fi

if [ -n "${VLLM_TOKENIZER_REVISION:-}" ] && supports_arg "--tokenizer-revision"; then
  cmd+=(--tokenizer-revision "${VLLM_TOKENIZER_REVISION}")
fi

if [ -n "${VLLM_CODE_REVISION:-}" ] && supports_arg "--code-revision"; then
  cmd+=(--code-revision "${VLLM_CODE_REVISION}")
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

# bitsandbytes + Qwen3-VL sequence-classification reranker can fail loading
# classifier weights. Use fp16 for rerank until that path is proven stable.
# Match explicit rerank intents only. Embedding sidecars also use the pooling
# runner, so pooling alone must never strip their required quantization payload.
_vllm_task_trim="${VLLM_TASK:-}"
_vllm_task_trim="${_vllm_task_trim%%[$'\r']}"
_rerank_bnb_strip=0
if ! strict_protected_args_enabled \
  && [[ "${VLLM_EXTRA_ARGS:-}" == *[Bb]itsandbytes* ]]; then
  if [ "${_vllm_task_trim}" = "score" ] \
    || [[ "${VLLM_MODEL:-}" == *[Rr]eranker* ]] \
    || [[ "${VLLM_EXTRA_ARGS:-}" == *is_original_qwen3_reranker* ]]; then
    _rerank_bnb_strip=1
  fi
fi
if [ "${_rerank_bnb_strip}" = "1" ]; then
  echo "WARN: Removing bitsandbytes flags from rerank VLLM_EXTRA_ARGS; incompatible with Qwen3-VL reranker." >&2
  VLLM_EXTRA_ARGS="$(
    VLLM_EXTRA_ARGS_IN="${VLLM_EXTRA_ARGS}" "${PYTHON_BIN}" - <<'PY'
import os
import shlex

raw = os.environ.get("VLLM_EXTRA_ARGS_IN", "")
try:
    toks = shlex.split(raw, posix=True)
except ValueError:
    toks = []

out: list[str] = []
i = 0
while i < len(toks):
    if toks[i] in ("--quantization", "--load-format") and i + 1 < len(toks) and toks[i + 1] == "bitsandbytes":
        i += 2
        continue
    if toks[i] == "--model-loader-extra-config" and i + 1 < len(toks):
        i += 2
        continue
    out.append(toks[i])
    i += 1

print(shlex.join(out))
PY
  )"
fi

if [ -n "${VLLM_EXTRA_ARGS:-}" ]; then
  if [ "${_strict_extra_args_preparsed}" != "1" ]; then
    parse_extra_args_into "${VLLM_EXTRA_ARGS}" extra_args
  fi
  if [ "${#extra_args[@]}" -gt 0 ]; then
    cmd+=("${extra_args[@]}")
  fi
fi

# Optional LMCache / KV connector flags (see .env.example: LMCACHE_*).
if [ "${LMCACHE_ENABLED:-0}" = "1" ] || [ "${LMCACHE_ENABLED:-}" = "true" ] || [ "${LMCACHE_ENABLED:-}" = "TRUE" ]; then
  if [ -n "${LMCACHE_EXTRA_ARGS:-}" ]; then
    parse_extra_args_into "${LMCACHE_EXTRA_ARGS}" lmc_args
    if [ "${#lmc_args[@]}" -gt 0 ]; then
      if strict_protected_args_enabled; then
        reject_protected_extra_args "${lmc_args[@]}"
      fi
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
  _rerank_bnb_strip \
  _strict_extra_args_preparsed \
  _vllm_task_trim \
  _VLLM_HELP_STATE \
  _VLLM_HELP_TEXT \
  VLLM_HOST \
  VLLM_PORT \
  VLLM_MODEL \
  VLLM_SERVED_MODEL_NAME \
  VLLM_TENSOR_PARALLEL_SIZE \
  VLLM_GPU_MEMORY_UTILIZATION \
  VLLM_MAX_MODEL_LEN \
  VLLM_DTYPE \
  VLLM_REVISION \
  VLLM_TOKENIZER_REVISION \
  VLLM_CODE_REVISION \
  VLLM_SERVICE_KIND \
  VLLM_RUNNER \
  VLLM_TASK \
  VLLM_API_KEY \
  VLLM_DOWNLOAD_DIR \
  VLLM_TOKENIZER \
  VLLM_TRUST_REMOTE_CODE \
  VLLM_STRICT_PROTECTED_ARGS \
  VLLM_EXTRA_ARGS \
  VLLM_PYTHON_BIN \
  VLLM_BASE_URL \
  LMCACHE_ENABLED \
  LMCACHE_EXTRA_ARGS || true

echo "Starting vLLM with model='${MODEL_TO_SERVE}' served_as='${SERVED_MODEL_NAME}' on ${HOST}:${PORT}"
exec "${cmd[@]}"
