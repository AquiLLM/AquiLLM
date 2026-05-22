#!/bin/sh
set -e

if python3 -c "import vllm._genesis" >/dev/null 2>&1; then
  python3 -m vllm._genesis.patches.apply_all \
    || echo "WARN: Genesis patch application returned non-zero; continuing to vLLM startup." >&2
else
  echo "WARN: Genesis module is not importable; starting plain vLLM." >&2
fi

exec /vllm_start.sh "$@"
