#!/bin/sh
set -e

if python3 -c "import vllm._genesis" >/dev/null 2>&1; then
  echo "Genesis module importable; vLLM plugin hook will apply patches at startup." >&2
else
  echo "WARN: Genesis module is not importable; starting plain vLLM." >&2
fi

exec /vllm_start.sh "$@"
