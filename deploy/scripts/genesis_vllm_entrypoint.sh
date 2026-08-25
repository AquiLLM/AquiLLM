#!/bin/sh
set -e

if python3 -c 'import sndr; from importlib.metadata import entry_points; assert any(ep.value == "sndr.plugin:register" for ep in entry_points(group="vllm.general_plugins"))' >/dev/null 2>&1; then
  echo "Genesis sndr plugin importable; vLLM plugin hook will apply patches at startup." >&2
else
  echo "ERROR: Genesis sndr plugin is not importable; refusing to start unpatched vLLM." >&2
  exit 1
fi

exec /vllm_start.sh "$@"
