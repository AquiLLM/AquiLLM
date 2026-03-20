#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible wrapper: Mem0 now runs in OSS SDK mode inside AquiLLM,
# so there is no external Mem0 REST server to start.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "start_mem0_local.sh is deprecated; using OSS relaunch flow."
exec "${SCRIPT_DIR}/relaunch_mem0_oss.sh"
