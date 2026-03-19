#!/usr/bin/env bash
set -euo pipefail

# Relaunch Mem0 in OSS mode for AquiLLM.
# In OSS mode there is no separate Mem0 REST server to start; Mem0 runs in-process
# and stores vectors in Qdrant. Relaunching means recreating the dependent services.
#
# Optional env vars:
#   AQUILLM_COMPOSE_FILE=deploy/compose/development.yml
#   RELAUNCH_MEM0_MODELS=0   # set to 1 to also recreate model services

AQUILLM_COMPOSE_FILE="${AQUILLM_COMPOSE_FILE:-deploy/compose/development.yml}"
RELAUNCH_MEM0_MODELS="${RELAUNCH_MEM0_MODELS:-0}"

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker is not installed." >&2
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "ERROR: docker compose plugin is not available." >&2
  exit 1
fi

if [ ! -f "$AQUILLM_COMPOSE_FILE" ]; then
  echo "ERROR: compose file '$AQUILLM_COMPOSE_FILE' was not found in $(pwd)." >&2
  exit 1
fi

mapfile -t compose_services < <(docker compose -f "$AQUILLM_COMPOSE_FILE" config --services)

service_exists() {
  local needle="$1"
  for svc in "${compose_services[@]}"; do
    if [ "$svc" = "$needle" ]; then
      return 0
    fi
  done
  return 1
}

services_to_relaunch=()
for required_service in qdrant web worker; do
  if service_exists "$required_service"; then
    services_to_relaunch+=("$required_service")
  else
    echo "WARNING: service '$required_service' is not defined in $AQUILLM_COMPOSE_FILE; skipping."
  fi
done

if [ "${RELAUNCH_MEM0_MODELS}" = "1" ]; then
  for model_service in vllm vllm_ocr vllm_transcribe vllm_embed vllm_rerank; do
    if service_exists "$model_service"; then
      services_to_relaunch+=("$model_service")
    fi
  done
fi

if [ "${#services_to_relaunch[@]}" -eq 0 ]; then
  echo "ERROR: no services found to relaunch." >&2
  exit 1
fi

echo "Relaunching Mem0 OSS dependencies via $AQUILLM_COMPOSE_FILE"
echo "Services: ${services_to_relaunch[*]}"
docker compose -f "$AQUILLM_COMPOSE_FILE" up -d --force-recreate "${services_to_relaunch[@]}"

if service_exists qdrant; then
  echo "Checking Qdrant readiness on http://localhost:6333/healthz"
  for i in $(seq 1 30); do
    if curl -fsS http://localhost:6333/healthz >/dev/null 2>&1 || curl -fsS http://localhost:6333/ >/dev/null 2>&1; then
      echo "Qdrant is reachable."
      break
    fi
    if [ "$i" -eq 30 ]; then
      echo "WARNING: Qdrant did not report ready within timeout." >&2
    fi
    sleep 2
  done
fi

echo "Done. Mem0 OSS is now relaunched through AquiLLM services."
