#!/usr/bin/env bash
set -euo pipefail

# Standard development startup for AquiLLM.
# Starts vLLM services in series
# (chat -> ocr -> transcribe -> embed -> rerank), waiting for each
# to become healthy before moving on, then starts web/worker.
#
# CLI arguments:
#   --no-gpu          Use no_gpu_dev.yml compose file, disable vLLM
#   --observability   Layer observability stack (Grafana, Prometheus, etc.)
#
# Optional env vars:
#   AQUILLM_COMPOSE_FILE=deploy/compose/development.yml
#   USE_VLLM=1
#   USE_OBSERVABILITY=0
#   USE_EDGE=0
#   RUN_CERTBOT=0
#   BUILD=0
#   FORCE_RECREATE=1
#   WAIT_TIMEOUT_SECONDS=1800

# Parse CLI arguments
for arg in "$@"; do
  case "$arg" in
    --no-gpu)
      AQUILLM_COMPOSE_FILE="deploy/compose/no_gpu_dev.yml"
      USE_VLLM=0
      ;;
    --observability)
      USE_OBSERVABILITY=1
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: $0 [--no-gpu] [--observability]" >&2
      exit 1
      ;;
  esac
done

AQUILLM_COMPOSE_FILE="${AQUILLM_COMPOSE_FILE:-deploy/compose/development.yml}"
USE_VLLM="${USE_VLLM:-1}"
USE_OBSERVABILITY="${USE_OBSERVABILITY:-0}"
USE_EDGE="${USE_EDGE:-0}"
RUN_CERTBOT="${RUN_CERTBOT:-0}"
BUILD="${BUILD:-0}"
FORCE_RECREATE="${FORCE_RECREATE:-1}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-1800}"

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

compose_cmd=(
  docker compose
  --env-file .env
  -f "$AQUILLM_COMPOSE_FILE"
)

if [ "$USE_OBSERVABILITY" = "1" ]; then
  compose_cmd+=(-f deploy/compose/observability.yml)
fi

if [ "$USE_VLLM" = "1" ]; then
  compose_cmd+=(--profile vllm)
fi

compose_up() {
  local services=("$@")
  local cmd=("${compose_cmd[@]}" up -d)
  if [ "$BUILD" = "1" ]; then
    cmd+=(--build)
  fi
  if [ "$FORCE_RECREATE" = "1" ]; then
    cmd+=(--force-recreate)
  fi
  cmd+=("${services[@]}")
  "${cmd[@]}"
}

wait_for_service_healthy() {
  local service="$1"
  local deadline=$((SECONDS + WAIT_TIMEOUT_SECONDS))

  echo "Waiting for '$service' to become healthy..."
  while [ "$SECONDS" -lt "$deadline" ]; do
    local cid
    cid="$("${compose_cmd[@]}" ps -q "$service" 2>/dev/null || true)"
    if [ -z "$cid" ]; then
      sleep 3
      continue
    fi

    local status
    status="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$cid" 2>/dev/null || true)"
    case "$status" in
      healthy|running)
        echo "Service '$service' is $status."
        return 0
        ;;
      exited|dead)
        echo "ERROR: service '$service' entered state '$status'." >&2
        "${compose_cmd[@]}" logs --tail=200 "$service" >&2 || true
        return 1
        ;;
    esac
    sleep 5
  done

  echo "ERROR: timed out waiting for service '$service' health." >&2
  "${compose_cmd[@]}" logs --tail=200 "$service" >&2 || true
  return 1
}

if [ "$USE_VLLM" = "1" ]; then
  compose_up vllm
  wait_for_service_healthy vllm

  compose_up vllm_ocr
  wait_for_service_healthy vllm_ocr

  compose_up vllm_transcribe
  wait_for_service_healthy vllm_transcribe

  compose_up vllm_embed
  wait_for_service_healthy vllm_embed

  compose_up vllm_rerank
  wait_for_service_healthy vllm_rerank
fi

compose_up web worker

if [ "$USE_EDGE" = "1" ]; then
  if [ "$RUN_CERTBOT" = "1" ]; then
    "${compose_cmd[@]}" stop nginx >/dev/null 2>&1 || true
    compose_up get_certs
  fi
  compose_up nginx
  wait_for_service_healthy nginx
fi

echo "Development stack is up."
