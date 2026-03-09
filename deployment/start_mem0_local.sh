#!/usr/bin/env bash
set -euo pipefail

# Starts a local OSS Mem0 server on the host (default port 8888),
# then verifies AquiLLM's web container can reach it.
#
# Optional env vars:
#   MEM0_DIR=/opt/mem0
#   MEM0_REPO_URL=https://github.com/mem0ai/mem0.git
#   MEM0_PORT=8888
#   AQUILLM_COMPOSE_FILE=docker-compose-development.yml

MEM0_DIR="${MEM0_DIR:-/opt/mem0}"
MEM0_REPO_URL="${MEM0_REPO_URL:-https://github.com/mem0ai/mem0.git}"
MEM0_PORT="${MEM0_PORT:-8888}"
AQUILLM_COMPOSE_FILE="${AQUILLM_COMPOSE_FILE:-docker-compose-development.yml}"

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker is not installed." >&2
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "ERROR: docker compose plugin is not available." >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git is not installed." >&2
  exit 1
fi

if [ ! -d "$MEM0_DIR/.git" ]; then
  echo "Cloning Mem0 OSS repo into $MEM0_DIR"
  mkdir -p "$(dirname "$MEM0_DIR")"
  git clone "$MEM0_REPO_URL" "$MEM0_DIR"
else
  echo "Updating Mem0 OSS repo in $MEM0_DIR"
  git -C "$MEM0_DIR" pull --ff-only
fi

MEM0_COMPOSE_CANDIDATES=(
  "$MEM0_DIR/docker-compose.yml"
  "$MEM0_DIR/docker-compose.yaml"
  "$MEM0_DIR/compose.yml"
  "$MEM0_DIR/server/docker-compose.yml"
  "$MEM0_DIR/server/docker-compose.yaml"
  "$MEM0_DIR/openmemory/docker-compose.yml"
  "$MEM0_DIR/openmemory/docker-compose.yaml"
)

MEM0_COMPOSE_FILE=""
for candidate in "${MEM0_COMPOSE_CANDIDATES[@]}"; do
  if [ -f "$candidate" ]; then
    MEM0_COMPOSE_FILE="$candidate"
    break
  fi
done

if [ -z "$MEM0_COMPOSE_FILE" ]; then
  echo "ERROR: Could not find a Mem0 compose file under $MEM0_DIR." >&2
  echo "Checked:" >&2
  printf '  - %s\n' "${MEM0_COMPOSE_CANDIDATES[@]}" >&2
  exit 1
fi

echo "Using Mem0 compose file: $MEM0_COMPOSE_FILE"
docker compose -f "$MEM0_COMPOSE_FILE" up -d --build

echo "Waiting for Mem0 API at http://localhost:${MEM0_PORT}/docs"
for i in $(seq 1 60); do
  if curl -fsS "http://localhost:${MEM0_PORT}/docs" >/dev/null 2>&1; then
    echo "Mem0 is reachable on localhost:${MEM0_PORT}"
    break
  fi
  if [ "$i" -eq 60 ]; then
    echo "ERROR: Mem0 did not become ready on localhost:${MEM0_PORT} within timeout." >&2
    echo "Run: docker compose -f \"$MEM0_COMPOSE_FILE\" logs --tail=200" >&2
    exit 1
  fi
  sleep 2
done

if [ ! -f "$AQUILLM_COMPOSE_FILE" ]; then
  echo "Skipping AquiLLM web connectivity check; $AQUILLM_COMPOSE_FILE not found in $(pwd)."
  exit 0
fi

echo "Recreating AquiLLM web container to pick up env/config changes"
docker compose -f "$AQUILLM_COMPOSE_FILE" up -d --force-recreate web

echo "Checking connectivity from AquiLLM web -> host Mem0"
docker compose -f "$AQUILLM_COMPOSE_FILE" exec web sh -lc \
  "curl -fsS http://host.docker.internal:${MEM0_PORT}/docs >/dev/null && echo mem0_reachable_from_web"

echo "Done."
