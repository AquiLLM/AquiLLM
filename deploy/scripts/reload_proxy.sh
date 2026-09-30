#!/bin/sh
set -eu
# Templates are rendered only at nginx container startup unless explicitly rerun.
# Re-render before reload so DNS and routing changes reach a running proxy.
docker compose --env-file "${AQUILLM_ENV_FILE:-.env}" \
  -f "${AQUILLM_COMPOSE_FILE:-deploy/compose/development.yml}" \
  exec -T nginx /docker-entrypoint.d/20-envsubst-on-templates.sh </dev/null
docker compose --env-file "${AQUILLM_ENV_FILE:-.env}" \
  -f "${AQUILLM_COMPOSE_FILE:-deploy/compose/development.yml}" \
  exec -T nginx nginx -t </dev/null
docker compose --env-file "${AQUILLM_ENV_FILE:-.env}" \
  -f "${AQUILLM_COMPOSE_FILE:-deploy/compose/development.yml}" \
  exec -T nginx nginx -s reload </dev/null
