#!/usr/bin/env bash
# Provider-neutral Task21 hybrid retrieval cloud acceptance runner.

set -euo pipefail

SCHEMA="task21-hybrid-cloud-evidence-v1"
REQUIRE_CLEAN_HEAD=0
REQUESTED_SCHEMA=""

while (($#)); do
  case "$1" in
    --require-clean-head)
      REQUIRE_CLEAN_HEAD=1
      shift
      ;;
    --evidence-schema)
      (($# >= 2)) || { echo "--evidence-schema requires a value" >&2; exit 64; }
      REQUESTED_SCHEMA="$2"
      shift 2
      ;;
    *)
      echo "unsupported argument: $1" >&2
      exit 64
      ;;
  esac
done

((REQUIRE_CLEAN_HEAD == 1)) || { echo "--require-clean-head is required" >&2; exit 64; }
[[ "$REQUESTED_SCHEMA" == "$SCHEMA" ]] || {
  echo "--evidence-schema must be $SCHEMA" >&2
  exit 64
}

required_environment=(
  TASK21_ENV_FILE
  TASK21_EVIDENCE_SIGNING_KEY
  TASK21_EVIDENCE_SIGNING_KEY_VERSION
)
for name in "${required_environment[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "required environment is missing: $name" >&2; exit 64; }
done
for name in TASK21_ENV_FILE; do
  value="${!name}"
  [[ "$value" == /* && -f "$value" ]] || {
    echo "$name must name an existing absolute file" >&2
    exit 64
  }
done

REPOSITORY="$(git rev-parse --show-toplevel)"
cd "$REPOSITORY"
if [[ -n "$(git status --porcelain=v1 --untracked-files=normal)" ]]; then
  echo "--require-clean-head rejected a dirty source tree" >&2
  exit 65
fi
SOURCE_COMMIT="$(git rev-parse HEAD)"
LIVE_GENERATOR="aquillm/apps/knowledge_graph/evals/task21_hybrid_live_observations.py"
[[ -f "$LIVE_GENERATOR" ]] || {
  echo "integrated production five-arm observation generator is unavailable" >&2
  exit 69
}

RUN_ID="$(python3 -c 'import secrets; print(secrets.token_hex(16))')"
PROJECT_NAME="aquillm-task21-$RUN_ID"
WORK_REL="artifacts/.task21-hybrid-$RUN_ID"
WORK_ROOT="$REPOSITORY/$WORK_REL"
OUTPUT_ROOT="$REPOSITORY/artifacts/task21-hybrid-cloud"
DEVELOPMENT="$REPOSITORY/deploy/compose/development.yml"
EVALUATION="$REPOSITORY/deploy/compose/knowledge-graph-eval.yml"
ARMS="$WORK_ROOT/arms.json"
ARMS_CANDIDATE="$WORK_ROOT/arms.valid.json"
TIMINGS="$WORK_ROOT/timings.json"
PROJECTIONS="$WORK_ROOT/projection-checksums.json"
ATTESTATION="$WORK_ROOT/observation-attestation.json"
LIVE_TRACE="$WORK_ROOT/live-trace.json"
RUN_LOG="$WORK_ROOT/runner.log"
STARTED_NS="$(python3 -c 'import time; print(time.time_ns())')"

install -d -m 700 "$WORK_ROOT" "$OUTPUT_ROOT"
printf '%s\n' '{"schema_version":"task21-hybrid-eval-v1","status":"not_completed"}' >"$ARMS"
printf '%s\n' '{"status":"not_completed"}' >"$TIMINGS"
printf '%s\n' '{"schema":"task21-hybrid-live-observation-v1","status":"not_completed"}' >"$ATTESTATION"
printf '%s\n' '{"schema":"task21-hybrid-live-trace-v1","status":"not_completed"}' >"$LIVE_TRACE"
printf '%s\n' \
  '{"generation_key":"0000000000000000000000000000000000000000000000000000000000000000","projection_checksum":"0000000000000000000000000000000000000000000000000000000000000000"}' \
  >"$PROJECTIONS"
chmod 600 "$ARMS" "$TIMINGS" "$PROJECTIONS" "$ATTESTATION" "$LIVE_TRACE"

compose=(
  docker compose
  --env-file "$TASK21_ENV_FILE"
  --project-name "$PROJECT_NAME"
  --file "$DEVELOPMENT"
  --file "$EVALUATION"
  --profile knowledge-graph
  --profile vllm
)
services=(
  web db redis memgraph_knowledge_graph knowledge_graph_query_extractor
  worker_knowledge_graph_projection worker_knowledge_graph vllm_embed vllm_rerank
)

write_timings() {
  local status="$1"
  local original_exit_code="$2"
  local finished_ns
  finished_ns="$(python3 -c 'import time; print(time.time_ns())')"
  python3 - \
    "$TIMINGS" "$STARTED_NS" "$finished_ns" "$status" "$original_exit_code" <<'PY'
import json
import os
import sys

path, started, finished, status, original_exit_code = sys.argv[1:]
payload = {
    "elapsed_ms": (int(finished) - int(started)) / 1_000_000,
    "finished_ns": int(finished),
    "original_exit_code": int(original_exit_code),
    "started_ns": int(started),
    "status": status,
}
temporary = path + ".tmp"
with open(temporary, "x", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.chmod(temporary, 0o600)
os.replace(temporary, path)
PY
}

finalize() {
  local original_status="$?"
  local status_label="passed"
  local evidence_status=0
  local timings_status=0
  trap - EXIT INT TERM
  set +e
  ((original_status == 0)) || status_label="failed"
  write_timings "$status_label" "$original_status" || timings_status="$?"
  python3 scripts/task21_hybrid_failure_bundle.py \
    --run-id "$RUN_ID" \
    --output-root "$OUTPUT_ROOT" \
    --project-name "$PROJECT_NAME" \
    --env-file "$TASK21_ENV_FILE" \
    --compose-file "$DEVELOPMENT" \
    --compose-file "$EVALUATION" \
    --profile knowledge-graph \
    --profile vllm \
    --arm-results "$ARMS" \
    --timings "$TIMINGS" \
    --projection-checksums "$PROJECTIONS" \
    --observation-attestation "$ATTESTATION" \
    --live-trace "$LIVE_TRACE" \
    --expected-source-commit "$SOURCE_COMMIT" \
    --claim-scope cloud \
    --signing-key-version "$TASK21_EVIDENCE_SIGNING_KEY_VERSION"
  evidence_status="$?"
  if ((timings_status != 0 && evidence_status == 0)); then
    evidence_status=70
  fi
  rm -f -- \
    "$WORK_ROOT/observations.json" "$WORK_ROOT/freshness.json" \
    "$WORK_ROOT/backend-parity.json" "$ATTESTATION" "$LIVE_TRACE" \
    "$ARMS" "$ARMS_CANDIDATE" "$TIMINGS" "$TIMINGS.tmp" \
    "$PROJECTIONS" "$RUN_LOG"
  rmdir -- "$WORK_ROOT" 2>/dev/null
  ((original_status == 0)) || exit "$original_status"
  exit "$evidence_status"
}
trap finalize EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"${compose[@]}" config --quiet >>"$RUN_LOG" 2>&1
"${compose[@]}" up --detach --wait --wait-timeout 300 db redis >>"$RUN_LOG" 2>&1
POSTGRES_NAME="$("${compose[@]}" exec -T db /bin/sh -c 'printf %s "$POSTGRES_DB"')"
[[ "$POSTGRES_NAME" =~ ^[A-Za-z0-9_]{1,63}$ ]] || {
  echo "runtime PostgreSQL database name is invalid" >&2
  exit 65
}
SOURCE_PASSWORD="$(python3 -c 'import secrets; print(secrets.token_hex(24))')"
STATE_PASSWORD="$(python3 -c 'import secrets; print(secrets.token_hex(24))')"
printf "CREATE ROLE aquillm_projection_source LOGIN PASSWORD '%s';\n" \
  "$SOURCE_PASSWORD" | "${compose[@]}" exec -T db /bin/sh -c \
  'exec psql --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" --set ON_ERROR_STOP=1' \
  >>"$RUN_LOG" 2>&1
printf "CREATE ROLE aquillm_projection_state LOGIN PASSWORD '%s';\n" \
  "$STATE_PASSWORD" | "${compose[@]}" exec -T db /bin/sh -c \
  'exec psql --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" --set ON_ERROR_STOP=1' \
  >>"$RUN_LOG" 2>&1
export KG_PROJECTION_POSTGRES_SOURCE_DSN="postgresql://aquillm_projection_source:$SOURCE_PASSWORD@db:5432/$POSTGRES_NAME"
export KG_PROJECTION_POSTGRES_STATE_DSN="postgresql://aquillm_projection_state:$STATE_PASSWORD@db:5432/$POSTGRES_NAME"
"${compose[@]}" config --quiet >>"$RUN_LOG" 2>&1
"${compose[@]}" up --detach --wait --wait-timeout 3600 "${services[@]}" >>"$RUN_LOG" 2>&1
"${compose[@]}" exec -T web /bin/sh -c \
  'cd /app/aquillm && exec /opt/venv/bin/python manage.py migrate --noinput' \
  >>"$RUN_LOG" 2>&1
"${compose[@]}" exec -T worker_knowledge_graph_projection /bin/sh -c \
  'cd /app/aquillm && exec /opt/venv/bin/python manage.py project_knowledge_graph --all' \
  >>"$RUN_LOG" 2>&1
"${compose[@]}" exec -T worker_knowledge_graph_projection /bin/sh -c \
  'cd /app/aquillm && exec /opt/venv/bin/python manage.py reconcile_knowledge_graph_projection --all' \
  >>"$RUN_LOG" 2>&1
"${compose[@]}" exec -T worker_knowledge_graph_projection /bin/sh -c \
  'cd /app/aquillm && exec /opt/venv/bin/python manage.py inspect_knowledge_graph_projection --all' \
  >>"$RUN_LOG" 2>&1

"${compose[@]}" exec -T worker_knowledge_graph /opt/venv/bin/python -m \
  apps.knowledge_graph.evals.task21_hybrid_live_observations \
  --run-id "$RUN_ID" \
  --source-commit "$SOURCE_COMMIT" \
  --observations-output "/app/$WORK_REL/observations.json" \
  --freshness-output "/app/$WORK_REL/freshness.json" \
  --backend-parity-output "/app/$WORK_REL/backend-parity.json" \
  --live-trace-output "/app/$WORK_REL/live-trace.json" \
  --attestation-output "/app/$WORK_REL/observation-attestation.json" \
  >>"$RUN_LOG" 2>&1

python3 scripts/task21_hybrid_observation_attestation.py \
  --run-id "$RUN_ID" \
  --source-commit "$SOURCE_COMMIT" \
  --env-file "$TASK21_ENV_FILE" \
  --compose-file "$DEVELOPMENT" \
  --compose-file "$EVALUATION" \
  --profile knowledge-graph \
  --profile vllm \
  --attestation "$ATTESTATION" \
  --observations "$WORK_ROOT/observations.json" \
  --freshness "$WORK_ROOT/freshness.json" \
  --backend-parity "$WORK_ROOT/backend-parity.json" \
  --live-trace "$LIVE_TRACE" \
  >>"$RUN_LOG" 2>&1

"${compose[@]}" exec -T worker_knowledge_graph /opt/venv/bin/python -m \
  apps.knowledge_graph.evals.task21_hybrid_eval_cli \
  --cases /app/aquillm/apps/knowledge_graph/evals/retrieval_cases.yaml \
  --observations "/app/$WORK_REL/observations.json" \
  --freshness "/app/$WORK_REL/freshness.json" \
  --backend-parity "/app/$WORK_REL/backend-parity.json" \
  --output "/app/$WORK_REL/arms.valid.json" \
  >>"$RUN_LOG" 2>&1
mv -- "$ARMS_CANDIDATE" "$ARMS"

python3 - "$WORK_ROOT/freshness.json" "$PROJECTIONS" <<'PY'
import json
import os
import re
import sys

source, destination = sys.argv[1:]
with open(source, encoding="utf-8") as handle:
    freshness = json.load(handle)
result = {
    key: freshness[key]
    for key in ("generation_key", "projection_checksum")
}
if any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in result.values()):
    raise SystemExit("projection identity is invalid")
temporary = destination + ".tmp"
with open(temporary, "x", encoding="utf-8") as handle:
    json.dump(result, handle, sort_keys=True, separators=(",", ":"))
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.chmod(temporary, 0o600)
os.replace(temporary, destination)
PY

[[ "$(git rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || {
  echo "source commit changed during evaluation" >&2
  exit 65
}
if [[ -n "$(git status --porcelain=v1 --untracked-files=normal)" ]]; then
  echo "source tree changed during evaluation" >&2
  exit 65
fi
