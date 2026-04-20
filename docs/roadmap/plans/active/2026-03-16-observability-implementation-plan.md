# Observability Stack Refresh (Logging, Metrics, Profiling) Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refresh and execute observability work against the current AquiLLM architecture (`apps/*`, `lib/*`, `deploy/compose/*`) so runtime failures are diagnosable without ad hoc debugging.

**Architecture:** Add a profile-gated self-hosted observability stack (Prometheus, Grafana, Pyroscope), instrument current web/worker runtime paths, and normalize existing latency/cache telemetry logs into a consistent non-sensitive schema. Keep this work aligned with `docs/roadmap/roadmap-status.md` Phase 6 and avoid planning against legacy paths.

**Tech Stack:** Django 5.x, Channels, Celery, Python logging, `prometheus_client`, `pyroscope`, Docker Compose, Grafana, Prometheus.

---

## Roadmap Alignment and Scope Gates

- Source roadmap phase: `docs/roadmap/roadmap-status.md` -> **Phase 6: Observability and Runtime Operations**.
- This plan assumes the current codebase state as of 2026-03-25:
  - Existing telemetry-style logs already exist in RAG/chat paths (`chunk_search`, `rag_cache`, `context_packer`, `prompt_budget`, chat consumers).
  - No Prometheus/Grafana/Pyroscope manifests or compose wiring are present yet.
  - Logging config is centralized in `aquillm/aquillm/settings_logging.py` (not in `settings.py`).
- Scope gate for future roadmap items:
  - MCP/skills/agents specific metrics are **deferred** until Phase 4/4B/5 modules exist in this branch.
  - This plan adds the shared instrumentation foundation now and a follow-on hook task once those modules land.

---

## Current Baseline (Repository Evidence)

- Compose files in active use:
  - `deploy/compose/development.yml`
  - `deploy/compose/production.yml`
  - `deploy/compose/base.yml` (reference/compatibility, also tested)
- Launch scripts:
  - `deploy/scripts/start_dev.sh`
  - `deploy/scripts/run.sh`
- Logging:
  - `aquillm/aquillm/settings_logging.py` currently defines verbose/simple formatters with fixed levels.
- Existing observability-like logs (already in code):
  - `aquillm/apps/documents/services/chunk_search.py`
  - `aquillm/apps/documents/services/rag_cache.py`
  - `aquillm/lib/llm/utils/context_packer.py`
  - `aquillm/lib/llm/utils/prompt_budget.py`
  - `aquillm/apps/chat/consumers/chat_receive.py`
  - `aquillm/apps/chat/consumers/chat_delta.py`
- Missing today:
  - `/metrics` endpoint
  - Prometheus scrape config
  - Grafana provisioning
  - Pyroscope runtime wiring
  - env toggles for log level/JSON logging in `.env.example`

---

## File Structure and Responsibilities

### New files

- `deploy/observability/prometheus/prometheus.yml`
- `deploy/observability/grafana/provisioning/datasources/datasources.yml`
- `deploy/observability/grafana/provisioning/dashboards/dashboards.yml`
- `deploy/observability/grafana/provisioning/dashboards/json/aquillm-runtime-overview.json`
- `aquillm/aquillm/metrics.py`
- `aquillm/aquillm/logging_json.py`
- `aquillm/tests/integration/test_compose_observability_services.py`
- `aquillm/tests/integration/test_observability_settings.py`

### Modified files

- `deploy/compose/development.yml`
- `deploy/compose/production.yml`
- `deploy/compose/base.yml`
- `deploy/scripts/start_dev.sh`
- `.env.example`
- `README.md`
- `aquillm/aquillm/urls.py`
- `aquillm/aquillm/settings.py`
- `aquillm/aquillm/settings_logging.py`
- `aquillm/aquillm/asgi.py`
- `aquillm/aquillm/celery.py`
- `aquillm/apps/documents/services/chunk_search.py`
- `aquillm/apps/documents/services/rag_cache.py`
- `aquillm/lib/llm/utils/context_packer.py`
- `aquillm/lib/llm/utils/prompt_budget.py`
- `aquillm/apps/chat/consumers/chat_receive.py`
- `aquillm/apps/chat/consumers/chat_delta.py`
- `aquillm/apps/ingestion/services/upload_batches.py`
- `aquillm/apps/ingestion/services/arxiv_ingest.py`
- `aquillm/apps/ingestion/services/web_ingest.py`
- `aquillm/apps/documents/tasks/chunking.py`

---

## Chunk 1: Compose and Provisioning Foundation

### Task 1: Add repository-local observability config tree

**Files:**
- Create: `deploy/observability/prometheus/prometheus.yml`
- Create: `deploy/observability/grafana/provisioning/datasources/datasources.yml`
- Create: `deploy/observability/grafana/provisioning/dashboards/dashboards.yml`
- Create: `deploy/observability/grafana/provisioning/dashboards/json/aquillm-runtime-overview.json`

- [ ] **Step 1: Write Prometheus scrape config**
- [ ] **Step 2: Provision Grafana datasources for Prometheus and Pyroscope**
- [ ] **Step 3: Add one starter dashboard for request rate/latency and process health**
- [ ] **Step 4: Commit**

```bash
git add deploy/observability
git commit -m "chore(observability): add prometheus and grafana provisioning config"
```

### Task 2: Wire observability services into current compose files

**Files:**
- Modify: `deploy/compose/development.yml`
- Modify: `deploy/compose/production.yml`
- Modify: `deploy/compose/base.yml`

- [ ] **Step 1: Add `prometheus`, `grafana`, and `pyroscope` services to each compose file**
- [ ] **Step 2: Put services behind an `observability` profile so default startup remains unchanged**
- [ ] **Step 3: Add required mounts and named volumes (`prometheus_data`, `grafana_data`)**
- [ ] **Step 4: Add `PYROSCOPE_SERVER_ADDRESS` and metrics-related env vars to `web` and `worker`**
- [ ] **Step 5: Commit**

```bash
git add deploy/compose/development.yml deploy/compose/production.yml deploy/compose/base.yml
git commit -m "chore(compose): add optional observability profile services"
```

### Task 3: Update dev launcher for profile-aware observability startup

**Files:**
- Modify: `deploy/scripts/start_dev.sh`
- Test: `aquillm/tests/integration/test_dev_launch_script.py`

- [ ] **Step 1: Add `USE_OBSERVABILITY` env toggle (default `0`)**
- [ ] **Step 2: Include `--profile observability` when enabled**
- [ ] **Step 3: Optionally bring up observability services before app services**
- [ ] **Step 4: Update/extend integration test expectations**
- [ ] **Step 5: Commit**

```bash
git add deploy/scripts/start_dev.sh aquillm/tests/integration/test_dev_launch_script.py
git commit -m "chore(dev): add optional observability profile toggle to start_dev"
```

---

## Chunk 2: Runtime Instrumentation (Web + Worker)

### Task 4: Add metrics/profiling Python dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add `prometheus_client` and `pyroscope`**
- [ ] **Step 2: Commit**

```bash
git add requirements.txt
git commit -m "chore(deps): add prometheus_client and pyroscope"
```

### Task 5: Expose `/metrics` and request-level counters/histograms

**Files:**
- Create: `aquillm/aquillm/metrics.py`
- Modify: `aquillm/aquillm/urls.py`
- Modify: `aquillm/aquillm/settings.py`
- Test: `aquillm/tests/integration/test_observability_settings.py`

- [ ] **Step 1: Implement `metrics_view` and core HTTP counters/histograms**
- [ ] **Step 2: Add `path("metrics/", ...)` in root URL config**
- [ ] **Step 3: Guard high-cardinality labels (route name/pattern only, no raw query text)**
- [ ] **Step 4: Add integration test that `/metrics/` responds with Prometheus text format**
- [ ] **Step 5: Commit**

```bash
git add aquillm/aquillm/metrics.py aquillm/aquillm/urls.py aquillm/aquillm/settings.py aquillm/tests/integration/test_observability_settings.py
git commit -m "feat(observability): add prometheus metrics endpoint and request instrumentation"
```

### Task 6: Add Pyroscope bootstrap for ASGI and Celery

**Files:**
- Modify: `aquillm/aquillm/asgi.py`
- Modify: `aquillm/aquillm/celery.py`
- Modify: `aquillm/aquillm/settings.py`

- [ ] **Step 1: Add fail-open `pyroscope.configure(...)` in ASGI startup**
- [ ] **Step 2: Add fail-open `pyroscope.configure(...)` in Celery app startup**
- [ ] **Step 3: Gate with explicit env flag (`OBS_PROFILE_ENABLED`) to avoid accidental enablement**
- [ ] **Step 4: Commit**

```bash
git add aquillm/aquillm/asgi.py aquillm/aquillm/celery.py aquillm/aquillm/settings.py
git commit -m "feat(observability): add optional pyroscope bootstrap for web and worker"
```

---

## Chunk 3: Structured Logging and Configuration Hygiene

### Task 7: Modernize logging config for env-driven JSON/plain output

**Files:**
- Create: `aquillm/aquillm/logging_json.py`
- Modify: `aquillm/aquillm/settings_logging.py`
- Modify: `.env.example`

- [ ] **Step 1: Add JSON formatter helper (`logging.Formatter` subclass)**
- [ ] **Step 2: Add `LOG_LEVEL` + `LOG_JSON` env handling in `settings_logging.py`**
- [ ] **Step 3: Preserve existing logger names (`aquillm`, `chat`, `celery`, `ingest`, `lib.llm.utils`)**
- [ ] **Step 4: Keep fail-safe default behavior equivalent to current output when env vars are unset**
- [ ] **Step 5: Commit**

```bash
git add aquillm/aquillm/logging_json.py aquillm/aquillm/settings_logging.py .env.example
git commit -m "feat(logging): add env-driven json/plain logging and level controls"
```

### Task 8: Add logging settings and compose coverage tests

**Files:**
- Create: `aquillm/tests/integration/test_observability_settings.py`
- Create: `aquillm/tests/integration/test_compose_observability_services.py`
- Modify: `aquillm/tests/integration/test_compose_multimodal_services.py` (if needed)

- [ ] **Step 1: Add tests verifying observability services exist in compose definitions**
- [ ] **Step 2: Add tests verifying logging env toggles are loaded and safe defaults remain**
- [ ] **Step 3: Commit**

```bash
git add aquillm/tests/integration/test_observability_settings.py aquillm/tests/integration/test_compose_observability_services.py aquillm/tests/integration/test_compose_multimodal_services.py
git commit -m "test(observability): add compose and settings integration coverage"
```

---

## Chunk 4: Normalize Existing Telemetry in Runtime Hotspots

### Task 9: Standardize key runtime telemetry logs (non-sensitive, parseable)

**Files:**
- Modify: `aquillm/apps/documents/services/chunk_search.py`
- Modify: `aquillm/apps/documents/services/rag_cache.py`
- Modify: `aquillm/lib/llm/utils/context_packer.py`
- Modify: `aquillm/lib/llm/utils/prompt_budget.py`
- Modify: `aquillm/apps/chat/consumers/chat_receive.py`
- Modify: `aquillm/apps/chat/consumers/chat_delta.py`
- Modify: `aquillm/apps/ingestion/services/upload_batches.py`
- Modify: `aquillm/apps/ingestion/services/arxiv_ingest.py`
- Modify: `aquillm/apps/ingestion/services/web_ingest.py`
- Modify: `aquillm/apps/documents/tasks/chunking.py`

- [ ] **Step 1: Define event naming convention (`obs.<domain>.<event>`)**
- [ ] **Step 2: Replace ad hoc free-text timing messages with structured key/value payloads**
- [ ] **Step 3: Explicitly avoid logging raw user prompts, tool payload bodies, or secrets**
- [ ] **Step 4: Keep existing pass/fail-open semantics unchanged**
- [ ] **Step 5: Commit**

```bash
git add aquillm/apps/documents/services/chunk_search.py aquillm/apps/documents/services/rag_cache.py aquillm/lib/llm/utils/context_packer.py aquillm/lib/llm/utils/prompt_budget.py aquillm/apps/chat/consumers/chat_receive.py aquillm/apps/chat/consumers/chat_delta.py aquillm/apps/ingestion/services/upload_batches.py aquillm/apps/ingestion/services/arxiv_ingest.py aquillm/apps/ingestion/services/web_ingest.py aquillm/apps/documents/tasks/chunking.py
git commit -m "obs(runtime): normalize telemetry logs across chat rag and ingestion hotspots"
```

### Task 10: Update docs and rollout guidance

**Files:**
- Modify: `README.md`
- Modify: `docs/roadmap/roadmap-status.md` (Phase 6 checklist notes only; do not mark complete without evidence)
- Modify: `docs/roadmap/plans/pending/2026-03-16-observability-implementation-plan.md` (checklist progression)

- [ ] **Step 1: Add observability startup/verification section to README**
- [ ] **Step 2: Document expected local URLs and auth defaults for Grafana**
- [ ] **Step 3: Add rollout and rollback notes for observability profile enablement**
- [ ] **Step 4: Commit**

```bash
git add README.md docs/roadmap/roadmap-status.md docs/roadmap/plans/pending/2026-03-16-observability-implementation-plan.md
git commit -m "docs(observability): add rollout runbook and roadmap phase notes"
```

---

## Chunk 5: Roadmap-Gated Follow-On (MCP/Skills/Agents)

### Task 11: Define deferred instrumentation hook points for Phase 4/4B/5 modules

**Files:**
- Modify later (when modules exist):
  - `aquillm/apps/chat/services/runtime_context.py`
  - `aquillm/apps/chat/services/tool_registry.py`
  - `aquillm/lib/mcp/*`
  - `aquillm/lib/skills/*`
  - `aquillm/lib/agents/*`

- [ ] **Step 1: Add explicit "blocked by roadmap dependency" notes in this plan**
- [ ] **Step 2: Re-open this chunk only after those modules land on branch**
- [ ] **Step 3: Instrument tool discovery errors, execution latency, and policy denials**
- [ ] **Step 4: Add regression tests for non-sensitive logging in agent runtime**

---

## Verification

### Python and integration checks

- [ ] Run:
  - `cd aquillm && pytest tests/integration/test_compose_multimodal_services.py -q`
  - `cd aquillm && pytest tests/integration/test_compose_observability_services.py -q`
  - `cd aquillm && pytest tests/integration/test_observability_settings.py -q`

### Runtime checks (development)

- [ ] Start with observability enabled:
  - `docker compose --env-file .env -f deploy/compose/development.yml --profile observability up -d`
- [ ] Verify:
  - `/health/` returns 200 on web port
  - `/metrics/` returns Prometheus format
  - Grafana loads and sees Prometheus + Pyroscope datasources
  - Log output changes with `LOG_JSON=true` and `LOG_LEVEL=DEBUG`

### Runtime checks (production compose definition)

- [ ] Config-only validation:
  - `docker compose --env-file .env -f deploy/compose/production.yml --profile observability config > NUL`
- [ ] Smoke startup in non-public environment before enabling on shared infra.

---

## Definition of Done

- [ ] Observability stack can be enabled via compose profile in both development and production definitions.
- [ ] `/metrics` is available and scrapeable from Prometheus.
- [ ] Web and worker support optional Pyroscope profiling with fail-open behavior.
- [ ] Logging supports env-controlled level and JSON/plain output.
- [ ] Existing RAG/chat/ingestion telemetry logs are standardized and non-sensitive.
- [ ] Docs include startup, verification, rollback, and operational ownership notes.

---

## Execution Handoff

Plan refreshed in place at `docs/roadmap/plans/pending/2026-03-16-observability-implementation-plan.md` to match current repository structure and roadmap sequencing.
