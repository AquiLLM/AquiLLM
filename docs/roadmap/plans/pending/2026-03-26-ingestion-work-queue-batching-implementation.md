# Ingestion Work Queue and Batching Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent ingestion bursts from overloading OCR, transcription, and embedding sidecars by combining queue-specific concurrency caps with a DB-backed bounded dispatcher.

**Architecture:** Implement in two near-term tracks: (A) immediate queue routing + worker caps, and (B) database dispatcher with leases/retries and bounded dispatch batches. Preserve Celery/Redis first, while creating a backend seam for Kafka/SQS later.

**Tech Stack:** Django, Celery, Redis, PostgreSQL, Docker Compose, pytest.

---

## File Structure and Ownership

- Create: `aquillm/apps/ingestion/services/queue_classification.py`
- Create: `aquillm/apps/ingestion/services/dispatcher.py`
- Create: `aquillm/apps/ingestion/tests/test_dispatcher.py`
- Create: `aquillm/apps/ingestion/tests/test_queue_classification.py`
- Create: `aquillm/apps/ingestion/migrations/0003_ingestion_queue_dispatch_fields.py`
- Modify: `aquillm/apps/ingestion/models/batch.py`
- Modify: `aquillm/apps/ingestion/services/upload_batches.py`
- Modify: `aquillm/apps/ingestion/views/api/uploads.py`
- Modify: `aquillm/aquillm/tasks.py`
- Modify: `aquillm/aquillm/task_ingest_uploaded.py`
- Modify: `aquillm/apps/documents/models/document.py`
- Modify: `aquillm/aquillm/settings.py`
- Modify: `aquillm/aquillm/celery.py`
- Modify: `deploy/compose/base.yml`
- Modify: `deploy/compose/development.yml`
- Modify: `.env.example`
- Modify: `aquillm/apps/ingestion/tests/test_unified_ingestion_api.py`
- Modify: `docs/documents/operations/ocr.md`

---

## Chunk 1: Queue Routing and Concurrency Caps (Option 1 baseline)

### Task 1: Add queue/work-type classification utility

**Files:**
- Create: `aquillm/apps/ingestion/services/queue_classification.py`
- Test: `aquillm/apps/ingestion/tests/test_queue_classification.py`

- [ ] **Step 1: Write failing tests for file-type to work-type/queue mapping.**
- [ ] **Step 2: Implement deterministic classifier (`text_light`, `ocr_heavy`, `transcribe_heavy`, `mixed`).**
- [ ] **Step 3: Verify tests pass.**

Run: `cd aquillm && pytest apps/ingestion/tests/test_queue_classification.py -q`  
Expected: PASS

- [ ] **Step 4: Commit.**

```bash
git add aquillm/apps/ingestion/services/queue_classification.py aquillm/apps/ingestion/tests/test_queue_classification.py
git commit -m "feat(ingestion): classify upload work types for queue routing"
```

### Task 2: Route ingestion upload tasks to workload queues

**Files:**
- Modify: `aquillm/apps/ingestion/services/upload_batches.py`
- Modify: `aquillm/aquillm/tasks.py`
- Modify: `aquillm/apps/ingestion/tests/test_unified_ingestion_api.py`

- [ ] **Step 1: Update enqueue path to set queue target metadata and use `apply_async(queue=...)` instead of raw `.delay()`.**
- [ ] **Step 2: Keep existing API response contract intact (`202`, per-item rows).**
- [ ] **Step 3: Add/adjust tests to assert queue-targeted dispatch behavior.**
- [ ] **Step 4: Run ingestion API tests.**

Run: `cd aquillm && pytest apps/ingestion/tests/test_unified_ingestion_api.py -q`  
Expected: PASS

- [ ] **Step 5: Commit.**

```bash
git add aquillm/apps/ingestion/services/upload_batches.py aquillm/aquillm/tasks.py aquillm/apps/ingestion/tests/test_unified_ingestion_api.py
git commit -m "feat(ingestion): route upload tasks to queue-specific workers"
```

### Task 3: Cap worker concurrency by queue in compose/env

**Files:**
- Modify: `deploy/compose/base.yml`
- Modify: `deploy/compose/development.yml`
- Modify: `.env.example`

- [ ] **Step 1: Add queue-specific worker command patterns and concurrency env values.**
- [ ] **Step 2: Add low prefetch guidance for heavy queues (`ocr`, `transcribe`).**
- [ ] **Step 3: Document defaults in `.env.example`.**
- [ ] **Step 4: Validate compose config.**

Run: `docker compose -f deploy/compose/base.yml -f deploy/compose/development.yml config`  
Expected: PASS with valid worker definitions

- [ ] **Step 5: Commit.**

```bash
git add deploy/compose/base.yml deploy/compose/development.yml .env.example
git commit -m "chore(ingestion): add queue-specific worker concurrency caps"
```

---

## Chunk 2: Database Dispatcher (Option 2 core)

### Task 4: Add dispatch metadata fields to `IngestionBatchItem`

**Files:**
- Modify: `aquillm/apps/ingestion/models/batch.py`
- Create: `aquillm/apps/ingestion/migrations/0003_ingestion_queue_dispatch_fields.py`

- [ ] **Step 1: Add model fields for work type, queue name, attempts, lease, and dispatch timing.**
- [ ] **Step 2: Generate and validate migration for current schema.**
- [ ] **Step 3: Run migration tests/check.**

Run: `cd aquillm && python manage.py makemigrations --check`  
Expected: PASS

- [ ] **Step 4: Commit.**

```bash
git add aquillm/apps/ingestion/models/batch.py aquillm/apps/ingestion/migrations/0003_ingestion_queue_dispatch_fields.py
git commit -m "feat(ingestion): add queue-dispatch metadata fields to batch items"
```

### Task 5: Implement bounded dispatcher service and periodic task

**Files:**
- Create: `aquillm/apps/ingestion/services/dispatcher.py`
- Modify: `aquillm/aquillm/tasks.py`
- Modify: `aquillm/aquillm/settings.py`
- Modify: `aquillm/aquillm/celery.py`
- Test: `aquillm/apps/ingestion/tests/test_dispatcher.py`

- [ ] **Step 1: Write failing tests for claim/lease/dispatch logic with skip-locked semantics.**
- [ ] **Step 2: Implement dispatcher claim loop with per-queue capacity checks and bounded batch size.**
- [ ] **Step 3: Add periodic schedule config (`INGEST_DISPATCH_INTERVAL_SEC`) and feature flag guard (`INGEST_DISPATCHER_ENABLED`).**
- [ ] **Step 4: Ensure dispatcher records `celery_task_id`, `dispatched_at`, and lease window.**
- [ ] **Step 5: Run dispatcher tests.**

Run: `cd aquillm && pytest apps/ingestion/tests/test_dispatcher.py -q`  
Expected: PASS

- [ ] **Step 6: Commit.**

```bash
git add aquillm/apps/ingestion/services/dispatcher.py aquillm/aquillm/tasks.py aquillm/aquillm/settings.py aquillm/aquillm/celery.py aquillm/apps/ingestion/tests/test_dispatcher.py
git commit -m "feat(ingestion): add bounded DB dispatcher for queued uploads"
```

### Task 6: Update worker task to honor lease/retry semantics

**Files:**
- Modify: `aquillm/aquillm/task_ingest_uploaded.py`
- Modify: `aquillm/apps/ingestion/tests/test_dispatcher.py`

- [ ] **Step 1: Ensure worker validates item state/lease on start and updates status transitions consistently.**
- [ ] **Step 2: Add retry bookkeeping (`attempt_count`, `next_attempt_at`) for transient failures.**
- [ ] **Step 3: Mark terminal failure after `max_attempts`.**
- [ ] **Step 4: Verify tests for retry and dead-letter behavior.**

Run: `cd aquillm && pytest apps/ingestion/tests/test_dispatcher.py -q`  
Expected: PASS

- [ ] **Step 5: Commit.**

```bash
git add aquillm/aquillm/task_ingest_uploaded.py aquillm/apps/ingestion/tests/test_dispatcher.py
git commit -m "feat(ingestion): add lease-aware retry handling for ingestion workers"
```

---

## Chunk 3: Embed Queue Isolation and API Visibility

### Task 7: Route chunking workloads to dedicated embed queue

**Files:**
- Modify: `aquillm/apps/documents/models/document.py`
- Modify: `aquillm/aquillm/settings.py`

- [ ] **Step 1: Change chunk task enqueue to explicit embed queue routing.**
- [ ] **Step 2: Add env-driven queue name/config for embed processing.**
- [ ] **Step 3: Run document/chunking tests.**

Run: `cd aquillm && pytest apps/documents/tests -q`  
Expected: PASS

- [ ] **Step 4: Commit.**

```bash
git add aquillm/apps/documents/models/document.py aquillm/aquillm/settings.py
git commit -m "feat(ingestion): isolate chunk embedding on dedicated queue"
```

### Task 8: Extend upload status endpoint with dispatch metadata

**Files:**
- Modify: `aquillm/apps/ingestion/views/api/uploads.py`
- Modify: `aquillm/apps/ingestion/tests/test_unified_ingestion_api.py`

- [ ] **Step 1: Add queue/work-type/attempt metadata to status payload.**
- [ ] **Step 2: Preserve existing counts contract while adding additive fields.**
- [ ] **Step 3: Add API tests for new fields and compatibility.**
- [ ] **Step 4: Run API tests.**

Run: `cd aquillm && pytest apps/ingestion/tests/test_unified_ingestion_api.py -q`  
Expected: PASS

- [ ] **Step 5: Commit.**

```bash
git add aquillm/apps/ingestion/views/api/uploads.py aquillm/apps/ingestion/tests/test_unified_ingestion_api.py
git commit -m "feat(ingestion): expose queue dispatch metadata in upload status API"
```

---

## Chunk 4: Operations, Rollout, and Future External Queue Seam

### Task 9: Document operational runbook and tuning knobs

**Files:**
- Modify: `docs/documents/operations/ocr.md`
- Modify: `README.md` (if needed)

- [ ] **Step 1: Document sidecar-safe defaults for queue caps and dispatcher batch size.**
- [ ] **Step 2: Document rollout sequence and rollback env toggles.**
- [ ] **Step 3: Include troubleshooting for queue backlog and lease expiries.**
- [ ] **Step 4: Commit.**

```bash
git add docs/documents/operations/ocr.md README.md
git commit -m "docs(ingestion): add queue shaping and dispatcher operations guide"
```

### Task 10: Add future backend seam for Kafka/SQS migration

**Files:**
- Modify: `aquillm/apps/ingestion/services/dispatcher.py`
- Create: `aquillm/apps/ingestion/services/dispatcher_backend.py`
- Test: `aquillm/apps/ingestion/tests/test_dispatcher.py`

- [ ] **Step 1: Introduce lightweight backend interface (`DatabaseDispatcherBackend` default).**
- [ ] **Step 2: Keep implementation no-op for Kafka/SQS now, but document extension points.**
- [ ] **Step 3: Verify tests still pass with default backend.**

Run: `cd aquillm && pytest apps/ingestion/tests/test_dispatcher.py -q`  
Expected: PASS

- [ ] **Step 4: Commit.**

```bash
git add aquillm/apps/ingestion/services/dispatcher.py aquillm/apps/ingestion/services/dispatcher_backend.py aquillm/apps/ingestion/tests/test_dispatcher.py
git commit -m "refactor(ingestion): add dispatcher backend seam for future kafka/sqs"
```

---

## End-to-End Verification

- [ ] **Step 1: Run targeted ingestion + document + OCR tests.**

Run: `cd aquillm && pytest apps/ingestion/tests apps/documents/tests lib/ocr/tests -q`  
Expected: PASS

- [ ] **Step 2: Run mixed-file burst smoke test in staging (manual/scripted).**
- [ ] **Step 3: Confirm sidecar stability and bounded active concurrency by queue.**
- [ ] **Step 4: Record measured queue depth and p95 wait-to-start before/after.**

---

## Rollout Strategy

1. Deploy with `INGEST_DISPATCHER_ENABLED=0` and queue caps enabled.
2. Enable dispatcher in staging with conservative values.
3. Tune `INGEST_MAX_ACTIVE_*` and `INGEST_DISPATCH_BATCH_SIZE`.
4. Enable in production gradually.
5. Trigger Kafka/SQS design/implementation only when measured load justifies external queue complexity.

