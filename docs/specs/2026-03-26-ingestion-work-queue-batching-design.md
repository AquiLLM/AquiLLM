# Ingestion Work Queue and Batching Design

**Date:** 2026-03-26

## Goal

Prevent ingestion bursts from overwhelming OCR, transcription, and embedding sidecar models by introducing explicit queue shaping, bounded dispatch, and per-work-type concurrency limits.

## Decision Summary

Adopt a phased hybrid approach:

1. **Phase 1 (Immediate):** Keep Celery/Redis, add strict worker concurrency caps and queue routing by workload type.
2. **Phase 1.5 (Near-term):** Add a **database-backed dispatcher** that admits, leases, and dispatches work in bounded batches.
3. **Phase 2 (Future):** Abstract dispatch backend and introduce Kafka or SQS when throughput/operational needs exceed Redis + DB dispatcher limits.

This combines option 1 (quick caps) and option 2 (structured dispatch), with option 3 deferred.

## Problem Statement

Current upload behavior enqueues one Celery task per file immediately. Large multi-file uploads can fan out quickly and saturate:

- OCR sidecar (`vllm_ocr`)
- transcription sidecar (`vllm_transcribe`)
- embedding sidecar (`vllm_embed`) through chunking

The system needs:

- bounded parallelism by workload type
- fair queue progression under bursts
- predictable latency and reduced sidecar overload
- clear status visibility for users and operators

## Current State (Repository-Aligned)

- Upload admission creates `IngestionBatchItem` rows and immediately calls `ingest_uploaded_file_task.delay(item.id)`.
- Document save triggers `create_chunks.delay(str(self.id))`, which can create embedding pressure independently of upload queue pressure.
- Status endpoint reports coarse item states (`queued`, `processing`, `success`, `error`) but no queue-depth/backpressure telemetry.

## Scope

- Queue shaping for uploaded ingestion files.
- Concurrency controls for OCR/transcribe/embed-heavy workloads.
- Bounded dispatch batches and leasing semantics.
- Retry/backoff for transient failures.
- API/status surface updates for queue visibility.

## Non-Goals

- Replacing Celery in Phase 1.
- Rewriting parser/OCR/transcribe providers.
- Implementing Kafka/SQS in this first execution wave.

## Proposed Architecture

### 1. Workload Classification at Admission

At upload enqueue time, classify each item into a workload class (and queue target):

- `text_light` -> `ingest.text`
- `ocr_heavy` -> `ingest.ocr`
- `transcribe_heavy` -> `ingest.transcribe`
- `mixed` -> conservative fallback queue (`ingest.ocr` or `ingest.text` based on file type policy)

Classification uses existing file-type detection hints (extension + content type), before processing body text.

### 2. Immediate Concurrency Caps (Phase 1)

Route Celery tasks to dedicated queues and run dedicated workers (or queue-specific concurrency overrides):

- `ingest.text` higher concurrency
- `ingest.ocr` low concurrency
- `ingest.transcribe` very low concurrency
- `ingest.embed` low-medium concurrency

Also configure low prefetch for heavy queues to avoid one worker reserving too many expensive tasks.

### 3. Database-Backed Dispatcher (Phase 1.5)

Introduce a periodic dispatcher task (`dispatch_ingestion_items`) that:

1. reads current active counts per queue
2. computes available capacity by queue (`max_active - current_active`)
3. claims eligible queued rows via transactional lease (`select_for_update(skip_locked)`)
4. dispatches only up to bounded batch sizes per queue
5. records dispatch metadata (`dispatched_at`, `celery_task_id`, `leased_until`)

This turns upload spikes into controlled queue drain, not instantaneous worker fan-out.

### 4. Lease + Retry Model

Each work item supports:

- optimistic lease timeout (recover from worker crash)
- retry attempts with exponential backoff and jitter
- dead-letter state after `max_attempts`

On transient failure:

- increment `attempt_count`
- set `next_attempt_at`
- return to queued state

On terminal failure:

- set status `error`
- preserve structured `error_message`

### 5. Embedding Pressure Control

Route chunking/embedding work to an explicit `ingest.embed` queue and cap concurrency there, independent of file ingestion parsing queues.

This prevents file parsing throughput from collapsing due to embedding sidecar saturation.

### 6. Backpressure Policy

Add admission guardrails:

- global max queued+processing work items
- optional per-user active queue cap

If limits exceeded:

- return `429` with a retry hint
- keep system healthy rather than accepting unbounded work

### 7. Status and UX Enhancements

Extend upload status response with:

- per-queue counts
- estimated queue position (best effort)
- retry attempt counts
- whether item is waiting for lease/dispatch

Frontend polling can show "queued for OCR" vs generic "queued".

## Data Model Changes

Extend `IngestionBatchItem` with queue/dispatch fields:

- `work_type` (`text_light|ocr_heavy|transcribe_heavy|mixed`)
- `queue_name`
- `priority` (integer)
- `attempt_count`
- `max_attempts`
- `next_attempt_at`
- `leased_until`
- `dispatched_at`
- `celery_task_id`

Status values remain user-facing (`queued|processing|success|error`) to minimize UI breakage; dispatch internals are additive metadata.

## Configuration (Environment)

Add new settings/env knobs:

- `INGEST_DISPATCHER_ENABLED`
- `INGEST_DISPATCH_INTERVAL_SEC`
- `INGEST_DISPATCH_BATCH_SIZE`
- `INGEST_MAX_ACTIVE_TOTAL`
- `INGEST_MAX_ACTIVE_TEXT`
- `INGEST_MAX_ACTIVE_OCR`
- `INGEST_MAX_ACTIVE_TRANSCRIBE`
- `INGEST_MAX_ACTIVE_EMBED`
- `INGEST_LEASE_SECONDS`
- `INGEST_RETRY_BASE_SECONDS`
- `INGEST_RETRY_MAX_SECONDS`
- `INGEST_MAX_QUEUE_DEPTH`
- `INGEST_MAX_ACTIVE_PER_USER`

## Failure Modes and Handling

- **Worker crash after dispatch:** lease expiry returns item to eligible queue state.
- **Sidecar slowdown/outage:** queue depth increases, but active work remains bounded.
- **Dispatcher overlap/race:** row-level locking + skip-locked claims prevent double-dispatch.
- **Partial burst failure:** per-item isolation retains mixed success behavior.

## Observability

Track and expose:

- queue depth by queue/work type
- dispatcher loop runtime and dispatch counts
- lease expiry recovery count
- retry count and terminal failure rate
- p50/p95 wait-to-start latency
- sidecar-specific processing latency

## Rollout Plan

1. Ship queue routing + worker caps behind flags (dispatcher disabled).
2. Validate queue isolation and sidecar stability in staging.
3. Enable dispatcher in staging with conservative batch sizes.
4. Enable in production gradually.
5. Tune per-queue caps and batch sizes from observed latency/failure metrics.
6. Introduce backend abstraction and evaluate Kafka/SQS migration trigger thresholds.

## Kafka/SQS Future Path

Design for future queue backend abstraction now:

- `WorkDispatcherBackend` interface
- `DatabaseDispatcherBackend` (Phase 1.5 default)
- `ExternalQueueBackend` (Kafka/SQS future)

Migration trigger examples:

- sustained queue depths beyond DB/Redis comfort
- need for cross-service multi-consumer scaling
- stricter ordering/retention/replay needs

## Verification

- Burst test: 100 mixed files yields bounded active OCR/transcribe/embed concurrency.
- No unbounded immediate task fan-out at upload admission.
- Queue depths drain steadily under load without sidecar crashes.
- Retry and lease recovery work after forced worker kill.
- Status API reflects queue state accurately enough for user feedback.
