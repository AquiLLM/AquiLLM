# Large-Document Ingestion Memory Design

**Date:** 2026-08-04
**Status:** Approved for implementation planning
**Supersedes for implementation:** The parser/embed portions of `docs/specs/2026-03-26-ingestion-work-queue-batching-design.md`; queue classification, dispatcher, lease, and retry requirements from that design are incorporated here.

## Goal

Make document ingestion memory-bounded so large inputs either complete within an explicit worker budget or fail with a structured resource-limit error. Upload bursts must not multiply memory pressure beyond configured queue capacity.

## Current Failure Modes

The current pipeline has two independent failure classes:

1. A single task is unbounded. It reads the complete source into `bytes`, parsers build complete strings and payload lists, figure extraction retains every image, and chunking retains every chunk and 1,024-dimensional embedding until one final insert.
2. Cross-task concurrency is not workload-bounded. Upload tasks and document chunk tasks are placed on the main queue immediately, so parsing, OCR, transcription, and embedding can overlap across multiple worker processes.

A queue-only change limits the number of simultaneous failures but does not make one large document safe. A parser-only change leaves burst fan-out and hard-crash recovery unresolved. Both must be addressed.

## Decision Summary

Adopt a four-part bounded pipeline:

1. **Bound chunking and embeddings first.** Generate chunks lazily, embed fixed-size microbatches, persist each batch, and release it before continuing.
2. **Parse from a seekable disk source.** Copy storage-backed uploads to a temporary local file in fixed-size blocks. Parsers accept a path or opened stream and emit text segments and media payloads lazily.
3. **Spool extracted text with hard budgets.** Sanitize and hash text incrementally into a disk-backed spool. Materialize the final `full_text` string only after extraction succeeds and only up to the configured extracted-text limit.
4. **Shape queues and recover crashed work.** Route work by type, use low prefetch and explicit concurrency, add database leases/retries, and dispatch bounded batches.

The design intentionally preserves the current `Document.full_text` schema. The default extracted-text limit is 64 MiB, which is large enough for unusually large textual documents while keeping the final database save bounded. Supporting canonical text larger than this limit requires a separate schema design for paged/blob-backed document text and is out of scope.

## Success Criteria

- Production ingestion code contains no unrestricted `source_file.read()` or equivalent full-source byte materialization for document formats.
- Chunk embedding submits at most `INGEST_EMBED_BATCH_SIZE` chunks per provider request; default 64.
- Chunk database writes retain at most `INGEST_CHUNK_INSERT_BATCH_SIZE` objects; default 256.
- Extracted text is limited by UTF-8 bytes, default 64 MiB, independently of compressed upload size.
- ZIP-based formats enforce entry count, per-entry expanded bytes, total expanded bytes, and compression-ratio limits before extraction.
- Figure extraction enforces count, total encoded bytes, and decoded-pixel budgets and persists one figure before producing the next.
- A synthetic 50 MiB CSV completes below a 512 MiB parser-worker RSS budget in the benchmark harness.
- A 50 MiB extracted-text document never retains more than one embedding microbatch; the benchmark harness stays below a 384 MiB embed-worker RSS budget.
- A killed ingestion worker is recovered by lease expiry and retry rather than leaving an item permanently `processing`.
- Existing API response fields and document/chunk behavior remain compatible.

## Architecture

### Stored Source

`StoredIngestionSource` owns a temporary local file for one task. It copies `IngestionBatchItem.source_file` using fixed-size reads, verifies the observed size against the admission limit, and exposes a stable path for libraries that require seeking. The context manager always removes the temporary file.

This provides a common interface for filesystem and S3-backed Django storage without holding the source in memory.

### Resource Limits

`IngestionLimits` is an immutable configuration object constructed once per task. It contains:

- input bytes
- extracted UTF-8 text bytes
- parser segment bytes
- ZIP entry count and expansion budgets
- figure count, encoded bytes, and decoded pixels
- embed batch size and chunk insert batch size
- legacy parser input limit

All values must be positive; embed batches may not exceed insert batches; per-entry ZIP bytes may not exceed total ZIP bytes; and the figure-conversion source limit may not exceed the input limit. Defaults are 50 MiB input, 64 MiB extracted UTF-8 text, 256 KiB parser segments, 1,000 ZIP entries, 64 MiB total ZIP expansion, 16 MiB per ZIP entry, 64 MiB temporary archive disk, a 100:1 compression ratio, nesting depth 2, 50 figures, 64 MiB total encoded figure bytes, 40 megapixels per decoded figure, 500 megapixels cumulative decoded figures, 16,384 pixels on either image dimension, 64 embed inputs, 256 inserted chunks, 16 MiB for legacy parsers, and 20 MiB for Office-to-PDF figure conversion.

All resource-limit failures raise `IngestionResourceLimitError` with a stable code, configured limit, observed value, and phase. The task records those fields in `parser_metadata` and marks the item `error`; it does not retry deterministic limit failures.

### Streaming Parser Contract

The production entry point becomes:

```python
def iter_extracted_payloads(
    filename: str,
    source: StoredIngestionSource,
    *,
    content_type: str | None,
    limits: IngestionLimits,
) -> Iterator[ExtractedTextPayload]:
    ...
```

`ExtractedTextPayload` exposes `text_segments: Iterable[str]` rather than requiring a complete `full_text`. Every payload has a deterministic `payload_key`, and derived payloads have a `parent_payload_key`. Top-level archive members use a hash of their normalized member path as part of the key; figures add the format location and figure ordinal. Parent linkage is resolved by key, not iterator position.

Media is represented by a `MediaSource` with an `open() -> ContextManager[BinaryIO]` method plus filename, content type, expected bytes, and checksum. A payload never exposes an already-open stream. The consumer owns each returned context manager and must close it before advancing the payload iterator. Parser generators own and close their own document/archive handles in `finally` blocks.

The compatibility helper `extract_text_payloads(filename, data, ...)` may remain temporarily for small unit-test callers, but the Celery task must not use it. It applies the legacy input limit and is marked deprecated.

### Text Spooling and Persistence

`ExtractedTextSpool` accepts text segments, strips database-invalid NUL characters, inserts format-defined separators, updates SHA-256 incrementally, and writes UTF-8 to a `SpooledTemporaryFile` that rolls to disk after 1 MiB. It rejects the payload before exceeding the extracted-text budget.

After a payload finishes, the consumer decodes the bounded spool once and creates the existing document model. `Document.save()` accepts an optional validated precomputed hash so it does not encode the complete text again. PDF source storage is copied through Django `File` streaming rather than `ContentFile(data)`.

`IngestionArtifact` provides retry ownership across the database and object store. It has a unique `(ingestion_item, payload_key)` constraint and stores model label, document UUID, parent payload key, deterministic media object key, media checksum/bytes, and `writing|complete|error` state. The artifact row is created before persistence. Media is written to `ingestion_artifacts/<item-id>/<payload-key>/<safe-name>`; retries validate and reuse an existing checksum-matching object instead of creating a suffixed copy. A retry locks the artifact row and either reuses a complete document or resumes/cleans a stale `writing` artifact. A reconciliation task removes expired incomplete artifacts and their deterministic media objects. This makes duplicate delivery idempotent and gives every persisted object durable ownership even if a worker dies between storage and database operations.

Document saves during ingestion defer direct chunk-task enqueueing. Instead, the document and a `DocumentIndexJob` row are committed atomically. The index job is the durable outbox record and contains document UUID/model label, optional ingestion item, state, attempt/lease fields, and last error. A dispatcher publishes pending jobs to `ingest.embed`; broker publication failure leaves the row pending for reconciliation. This prevents chunk workers from observing half-finished ingestion state and closes the database-commit/broker-publication gap.

### Format Strategy

Formats receive one of two policies:

- **Streaming:** plain text, Markdown, CSV/TSV, JSONL, XML text, PDF pages, XLSX rows, DOCX paragraphs, PPTX slides, ODT/ODS/ODP XML, EPUB content documents, VTT/SRT captions, and ZIP archive members.
- **Bounded legacy adapter:** XLS, binary DOC/PPT, RTF, HTML, JSON, and YAML paths that still require full in-memory libraries. These run only below `INGEST_LEGACY_PARSER_MAX_BYTES`; larger inputs fail cleanly with `legacy_parser_limit`.

JSON and YAML preserve their current normalized output below the legacy limit. Larger JSON/YAML inputs fail cleanly rather than silently changing canonical text. XML and ZIP-based Office formats use incremental parsing and clear processed elements. For all migrated formats, fixture tests require byte-for-byte canonical extracted text and identical chunk boundaries for inputs below the legacy limit; changes to separators are not allowed in this project.

Nested archives share one `ExpansionBudget` instance across all recursion. It reserves declared entry bytes before opening, increments observed decompressed bytes on every streamed read, rejects when observed bytes exceed declared bytes or the shared total, and enforces compression ratio using observed bytes as well as metadata. Archive members are streamed to bounded temporary files, never `read()` into memory. Depth, entry count, expanded bytes, and temporary-disk consumption are cumulative across the complete nested archive tree.

### Figures

Figure extraction returns an iterator. Each figure is OCRed, saved, and released before the next figure is extracted. The budget object tracks cumulative encoded bytes and decoded pixels across the document.

Image headers are inspected before decoding. A figure is rejected before raster allocation if its width, height, or per-image pixel count exceeds its individual budget. Pillow decompression-bomb protection remains enabled; the pipeline does not suppress its warnings/errors. The cumulative pixel limit is defense in depth and is not relied on to control peak memory.

Office-to-PDF figure conversion is disabled above `INGEST_FIGURE_CONVERSION_MAX_SOURCE_BYTES`. Direct ZIP image extraction remains available within ZIP and figure budgets. Skipping optional figures is recorded as a warning in parser metadata and does not fail successful text ingestion.

### Chunking and Embedding

`iter_text_chunks()` yields chunk specifications without constructing the complete chunk list. Chunk visibility uses generation staging:

- `TextChunk` gains an indexed `index_generation` UUID, and its uniqueness constraints include that generation.
- `DocumentIndexState` has one row per document UUID and stores the active generation and active full-text hash.
- `DocumentIndexJob` is unique on `(document UUID, target full-text hash)` and each dispatch attempt receives a new generation UUID as well as a new lease token.
- Search querysets join/filter through `DocumentIndexState.active_generation`; chunks from an incomplete or stale generation are never visible.

The task computes total progress arithmetically and processes fixed-size batches:

1. build at most `INGEST_EMBED_BATCH_SIZE` `TextChunk` objects
2. request embeddings for that batch
3. fall back to per-chunk embedding only within that batch
4. insert the batch
5. release text and vectors

Every batch is tagged with the attempt generation. A stale worker may finish an already-started database insert after losing its lease, but those rows remain invisible because it cannot pass the lease-token compare-and-swap that atomically activates `DocumentIndexState.active_generation`. A replacement attempt writes a different generation, so the two workers cannot collide on chunk uniqueness. After activation, a cleanup task deletes inactive generations. No active generation is deleted at task start.

Duplicate-document chunk copying uses queryset `.iterator()` and the same generation-tagged insert batching. Existing chunks receive a migration-generated generation and corresponding `DocumentIndexState` row before generation filtering is enabled. This preserves existing search availability during rollout.

### Queue Shaping and Recovery

The queue model incorporates the existing 2026-03-26 design. `IngestionBatchItem` adds `work_type`, `queue_name`, `attempt_count`, `max_attempts`, `next_attempt_at`, `leased_until`, `lease_token`, `heartbeat_at`, `dispatched_at`, and `celery_task_id`. `DocumentIndexJob` has the same attempt/lease fields plus document UUID/model label, target full-text hash, attempt generation, and `pending|dispatched|processing|success|error` state. Its `(document UUID, target full-text hash)` uniqueness rule ensures there is only one logical indexing job per document version; redispatch changes the attempt generation rather than creating a concurrent logical job.

The fenced state machine is `queued/pending -> dispatched -> processing -> success|error`; a transient failure returns to `queued/pending` with `next_attempt_at`. A dispatcher creates a random lease token while claiming a row and passes it to the task. Task start, heartbeat, artifact completion, and terminal state updates use compare-and-swap filters containing row ID, expected state, and lease token. A stale worker whose lease was replaced cannot persist a new artifact state or mark the item/job complete. Active tasks heartbeat at a configurable interval, and lease recovery only redispatches work whose heartbeat and lease are both expired.

Database leases cover both ingestion items and `DocumentIndexJob` rows. The concrete behavior is:

- classify uploads into `ingest.text`, `ingest.ocr`, and `ingest.transcribe`
- route document chunks to `ingest.embed`
- run heavy queues with prefetch 1 and explicit concurrency
- admit work through a database dispatcher in bounded batches
- lease dispatched rows and retry expired leases with bounded exponential backoff
- enable late acknowledgements and rejection on worker loss for ingestion/embed tasks
- reject admission with `429` when global or per-user queue limits are exceeded

`worker_max_memory_per_child` and `worker_max_tasks_per_child` are defense-in-depth controls for process recycling; they are not substitutes for bounded task memory.

## Error Handling

- Deterministic resource-limit errors are terminal and use `input_limit`, `extracted_text_limit`, `parser_segment_limit`, `archive_entry_count_limit`, `archive_entry_bytes_limit`, `archive_expansion_limit`, `archive_ratio_limit`, `archive_depth_limit`, `archive_temp_disk_limit`, `figure_count_limit`, `figure_bytes_limit`, `figure_image_pixels_limit`, `figure_total_pixels_limit`, `figure_dimension_limit`, or `legacy_parser_limit`.
- Corrupt/unsupported documents are terminal and use `corrupt_document` or `unsupported_type`, with parser phase/type metadata.
- Storage timeouts, database operational errors, broker failures, provider timeouts, HTTP 429, and provider HTTP 5xx are transient and retried through the lease model.
- Authentication/configuration failures and provider HTTP 4xx other than 408/409/429 are terminal configuration errors.
- Embed context-limit errors may shrink only the affected input according to the existing context policy. A provider capability error may switch one bounded batch to the configured fallback provider. Generic batch failures never trigger an unbounded per-chunk retry loop; transient failures return the index job to pending.
- Figure OCR and optional figure conversion failures are warnings after one bounded attempt and do not fail source text. OCR for a top-level image and transcription for top-level audio/video are required and follow the transient/terminal taxonomy.
- A source document is not marked successful until all required text persistence completes.
- Optional figure failures produce warnings and do not discard source text.
- Partial chunks from a failed embedding task are deleted on retry.
- The existing behavior of appending processing errors to `Document.full_text` is removed; errors belong in task/item state, not canonical document content.

Every worker transition is fenced by the current lease token. If fencing fails, the worker stops without further writes. Terminal item state records a structured `error_code`, `error_phase`, and sanitized message; warnings are stored separately in parser metadata.

## Observability

Emit structured fields for:

- source bytes and extracted UTF-8 bytes
- parser type, elapsed time, and emitted segment count
- figure count, encoded bytes, and decoded pixels
- total chunks, current batch size, and embedding provider
- task RSS at phase boundaries when `INGEST_MEMORY_METRICS_ENABLED` is enabled
- queue depth, lease recoveries, retries, and wait-to-start latency

Do not log document text or media content.

## Testing Strategy

- Unit tests prove each limit at the exact boundary and one unit above it.
- Spy providers prove embedding requests and inserts never exceed configured batch sizes.
- Streaming-source tests use a file object that raises on unrestricted `read()`.
- Parser tests cover large synthetic CSV/JSONL, multi-page PDF, read-only XLSX, ZIP expansion limits, and figure-byte limits.
- Task tests prove payloads and media are consumed one at a time and temporary files are removed on success/failure.
- Duplicate-delivery tests prove one artifact/document per payload key.
- Stale-worker tests prove an expired lease token cannot save an artifact or terminal state.
- Index-generation tests prove stale workers cannot activate their chunks, partial generations are never searchable, and duplicate document-version jobs cannot execute concurrently.
- Crash-after-document/media persistence tests prove artifact reconciliation and retry reuse.
- Broker-publication failure tests prove pending index jobs are later dispatched.
- Crash-recovery tests expire a processing lease and verify redispatch.
- Nested-ZIP tests use lying metadata and enforce observed-byte accounting.
- A subprocess benchmark records peak RSS for representative 50 MiB text and figure-rich documents. It is an explicit performance gate, separate from fast unit tests.

## Rollout

1. Deploy bounded chunk/embedding batches first; behavior remains otherwise unchanged.
2. Deploy the chunk-generation migration, `DocumentIndexJob` outbox dispatcher, and `ingest.embed` worker. Backfill active generations before enabling generation-filtered search.
3. Deploy streaming source/text infrastructure behind `INGEST_STREAMING_PIPELINE_ENABLED=0`; its index jobs now have a live dispatcher from step 2.
4. Enable streaming in staging by format, beginning with TXT/CSV/PDF/XLSX.
5. Add ingestion queue-specific workers and late-ack/recovery settings, then enable the ingestion dispatcher in staging.
6. Run burst and forced-worker-kill tests, gradually enable production, and remove the legacy production parser entry point after every supported format has a streaming or explicitly bounded adapter.

Rollback uses the streaming and dispatcher flags. The bounded embedding batches and resource-limit definitions remain enabled because they are backward-compatible safety controls.

## Non-Goals

- Supporting more than the configured extracted-text ceiling in one `Document.full_text` row.
- Replacing Celery/Redis with Kafka or SQS.
- Replacing OCR, transcription, or embedding providers.
- Redesigning retrieval/reranking semantics.
- Changing current canonical JSON/YAML extraction within the supported legacy-parser limit.
