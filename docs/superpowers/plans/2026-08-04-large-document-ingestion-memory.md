# Large-Document Ingestion Memory Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make document ingestion memory-bounded, generation-safe, crash-recoverable, and capable of ingesting 50 MiB textual documents without exhausting a worker.

**Architecture:** Replace whole-file parsing with a seekable temporary source, lazy payload/text iterators, and a disk-backed bounded text spool. Stage chunk embeddings in fixed-size generations, atomically activate completed generations, and use database outbox/lease rows to shape ingestion and embed work safely across crashes.

**Tech Stack:** Python 3.12, Django, PostgreSQL/pgvector, Celery, Redis, Django storage/S3, pypdf, openpyxl, Pillow, pytest, Docker Compose.

**Reference spec:** `docs/superpowers/specs/2026-08-04-large-document-ingestion-memory-design.md`

**Plan relationship:** This plan absorbs and supersedes `docs/roadmap/plans/pending/2026-03-26-ingestion-work-queue-batching-implementation.md`. Do not execute both plans.

---

## File Structure and Ownership

### Ingestion resource and streaming core

- Create: `aquillm/aquillm/ingestion/limits.py` — immutable limits, shared archive/figure accounting, stable limit errors.
- Create: `aquillm/aquillm/ingestion/source.py` — storage-to-local fixed-block materialization and bounded media sources.
- Create: `aquillm/aquillm/ingestion/text_spool.py` — disk-backed text accumulation, sanitation, byte accounting, and incremental hash.
- Modify: `aquillm/aquillm/ingestion/types.py` — lazy payload keys, parent keys, segment iterables, and `MediaSource`.
- Modify: `aquillm/aquillm/ingestion/parsers.py` — production iterator registry and bounded compatibility entry point.
- Modify: `aquillm/aquillm/ingestion/parser_archive.py` — nested iterator with shared observed expansion accounting.
- Modify: `aquillm/aquillm/ingestion/parser_figures.py` — yield one budgeted figure payload at a time.
- Create: `aquillm/aquillm/task_ingest_persistence.py` — artifact-owned, idempotent payload persistence.
- Modify: `aquillm/aquillm/task_ingest_uploaded.py` — fenced orchestration only.

### Format adapters

- Modify: `aquillm/lib/parsers/text_utils.py` and `aquillm/lib/parsers/media/vtt.py` — bounded text/transcript iterators.
- Modify: `aquillm/lib/parsers/spreadsheets/csv_parser.py`, `xlsx.py`, and `ods.py` — lazy row/XML extraction.
- Modify: `aquillm/lib/parsers/documents/pdf.py`, `docx.py`, and `epub.py`; create `odt.py` — page/ZIP-member iterators.
- Modify: `aquillm/lib/parsers/presentations/pptx.py` and `odp.py` — bounded slide/XML iterators.
- Modify: `aquillm/lib/parsers/structured/json_parser.py`, `xml_parser.py`, and `yaml_parser.py` — bounded legacy/incremental policies.
- Modify: each corresponding parser package `__init__.py` plus `aquillm/lib/parsers/__init__.py` — iterator exports.
- Modify: the explicitly listed files under `aquillm/aquillm/ingestion/figure_extraction/` in Task 12 — one-at-a-time figure budgets.

### Document indexing

- Create: `aquillm/apps/documents/models/indexing.py` — `DocumentIndexState` and `DocumentIndexJob`.
- Modify: `aquillm/apps/documents/models/chunks.py` — index generation field and generation-aware constraints/queryset.
- Modify: `aquillm/apps/documents/models/__init__.py` — indexing model exports.
- Create: `aquillm/apps/documents/migrations/0004_document_index_generations.py` — schema plus active-generation backfill.
- Create: `aquillm/apps/documents/services/chunk_batches.py` — lazy chunk specs and bounded batches.
- Create: `aquillm/apps/documents/services/index_jobs.py` — durable job creation, dispatch, fencing, activation, and cleanup.
- Modify: `aquillm/apps/documents/tasks/chunking.py` — bounded generation writer.
- Modify: `aquillm/apps/documents/models/document.py` — precomputed hash support and atomic index-job creation.
- Modify: `aquillm/apps/documents/services/chunk_search.py` — active-generation filtering.
- Modify: `aquillm/apps/documents/views/api.py`, `aquillm/apps/documents/views/pages.py`, and `aquillm/apps/documents/models/document.py` — generation-filtered direct chunk reads.

### Queue ownership and recovery

- Modify: `aquillm/apps/ingestion/models/batch.py` — lease, retry, error, and warning fields.
- Create: `aquillm/apps/ingestion/models/artifact.py` — deterministic `IngestionArtifact` ownership.
- Modify: `aquillm/apps/ingestion/models/__init__.py` — artifact export.
- Create: `aquillm/apps/ingestion/migrations/0003_ingestion_queue_leases.py`.
- Create: `aquillm/apps/ingestion/migrations/0004_ingestion_artifacts.py`.
- Create: `aquillm/apps/ingestion/services/queue_classification.py` — deterministic work-type mapping.
- Create: `aquillm/apps/ingestion/services/dispatcher.py` — bounded claims, publication, heartbeat, fencing, recovery.
- Create: `aquillm/apps/ingestion/services/leases.py` — shared compare-and-swap lease transitions and guards.
- Modify: `aquillm/apps/ingestion/services/upload_batches.py` — admission only; no immediate fan-out when dispatcher is enabled.
- Modify: `aquillm/apps/ingestion/views/api/uploads.py` — structured limits and queue status.
- Modify: `aquillm/aquillm/tasks.py` — bound task wrappers and reconciliation tasks.
- Modify: `aquillm/aquillm/celery.py` and `aquillm/aquillm/settings.py` — routes, schedules, late acknowledgements, and settings.
- Modify: `deploy/compose/base.yml`, `deploy/compose/development.yml`, and `deploy/compose/production.yml` — queue-specific workers.
- Modify: `.env.example` — documented defaults.

### Tests and operations

- Create: `aquillm/apps/ingestion/tests/test_ingestion_limits.py`.
- Create: `aquillm/apps/ingestion/tests/test_stored_source.py`.
- Create: `aquillm/apps/ingestion/tests/test_text_spool.py`.
- Create: `aquillm/apps/ingestion/tests/test_streaming_parsers.py`.
- Create: `aquillm/apps/ingestion/tests/test_streaming_archive.py`.
- Create: `aquillm/apps/ingestion/tests/test_ingest_idempotency.py`.
- Create: `aquillm/apps/ingestion/tests/test_dispatcher.py`.
- Create: `aquillm/apps/ingestion/tests/test_ingestion_observability.py`.
- Create: `aquillm/apps/ingestion/tests/fixtures/streaming/simple_formats.json` and `container_formats.json`.
- Create: `aquillm/apps/documents/tests/test_chunk_batches.py`.
- Create: `aquillm/apps/documents/tests/test_index_generations.py`.
- Create: `aquillm/apps/documents/tests/test_index_dispatcher.py`.
- Create: `scripts/benchmark_ingestion_memory.py`.
- Create: `scripts/verify_ingestion_resilience.py`.
- Create: `tests/unit/test_ingestion_resilience_harness.py`.
- Create: `tests/unit/test_ingestion_memory_benchmark_harness.py`.
- Create: `aquillm/apps/ingestion/management/__init__.py`.
- Create: `aquillm/apps/ingestion/management/commands/__init__.py`.
- Create: `aquillm/apps/ingestion/management/commands/ingestion_diagnostics.py`.
- Create: `aquillm/apps/ingestion/tests/test_ingestion_diagnostics.py`.
- Create: `docs/documents/operations/large-document-ingestion.md`.

---

## Chunk 1: Bound and Stage Document Indexing

### Task 1: Add lazy chunk and batching primitives

**Files:**
- Create: `aquillm/apps/documents/services/chunk_batches.py`
- Create: `aquillm/apps/documents/tests/test_chunk_batches.py`

- [ ] **Step 1: Write failing tests for chunk compatibility and bounded iteration.**

```python
def test_iter_text_chunks_preserves_current_boundaries():
    text = "x" * 5000
    chunks = list(iter_text_chunks(text, chunk_size=2048, overlap=384))
    assert [(c.start, c.end) for c in chunks] == [(0, 2048), (1664, 3712), (3328, 5000), (4992, 5000)]

def test_batched_never_reads_past_requested_batch():
    source = CountingIterator(1000)
    first = next(batched(source, 64))
    assert len(first) == 64
    assert source.consumed == 64
```

- [ ] **Step 2: Run the tests and confirm missing imports/functions fail.**

Run: `rtk pytest aquillm/apps/documents/tests/test_chunk_batches.py -q`
Expected: FAIL because `chunk_batches` does not exist.

- [ ] **Step 3: Implement immutable `ChunkSpec`, `iter_text_chunks()`, and generic `batched()`.**

```python
@dataclass(frozen=True)
class ChunkSpec:
    content: str
    start: int
    end: int
    number: int

def iter_text_chunks(text: str, *, chunk_size: int, overlap: int) -> Iterator[ChunkSpec]:
    pitch = chunk_size - overlap
    if chunk_size <= 0 or pitch <= 0:
        raise ValueError("chunk_size must be positive and overlap smaller than chunk_size")
    for number, start in enumerate(range(0, len(text), pitch)):
        end = min(start + chunk_size, len(text))
        yield ChunkSpec(text[start:end], start, end, number)
```

- [ ] **Step 4: Run the focused tests.**

Run: `rtk pytest aquillm/apps/documents/tests/test_chunk_batches.py -q`
Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add aquillm/apps/documents/services/chunk_batches.py aquillm/apps/documents/tests/test_chunk_batches.py
git commit -m "feat(documents): add lazy bounded chunk batches"
```

### Task 2: Add generation-staged index models and migration

**Files:**
- Create: `aquillm/apps/documents/models/indexing.py`
- Modify: `aquillm/apps/documents/models/chunks.py`
- Modify: `aquillm/apps/documents/models/__init__.py`
- Create: `aquillm/apps/documents/migrations/0004_document_index_generations.py`
- Create: `aquillm/apps/documents/tests/test_index_generations.py`

- [ ] **Step 1: Write model tests for job uniqueness, generation-aware chunk uniqueness, and active state.**
- [ ] **Step 2: Run the model tests and confirm they fail before the models exist.**

Run: `rtk pytest aquillm/apps/documents/tests/test_index_generations.py -q`
Expected: FAIL on missing `DocumentIndexState`/`DocumentIndexJob`.

- [ ] **Step 3: Implement the models with the exact invariants from the spec.**

```python
class DocumentIndexState(models.Model):
    doc_id = models.UUIDField(unique=True)
    active_generation = models.UUIDField(null=True)
    active_full_text_hash = models.CharField(max_length=64, blank=True, default="")

class DocumentIndexJob(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending"
        DISPATCHED = "dispatched"
        PROCESSING = "processing"
        SUCCESS = "success"
        ERROR = "error"
    doc_id = models.UUIDField()
    document_model = models.CharField(max_length=100)
    target_full_text_hash = models.CharField(max_length=64)
    ingestion_item = models.ForeignKey(
        "apps_ingestion.IngestionBatchItem",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="index_jobs",
    )
    attempt_generation = models.UUIDField(null=True)
    lease_token = models.UUIDField(null=True)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING)
    attempt_count = models.PositiveIntegerField(default=0)
    max_attempts = models.PositiveIntegerField(default=3)
    next_attempt_at = models.DateTimeField(null=True)
    leased_until = models.DateTimeField(null=True)
    heartbeat_at = models.DateTimeField(null=True)
    dispatched_at = models.DateTimeField(null=True)
    celery_task_id = models.CharField(max_length=255, blank=True, default="")
    last_error_code = models.CharField(max_length=64, blank=True, default="")
    last_error_message = models.TextField(blank=True, default="")

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["doc_id", "target_full_text_hash"],
                name="unique_document_index_version",
            )
        ]
```

- [ ] **Step 4: Add nullable `TextChunk.index_generation`, include it in both new uniqueness constraints only after backfill, and index `(doc_id, index_generation)`.**
- [ ] **Step 5: Write the migration in three explicit phases: add the nullable field/models; iterate distinct `TextChunk.doc_id` values and assign exactly one UUID to all chunks for each UUID while creating one `DocumentIndexState`; then alter `index_generation` to non-null and replace the old uniqueness constraints. Documents with no chunks intentionally receive no state row until their first index job.**
- [ ] **Step 6: Store concrete model identity as Django `_meta.label` (`apps_documents.PDFDocument`, etc.); do not attempt to backfill a model label from the abstract `Document`. Add migration tests with existing chunks and with chunkless documents.**
- [ ] **Step 7: Run migration and model checks.**

Run: `cd aquillm && rtk python manage.py makemigrations --check && rtk python manage.py migrate --plan`
Expected: no uncommitted model changes and a valid migration plan.

- [ ] **Step 8: Run model tests and commit.**

```bash
git add aquillm/apps/documents/models aquillm/apps/documents/migrations/0004_document_index_generations.py aquillm/apps/documents/tests/test_index_generations.py
git commit -m "feat(documents): add generation-staged document indexes"
```

### Task 3: Gate all retrieval on the active chunk generation

**Files:**
- Modify: `aquillm/apps/documents/models/chunks.py`
- Modify: `aquillm/apps/documents/services/chunk_search.py`
- Modify: `aquillm/lib/tools/search/vector_search.py`
- Modify: `aquillm/apps/documents/views/api.py`
- Modify: `aquillm/apps/documents/views/pages.py`
- Modify: `aquillm/apps/documents/models/document.py`
- Modify: `aquillm/apps/documents/tests/test_index_generations.py`

- [ ] **Step 1: Add failing tests with active, partial, and stale generations.**
- [ ] **Step 2: Prove current search returns partial/stale rows.**

Run: `rtk pytest aquillm/apps/documents/tests/test_index_generations.py -q`
Expected: FAIL because generation visibility is not filtered.

- [ ] **Step 3: Add one reusable queryset method and route every document-search entry point through it.**

```python
def active(self):
    return self.filter(
        index_generation=models.Subquery(
            DocumentIndexState.objects.filter(doc_id=models.OuterRef("doc_id"))
            .values("active_generation")[:1]
        )
    )
```

- [ ] **Step 4: Route `Document.chunks`, citation/detail API lookups, document-page chunk highlighting, and tool search through `.active()`. Only index-job internals may intentionally query staging rows.**
- [ ] **Step 5: Search for direct `TextChunk.objects` reads and require an inline `# staging-generation access` comment on every intentional exception.**

Run: `rtk rg -n "TextChunk\.objects|objects\.filter\(.*doc_id" aquillm -g '*.py'`
Expected: production retrieval paths use the active-generation queryset.

- [ ] **Step 6: Run document/search suites and commit.**

Run: `rtk pytest aquillm/apps/documents/tests aquillm/lib/tools/search/tests -q`
Expected: PASS.

### Task 4: Rewrite chunking as bounded generation batches

**Files:**
- Create: `aquillm/apps/documents/services/index_jobs.py`
- Modify: `aquillm/apps/documents/tasks/chunking.py`
- Modify: `aquillm/apps/documents/models/document.py`
- Create: `aquillm/apps/documents/tests/test_index_dispatcher.py`
- Modify: `aquillm/apps/documents/tests/test_index_generations.py`

- [ ] **Step 1: Write failing tests proving provider batches never exceed 64, inserts never exceed 256, partial generations remain invisible, and concrete `_meta.label` rehydration works for PDF, raw-text, image, media, and figure models.**
- [ ] **Step 2: Add stale-lease tests in which attempt A writes a batch, attempt B replaces its token/generation, and attempt A cannot activate.**
- [ ] **Step 3: Implement `ensure_index_job()` with unique `(doc_id, target_full_text_hash)`, concrete `_meta.label`, optional `ingestion_item`, and explicit reuse/reset semantics. All state-changing index paths use one lock order: job row, concrete document row, then index-state row. In `ensure_index_job()`, get/create and lock the target-hash job first, then lock/revalidate the concrete row by its UUID `id`; never hold the document lock while acquiring a job lock. Reuse an existing successful job only when its generation rows still exist and `DocumentIndexState` still activates that generation. If a document reverts to an older hash whose successful generation was cleaned or is no longer active, reset that same unique job row to `pending`: assign no generation yet, clear lease/task/error/timing fields, reset attempts, and dispatch it again. If the document changed before revalidation, leave the stale target job undispatched and retry `ensure_index_job()` for the current hash after commit.**
- [ ] **Step 4: Implement activation as one ordered transaction that locks the job, then the concrete document row, then the index-state row. Validate token/generation, check supersession before recording success, activate the generation, and mark success in the same commit. On a hash mismatch, mark the old job `error` with `last_error_code="superseded_version"`; after that transaction, call `ensure_index_job()` for the locked document's current hash.**

```python
with transaction.atomic():
    job = DocumentIndexJob.objects.select_for_update().get(pk=job_id)
    if (
        job.status != DocumentIndexJob.Status.PROCESSING
        or job.lease_token != lease_token
        or job.attempt_generation != generation
    ):
        raise StaleLeaseError(job_id)

    model = apps.get_model(job.document_model)
    document = model.objects.select_for_update().get(id=job.doc_id)
    if document.full_text_hash != job.target_full_text_hash:
        job.mark_superseded()
        superseded = True
    else:
        state, _ = DocumentIndexState.objects.select_for_update().get_or_create(
            doc_id=job.doc_id
        )
        state.activate(generation, job.target_full_text_hash)
        job.mark_success()
        superseded = False

if superseded:
    ensure_index_job(document)
    raise SupersededIndexVersion(job_id)
```

- [ ] **Step 5: Add race tests proving activation cannot succeed after the concrete row changes, job success and state activation cannot split across commits, concurrent `ensure_index_job()` and activation obey job → document → state ordering without deadlock, and revert-after-cleanup resets/rebuilds the existing unique job. Never delete the previous active generation before activation.**
- [ ] **Step 6: Rewrite `create_chunks` to iterate `ChunkSpec` batches, embed one batch, `bulk_create(..., batch_size=...)`, release it, then heartbeat.**
- [ ] **Step 7: Replace generic per-chunk fallback with the spec taxonomy: context/capability fallback stays within one batch; timeout/429/5xx requeues the job.**
- [ ] **Step 8: Stream duplicate-document chunk copying with queryset `.iterator(chunk_size=...)`.**
- [ ] **Step 9: Remove the code that appends failures to `Document.full_text`.**
- [ ] **Step 10: Add `DOCUMENT_INDEX_OUTBOX_ENABLED`: when false, create the durable job but publish it with the legacy `transaction.on_commit()` path; when true, only the outbox dispatcher publishes. Do not enable the flag until the embed dispatcher/worker from Task 16 is deployed.**
- [ ] **Step 11: Run focused and full document tests.**

Run: `rtk pytest aquillm/apps/documents/tests/test_chunk_batches.py aquillm/apps/documents/tests/test_index_generations.py aquillm/apps/documents/tests/test_index_dispatcher.py -q`
Expected: PASS with observed batch maxima at configured values.

- [ ] **Step 12: Commit.**

```bash
git add aquillm/apps/documents aquillm/apps/documents/tests
git commit -m "feat(documents): index chunks in bounded atomic generations"
```

---

## Chunk 2: Stream and Own Ingestion Data

### Task 5: Define validated ingestion limits and accounting

**Files:**
- Create: `aquillm/aquillm/ingestion/limits.py`
- Create: `aquillm/apps/ingestion/tests/test_ingestion_limits.py`
- Modify: `aquillm/aquillm/settings.py`
- Modify: `.env.example`

- [ ] **Step 1: Write boundary tests for every stable error code listed in the spec.**
- [ ] **Step 2: Implement `IngestionLimits`, `ExpansionBudget`, `FigureBudget`, and `IngestionResourceLimitError`.**

```python
class IngestionResourceLimitError(RuntimeError):
    def __init__(self, *, code: str, phase: str, limit: int, observed: int):
        self.code, self.phase = code, phase
        self.limit, self.observed = limit, observed
        super().__init__(f"{code}: observed {observed}, limit {limit}")
```

- [ ] **Step 3: Validate cross-field relationships at settings construction and fail startup on invalid values.**
- [ ] **Step 4: Run settings and limit tests.**

Run: `rtk pytest aquillm/apps/ingestion/tests/test_ingestion_limits.py aquillm/tests/integration/test_settings_security_flags.py -q`
Expected: PASS.

- [ ] **Step 5: Commit.**

### Task 6: Materialize storage sources without whole-file reads

**Files:**
- Create: `aquillm/aquillm/ingestion/source.py`
- Create: `aquillm/apps/ingestion/tests/test_stored_source.py`

- [ ] **Step 1: Create a test source whose `read(-1)` raises and whose bounded reads record their maximum size.**
- [ ] **Step 2: Write failing tests for S3-shaped streams, early EOF, oversized observed bytes, checksum, and cleanup after exceptions.**
- [ ] **Step 3: Implement `StoredIngestionSource` as a context manager using `NamedTemporaryFile(delete=False)` and fixed 1 MiB reads.**
- [ ] **Step 4: Implement `MediaSource.open()` without retaining an already-open stream.**
- [ ] **Step 5: Run tests and verify no test permits unrestricted reads.**

Run: `rtk pytest aquillm/apps/ingestion/tests/test_stored_source.py -q`
Expected: PASS; maximum source read request is 1 MiB.

- [ ] **Step 6: Commit.**

### Task 7: Add bounded text spooling and precomputed document hashes

**Files:**
- Create: `aquillm/aquillm/ingestion/text_spool.py`
- Create: `aquillm/apps/ingestion/tests/test_text_spool.py`
- Modify: `aquillm/apps/documents/models/document.py`
- Modify: `aquillm/apps/documents/models/document_types/pdf.py`

- [ ] **Step 1: Write tests for incremental sanitation, exact separators, UTF-8 byte boundaries, rollover to disk, checksum equivalence, and cleanup.**
- [ ] **Step 2: Implement `ExtractedTextSpool.write_segment()` and `materialize()`.**
- [ ] **Step 3: Add a `precomputed_full_text_hash` keyword to `Document.save()` and verify it matches `hash_fn(full_text)` before trusting it in debug/tests.**
- [ ] **Step 4: Ensure the PDF subclass forwards the keyword without triggering its legacy extraction method.**
- [ ] **Step 5: Run spool and document tests.**

Run: `rtk pytest aquillm/apps/ingestion/tests/test_text_spool.py aquillm/apps/documents/tests -q`
Expected: PASS.

- [ ] **Step 6: Commit.**

### Task 8: Introduce lazy payload and parser interfaces

**Files:**
- Modify: `aquillm/aquillm/ingestion/types.py`
- Modify: `aquillm/aquillm/ingestion/parsers.py`
- Modify: `aquillm/aquillm/ingestion/__init__.py`
- Create: `aquillm/apps/ingestion/tests/test_streaming_parsers.py`
- Modify: `aquillm/apps/ingestion/tests/test_unified_ingestion_parsers.py`

- [ ] **Step 1: Write contract tests for deterministic payload keys, parent keys, lazy segment consumption, and media context ownership.**
- [ ] **Step 2: Run the tests and confirm the old eager payload contract fails.**

Run: `rtk pytest aquillm/apps/ingestion/tests/test_streaming_parsers.py -q`
Expected: FAIL because payload keys, segment iterables, and media sources do not exist.

- [ ] **Step 3: Implement the contract without changing parser output.**

```python
@dataclass
class ExtractedTextPayload:
    payload_key: str
    title: str
    normalized_type: str
    text_segments: Iterable[str]
    parent_payload_key: str | None = None
    media_source: MediaSource | None = None
    modality: str = "text"
    metadata: dict[str, Any] = field(default_factory=dict)
```

- [ ] **Step 4: Add `iter_extracted_payloads()` and keep `extract_text_payloads()` only as a size-limited compatibility collector.**
- [ ] **Step 5: Add a test that production task code imports only `iter_extracted_payloads`.**
- [ ] **Step 6: Run parser contract tests.**

Run: `rtk pytest aquillm/apps/ingestion/tests/test_streaming_parsers.py aquillm/apps/ingestion/tests/test_unified_ingestion_parsers.py -q`
Expected: PASS.

- [ ] **Step 7: Commit.**

### Task 9: Stream simple and structured textual formats

**Files:**
- Modify: `aquillm/lib/parsers/text_utils.py`
- Modify: `aquillm/lib/parsers/spreadsheets/csv_parser.py`
- Modify: `aquillm/lib/parsers/spreadsheets/__init__.py`
- Modify: `aquillm/lib/parsers/structured/json_parser.py`
- Modify: `aquillm/lib/parsers/structured/xml_parser.py`
- Modify: `aquillm/lib/parsers/structured/yaml_parser.py`
- Modify: `aquillm/lib/parsers/structured/__init__.py`
- Modify: `aquillm/lib/parsers/media/vtt.py`
- Modify: `aquillm/lib/parsers/media/__init__.py`
- Modify: `aquillm/lib/parsers/__init__.py`
- Modify: `aquillm/aquillm/ingestion/parsers.py`
- Modify: `aquillm/apps/ingestion/tests/test_streaming_parsers.py`
- Create: `aquillm/apps/ingestion/tests/fixtures/streaming/simple_formats.json`

- [ ] **Step 1: Add golden fixtures proving small TXT/MD/CSV/TSV/JSONL/XML/VTT/SRT output and chunk boundaries are unchanged.**
- [ ] **Step 2: Add a 50 MiB synthetic CSV test that consumes only a prefix and proves the parser has not read the remainder. Task 19 later runs this production iterator to EOF under RSS measurement.**
- [ ] **Step 3: Implement `bounded_segments(text, max_utf8_bytes)` and require every streaming parser to yield through it. Set `csv.field_size_limit()` to the segment budget; a single oversized field raises `parser_segment_limit`.**
- [ ] **Step 4: Implement line/row iterators for TXT, MD, CSV, TSV, JSONL, VTT, and SRT.**
- [ ] **Step 5: Implement incremental XML text extraction with element clearing; reject or split oversized text nodes at the segment boundary.**
- [ ] **Step 6: Route JSON/YAML/HTML/RTF through the legacy adapter and enforce `legacy_parser_limit` before reading.**
- [ ] **Step 7: Run golden and streaming tests.**

Run: `rtk pytest aquillm/apps/ingestion/tests/test_streaming_parsers.py -q`
Expected: PASS with byte-for-byte golden output.

- [ ] **Step 8: Commit.**

### Task 10: Stream PDF and ZIP-based document formats

**Files:**
- Modify: `aquillm/lib/parsers/documents/pdf.py`
- Modify: `aquillm/lib/parsers/documents/docx.py`
- Modify: `aquillm/lib/parsers/documents/epub.py`
- Create: `aquillm/lib/parsers/documents/odt.py`
- Modify: `aquillm/lib/parsers/documents/__init__.py`
- Modify: `aquillm/lib/parsers/spreadsheets/xlsx.py`
- Modify: `aquillm/lib/parsers/spreadsheets/ods.py`
- Modify: `aquillm/lib/parsers/spreadsheets/__init__.py`
- Modify: `aquillm/lib/parsers/presentations/pptx.py`
- Modify: `aquillm/lib/parsers/presentations/odp.py`
- Modify: `aquillm/lib/parsers/presentations/__init__.py`
- Modify: `aquillm/lib/parsers/__init__.py`
- Modify: `aquillm/aquillm/ingestion/parsers.py`
- Modify: `aquillm/apps/ingestion/tests/test_streaming_parsers.py`
- Create: `aquillm/apps/ingestion/tests/fixtures/streaming/container_formats.json`

- [ ] **Step 1: Add golden tests for PDF page separators, XLSX rows, DOCX paragraphs, PPTX slides, ODT/ODS/ODP XML, and EPUB spine order.**
- [ ] **Step 2: Run the focused tests and confirm eager parsers fail laziness/segment assertions.**

Run: `rtk pytest aquillm/apps/ingestion/tests/test_streaming_parsers.py -q -k "pdf or xlsx or docx or pptx or odt or ods or odp or epub"`
Expected: FAIL before iterator implementations.

- [ ] **Step 3: Change PDF extraction to accept a path/seekable file, yield one bounded page at a time, and reject/split oversized page text.**
- [ ] **Step 4: Keep XLSX `read_only=True`, yield bounded rows without collecting `lines`, and reject oversized cells.**
- [ ] **Step 5: For DOCX/PPTX/ODT/ODS/ODP/EPUB, iterate ZIP members and XML/HTML elements with shared expansion accounting; clear processed elements and route emitted text through `bounded_segments()`.**
- [ ] **Step 6: Keep XLS and binary DOC/PPT on the bounded legacy adapter.**
- [ ] **Step 7: Run all format tests.**

Run: `rtk pytest aquillm/apps/ingestion/tests/test_streaming_parsers.py aquillm/apps/ingestion/tests/test_unified_ingestion_parsers.py -q`
Expected: PASS with canonical fixture output unchanged.

- [ ] **Step 8: Commit.**

### Task 11: Stream nested archives under shared budgets

**Files:**
- Modify: `aquillm/aquillm/ingestion/parser_archive.py`
- Create: `aquillm/apps/ingestion/tests/test_streaming_archive.py`

- [ ] **Step 1: Write adversarial ZIP tests for lying sizes, excessive ratio, nested shared totals, entry/depth/temp-disk boundaries, and member cleanup.**
- [ ] **Step 2: Run the tests and confirm the eager archive collector fails.**

Run: `rtk pytest aquillm/apps/ingestion/tests/test_streaming_archive.py -q`
Expected: FAIL on eager reads/shared-budget assertions.

- [ ] **Step 3: Replace archive payload list extension with recursive `yield from` over bounded temporary member files.**
- [ ] **Step 4: Account actual bytes on every decompression read; never rely only on `ZipInfo.file_size`.**
- [ ] **Step 5: Run archive tests and commit.**

Run: `rtk pytest aquillm/apps/ingestion/tests/test_streaming_archive.py -q`
Expected: PASS.

### Task 12: Stream and budget figure extraction

**Files:**
- Modify: `aquillm/aquillm/ingestion/parser_figures.py`
- Modify: `aquillm/aquillm/ingestion/figure_extraction/__init__.py`
- Modify: `aquillm/aquillm/ingestion/figure_extraction/pdf.py`
- Modify: `aquillm/aquillm/ingestion/figure_extraction/pdf_page_extractors.py`
- Modify: `aquillm/aquillm/ingestion/figure_extraction/pdf_geometry.py`
- Modify: `aquillm/aquillm/ingestion/figure_extraction/office.py`
- Modify: `aquillm/aquillm/ingestion/figure_extraction/office_convert.py`
- Modify: `aquillm/aquillm/ingestion/figure_extraction/spreadsheet.py`
- Modify: `aquillm/aquillm/ingestion/figure_extraction/ebook.py`
- Modify: `aquillm/apps/ingestion/tests/test_figure_extraction.py`
- Modify: `aquillm/apps/ingestion/tests/test_streaming_parsers.py`

- [ ] **Step 1: Add failing tests for one-at-a-time consumption, every figure limit, Pillow bomb errors, Office cutoff, warning metadata, and handle cleanup.**
- [ ] **Step 2: Run focused tests and confirm the list-appending path fails.**

Run: `rtk pytest aquillm/apps/ingestion/tests/test_figure_extraction.py aquillm/apps/ingestion/tests/test_streaming_parsers.py -q -k figure`
Expected: FAIL before iterator/budget integration.

- [ ] **Step 3: Make extractors yield one figure, inspect its header, charge budgets, OCR/save it through the consumer, and release it before advancing.**
- [ ] **Step 4: Keep Office-to-PDF conversion on disk, enforce its source cutoff before launch, and convert optional conversion/OCR failures to structured warnings.**
- [ ] **Step 5: Run figure tests and commit.**

Run: `rtk pytest aquillm/apps/ingestion/tests/test_figure_extraction.py aquillm/apps/ingestion/tests/test_streaming_parsers.py -q -k figure`
Expected: PASS.

---

## Chunk 3: Idempotent Persistence and Durable Dispatch

### Task 13: Add fenced ingestion lease schema and shared transitions

**Files:**
- Modify: `aquillm/apps/ingestion/models/batch.py`
- Create: `aquillm/apps/ingestion/migrations/0003_ingestion_queue_leases.py`
- Create: `aquillm/apps/ingestion/services/queue_classification.py`
- Create: `aquillm/apps/ingestion/services/leases.py`
- Create: `aquillm/apps/ingestion/tests/test_dispatcher.py`

- [ ] **Step 1: Write failing tests for deterministic queue classification and every permitted/forbidden state-and-token transition: claim, start, heartbeat, requeue, success, terminal error, expired lease, stale token, backoff, and maximum attempts.**
- [ ] **Step 2: Add `DISPATCHED` plus `dispatch_mode=legacy|managed` and the exact lease/retry/error fields. Add fields nullable first, mark all existing rows legacy, and never lease or redispatch existing processing rows. New rows remain legacy while the dispatcher flag is off.**
- [ ] **Step 3: Implement `LeaseGuard` and a single compare-and-swap transition API used later by ingestion items and index jobs. Every mutation checks expected state plus token; every new attempt receives a fresh token.**
- [ ] **Step 4: Test migration compatibility and prove legacy rows preserve current processing behavior while managed rows obey fencing.**
- [ ] **Step 5: Run focused tests and commit.**

Run: `rtk pytest aquillm/apps/ingestion/tests/test_dispatcher.py -q`
Expected: PASS for schema compatibility, transition fencing, retry bounds, and classification.

### Task 14: Add deterministic ingestion artifact ownership

**Files:**
- Create: `aquillm/apps/ingestion/models/artifact.py`
- Modify: `aquillm/apps/ingestion/models/__init__.py`
- Create: `aquillm/apps/ingestion/migrations/0004_ingestion_artifacts.py`
- Create: `aquillm/aquillm/task_ingest_persistence.py`
- Create: `aquillm/apps/ingestion/tests/test_ingest_idempotency.py`

- [ ] **Step 1: Write failing tests for duplicate delivery, crash after media write, checksum mismatch, completed artifact reuse, parent-key resolution, stale-writing cleanup, and concrete `_meta.label` construction for PDF/raw-text/image/media/figure documents.**
- [ ] **Step 2: Implement `IngestionArtifact` with unique `(ingestion_item, payload_key)` and explicit `writing|complete|error` state.**
- [ ] **Step 3: Implement deterministic object keys and checksum/length validation before reuse.**
- [ ] **Step 4: Implement the cross-system sequence without holding a database lock during object-store I/O: commit/get the `writing` artifact; fence the item lease; write/reuse the deterministic object; fence again; then use one short `transaction.atomic()` block to lock the artifact, create/rehydrate the concrete model via `apps.get_model(label)`, create the associated index job, and mark the artifact complete.**
- [ ] **Step 5: Implement reconciliation for expired writing artifacts and deterministic media objects. A checksum-matching object is reusable; mismatches are deleted before retry.**
- [ ] **Step 6: Run migration/idempotency tests and commit.**

Run: `rtk pytest aquillm/apps/ingestion/tests/test_ingest_idempotency.py -q`
Expected: PASS for all concrete document subtypes and crash boundaries.

### Task 15: Convert the streaming worker and add bounded ingestion dispatch

**Files:**
- Modify: `aquillm/aquillm/task_ingest_uploaded.py`
- Create: `aquillm/apps/ingestion/services/dispatcher.py`
- Modify: `aquillm/apps/ingestion/services/upload_batches.py`
- Modify: `aquillm/aquillm/tasks.py`
- Modify: `aquillm/apps/ingestion/tests/test_dispatcher.py`
- Modify: `aquillm/apps/ingestion/tests/test_multimodal_ingestion_media_storage.py`
- Modify: `aquillm/apps/ingestion/tests/test_ingest_idempotency.py`

- [ ] **Step 1: Write a failing task test whose source raises on unrestricted reads and whose payload iterator asserts one-at-a-time consumption. Add dispatcher tests for bounded claims, simultaneous admission, per-user/global/queue capacity, publication failure, heartbeat, expiry, and disabled-mode compatibility.**
- [ ] **Step 2: Wrap the task in `StoredIngestionSource`, build limits once, and consume `iter_extracted_payloads()` sequentially. Spool one payload, persist it, close/release it, then request the next.**
- [ ] **Step 3: Call `LeaseGuard.assert_current()` before advancing each payload, before/after text spooling, before/after every media write, and before the short completion transaction. A failed fence stops all further writes.**
- [ ] **Step 4: Stream source PDF/media copies through Django `File`; remove all `ContentFile(data)` use from this task. Record stable terminal errors and warnings separately; re-raise transient failures for retry ownership.**
- [ ] **Step 5: Preserve behavior while disabled: legacy admission publishes immediately. When enabled, serialize capacity with PostgreSQL advisory transaction locks acquired global-then-user-then-queue, and count/create/claim managed work in that transaction.**
- [ ] **Step 6: Claim with `select_for_update(skip_locked)`, assign fresh tokens without exceeding locked capacity, and publish only claimed rows. Publication failures remain recoverable.**
- [ ] **Step 7: Make deterministic errors terminal and transient errors requeue with bounded exponential backoff through the shared transition API.**
- [ ] **Step 8: Run task and dispatcher suites and commit.**

Run: `rtk pytest aquillm/apps/ingestion/tests/test_dispatcher.py aquillm/apps/ingestion/tests/test_ingest_idempotency.py aquillm/apps/ingestion/tests/test_multimodal_ingestion_media_storage.py -q`
Expected: PASS, including concurrent admission, disabled-dispatcher compatibility, crash boundaries, and no unrestricted reads.

### Task 16: Add durable index-job dispatch and inactive-generation cleanup

**Files:**
- Modify: `aquillm/apps/documents/services/index_jobs.py`
- Modify: `aquillm/aquillm/tasks.py`
- Modify: `aquillm/aquillm/celery.py`
- Modify: `aquillm/apps/documents/tests/test_index_dispatcher.py`

- [ ] **Step 1: Add tests for broker failure, duplicate dispatch, stale generation, heartbeats, expired jobs, and cleanup that preserves the active generation.**
- [ ] **Step 2: Implement periodic dispatch/recovery for `DocumentIndexJob` using the shared transition/lease API from Task 15.**
- [ ] **Step 3: Route index jobs to `ingest.embed` and configure late acknowledgement plus reject-on-worker-loss for ingestion/embed tasks.**
- [ ] **Step 4: Implement cleanup that deletes only inactive generations not referenced by a live job. When a later document version reuses a hash whose prior generation was cleaned, Task 4's `ensure_index_job()` resets and rebuilds the existing unique job rather than treating the cleaned success as reusable.**
- [ ] **Step 5: Add flag-transition tests: outbox disabled publishes directly; outbox enabled leaves publication to the live embed dispatcher; toggling cannot strand pending jobs.**
- [ ] **Step 6: Run index dispatcher tests.**

Run: `rtk pytest aquillm/apps/documents/tests/test_index_dispatcher.py -q`
Expected: PASS.

- [ ] **Step 7: Commit.**

### Task 17: Configure queue workers, admission backpressure, and status API

**Files:**
- Modify: `deploy/compose/base.yml`
- Modify: `deploy/compose/development.yml`
- Modify: `deploy/compose/production.yml`
- Modify: `aquillm/aquillm/settings.py`
- Modify: `.env.example`
- Modify: `aquillm/apps/ingestion/views/api/uploads.py`
- Modify: `aquillm/apps/ingestion/tests/test_unified_ingestion_api.py`

- [ ] **Step 1: Add API tests for `429` queue saturation and additive queue/error metadata.**
- [ ] **Step 2: Define `ingest.text`, `ingest.ocr`, `ingest.transcribe`, and `ingest.embed` workers with prefetch 1 and explicit env-driven concurrency.**
- [ ] **Step 3: Set conservative defaults: text 2, OCR 1, transcription 1, embed 1; add max-tasks/max-memory child recycling as defense in depth.**
- [ ] **Step 4: Add dispatcher schedules and feature flags with both disabled by default; document that disabled flags preserve immediate legacy publishing.**
- [ ] **Step 5: Validate Compose and run API tests.**

Run: `docker compose -f deploy/compose/base.yml -f deploy/compose/production.yml config`
Expected: valid queue-specific worker definitions.

Run: `rtk pytest aquillm/apps/ingestion/tests/test_unified_ingestion_api.py -q`
Expected: PASS.

- [ ] **Step 6: Commit.**

---

## Chunk 4: Memory Gates, Observability, and Rollout

### Task 18: Add phase-level memory and resource telemetry

**Files:**
- Modify: `aquillm/aquillm/task_ingest_uploaded.py`
- Modify: `aquillm/apps/documents/tasks/chunking.py`
- Modify: `aquillm/apps/ingestion/services/dispatcher.py`
- Modify: `aquillm/apps/documents/services/index_jobs.py`
- Create: `aquillm/apps/ingestion/tests/test_ingestion_observability.py`

- [ ] **Step 1: Add failing structured-log tests that assert counts/bytes and prove source text/media never appear.**
- [ ] **Step 2: Run the focused test and confirm required fields are absent.**

Run: `rtk pytest aquillm/apps/ingestion/tests/test_ingestion_observability.py -q`
Expected: FAIL before telemetry fields exist.

- [ ] **Step 3: Emit source/extracted bytes, segments, figures, chunks, batch maxima, elapsed time, queue wait, retries, and lease recoveries.**
- [ ] **Step 4: Gate RSS sampling behind `INGEST_MEMORY_METRICS_ENABLED`; use `resource` on Linux and omit unsupported platforms cleanly.**
- [ ] **Step 5: Run tests and logging checks.**

Run: `rtk pytest aquillm/apps/ingestion/tests/test_ingestion_observability.py -q && rtk python scripts/check_logging_conventions.py`
Expected: PASS.

- [ ] **Step 6: Commit.**

```bash
git add aquillm/aquillm/task_ingest_uploaded.py aquillm/apps/documents/tasks/chunking.py aquillm/apps/ingestion/services/dispatcher.py aquillm/apps/documents/services/index_jobs.py aquillm/apps/ingestion/tests/test_ingestion_observability.py
git commit -m "feat(ingestion): report bounded pipeline resource metrics"
```

### Task 19: Add reproducible peak-memory benchmark gates

**Files:**
- Create: `scripts/benchmark_ingestion_memory.py`
- Create: `tests/unit/test_ingestion_memory_benchmark_harness.py`
- Modify: `aquillm/apps/ingestion/tests/test_streaming_parsers.py`
- Modify: `aquillm/apps/documents/tests/test_chunk_batches.py`

- [ ] **Step 1: Write harness tests for parent fixture generation, child execution, malformed child output, threshold failure, and this JSON schema.**

```json
{"case":"csv","status":"pass","baseline_rss_mib":0.0,"peak_rss_mib":0.0,"delta_rss_mib":0.0,"quartile_rss_mib":[0.0,0.0,0.0,0.0],"provider_batch_max":0,"insert_batch_max":0,"input_bytes":0,"output_bytes":0}
```

- [ ] **Step 2: Implement parent mode so it streams fixtures to a temporary directory before starting the measured child; fixture-generation memory is excluded. Parent accepts only the child's final JSON line and exit code.**
- [ ] **Step 3: Implement Linux child mode: warm production imports, record baseline current RSS from `/proc/self/status`, run the production parser or `run_index_generation()` to completion, sample current RSS at 25/50/75/100 percent, and record peak RSS with `resource.getrusage`.**
- [ ] **Step 4: For embed, inject bounded fake provider/insert callables into the same production generation service used by Celery. Create fresh vectors per batch, assert provider max <=64 and insert max <=256, and require 25%-to-100% current-RSS growth <64 MiB so linear retention fails.**
- [ ] **Step 5: Add CSV-to-EOF, figure-rich PDF, and nested-ZIP cases. Parent writes inputs; child reads them through production `StoredIngestionSource` and parser entry points.**
- [ ] **Step 6: Add default caps of 512 MiB parser peak and 384 MiB embed peak. Unsupported non-Linux child mode exits 2 rather than passing.**
- [ ] **Step 7: Run harness unit tests.**

Run: `rtk pytest tests/unit/test_ingestion_memory_benchmark_harness.py -q`
Expected: PASS.

- [ ] **Step 8: Run benchmarks in the production worker image.**

Run: `docker compose -f deploy/compose/base.yml -f deploy/compose/production.yml run --rm worker python /app/scripts/benchmark_ingestion_memory.py --case all`
Expected: all cases return `"status":"pass"`; CSV peak <512 MiB, embed peak <384 MiB, and retention growth <64 MiB.

- [ ] **Step 9: Commit.**

### Task 20: Document rollout, recovery, and rollback

**Files:**
- Create: `docs/documents/operations/large-document-ingestion.md`
- Modify: `docs/roadmap/plans/pending/2026-03-26-ingestion-work-queue-batching-implementation.md`
- Modify: `docs/specs/README.md`

- [ ] **Step 1: Document every limit/queue setting, stable error code, dashboard field, and safe starting value.**
- [ ] **Step 2: Document rollout order: nullable generation migration/backfill, generation search, embed worker/outbox, streaming flag, per-format staging, ingestion dispatcher.**
- [ ] **Step 3: Document forced-worker-kill verification, stale-artifact reconciliation, inactive-generation cleanup, compatibility flags, and rollback.**
- [ ] **Step 4: Mark the 2026-03-26 implementation plan superseded without deleting history.**
- [ ] **Step 5: Commit.**

### Task 21: Add and run reproducible staging fault injection

**Files:**
- Create: `scripts/verify_ingestion_resilience.py`
- Create: `tests/unit/test_ingestion_resilience_harness.py`
- Create: `aquillm/apps/ingestion/management/__init__.py`
- Create: `aquillm/apps/ingestion/management/commands/__init__.py`
- Create: `aquillm/apps/ingestion/management/commands/ingestion_diagnostics.py`
- Create: `aquillm/apps/ingestion/tests/test_ingestion_diagnostics.py`
- Modify: `docs/documents/operations/large-document-ingestion.md`

- [ ] **Step 1: Write harness tests for fixture manifests, polling timeouts, queue-cap assertions, worker-kill commands, rollback restoration, redaction, and evidence JSON.**
- [ ] **Step 2: Implement `ingestion_diagnostics --batch-id <integer> --json` and unit tests using `call_command`. The harness obtains the integer batch primary key from the upload response and passes it unchanged. Return managed active counts by queue, item status/attempt counts, artifact payload keys/checksums, index-job status/attempt/generation, and active-generation counts per document. Never return lease tokens, credentials, or source/extracted content.**
- [ ] **Step 3: Implement cases `burst`, `kill-parse`, `kill-embed`, and `rollback`. Stream a generated 50-file TXT/CSV/PDF/XLSX/ZIP manifest, authenticate via CLI options, upload, poll item/job states, and query the diagnostic command for bounded active counts and uniqueness assertions.**
- [ ] **Step 4: In kill cases, target Compose services `worker_ingest_text` and `worker_embed`, wait for a processing lease, kill, wait past expiry, restart, and assert success, attempt increment, one artifact per key, and one active generation.**
- [ ] **Step 5: Require `--staging-env-path <path>` for rollback. Resolve it to an existing regular file, copy its exact bytes to a sibling temporary backup, modify only that resolved file to disable streaming/dispatcher flags, recreate workers, prove legacy immediate publishing plus active-generation search, and restore the original bytes in `finally`. Never infer or default to a repository `.env`.**
- [ ] **Step 6: Emit `{run_id,started_at,git_sha,cases:[{name,status,duration_sec,assertions}],metrics:{peak_rss_mib,queue_wait_p95_sec,success_rate,retry_count}}`; exclude credentials/content.**
- [ ] **Step 7: Run diagnostic, focused backend, migration, and hygiene checks.**

Run: `rtk pytest aquillm/apps/ingestion/tests/test_ingestion_diagnostics.py -q`
Expected: PASS and diagnostic JSON contains only the documented non-sensitive fields.

Run: `rtk pytest aquillm/apps/ingestion/tests aquillm/apps/documents/tests aquillm/lib/parsers/tests aquillm/lib/embeddings/tests -q`
Expected: PASS.

Run: `cd aquillm && rtk python manage.py makemigrations --check && rtk python manage.py check`
Expected: PASS with no pending migrations.

Run: `rtk powershell -File scripts/check_hygiene.ps1`
Expected: PASS.

- [ ] **Step 8: Run harness unit tests and staging cases.**

Run: `rtk pytest tests/unit/test_ingestion_resilience_harness.py -q`
Expected: PASS.

Run: `python scripts/verify_ingestion_resilience.py --compose deploy/compose/production.yml --staging-env-path "$STAGING_ENV_PATH" --base-url "$STAGING_BASE_URL" --username "$STAGING_USER" --password-env STAGING_PASSWORD --cases burst,kill-parse,kill-embed,rollback --json-out ingestion-resilience-evidence.json`
Expected: exit 0 and every case has `"status":"pass"`.

- [ ] **Step 9: Run the Task 19 memory gate.**
- [ ] **Step 10: Copy the evidence summary into the operations document and commit.**

```bash
git add scripts/verify_ingestion_resilience.py tests/unit/test_ingestion_resilience_harness.py aquillm/apps/ingestion/management aquillm/apps/ingestion/tests/test_ingestion_diagnostics.py docs/documents/operations/large-document-ingestion.md
git commit -m "test(ingestion): verify bounded pipeline resilience"
```

---

## Delivery Gates

Do not enable `INGEST_STREAMING_PIPELINE_ENABLED` until Tasks 2–16 and the active-generation backfill are deployed. Do not enable the ingestion dispatcher until queue-specific workers are running and the forced-worker-kill test passes. Do not raise the 64 MiB extracted-text limit until the benchmark demonstrates the configured worker memory headroom at the proposed value.
