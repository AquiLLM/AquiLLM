# RAG Source and Index Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair F01, F02, F15, F16 and F17 by preserving source meaning and member artifacts, tracking every ingestion output, and serving only explicitly compatible active index generations.

**Architecture:** Immutable source artifacts and ordered extraction revisions feed staged representation generations; representation activation publishes citable chunks and graph invalidation together while retaining old chunk identities. A separate embedding generation stores response-bound vectors beside those chunks; switching a dense profile never changes source text or graph identity. Unknown or mismatched dense identity produces explicit lexical-only retrieval of the active representation.

**Tech Stack:** Python >=3.12, existing Django/Celery/PostgreSQL 16/pgvector, python-docx, python-pptx, openpyxl, xlrd, odfpy and current embedding clients. No new model service or parser dependency is required.

**Spec:** [Shared remediation design](../specs/2026-10-10-rag-remediation-design.md). The [program plan](2026-10-10-rag-remediation-program.md) owns execution, migration allocation and cross-lane acceptance. Evidence is the [public code audit](../../audits/2026-10-10-rag-adversarial/REPORT.md) at `development` commit `6e756e252999ee6ccf604f88ea5326f49ddaa6d4`.

## Global Constraints

- Planning only: this document does not execute migrations, change application code, run a reindex, load credentials or contact a server.
- Every shell command starts with `rtk`. Test commands below run only in the coordinator's P0 disposable CI-compatible PostgreSQL 16/pgvector/full-app environment, using the existing `pytest.ini`, fake provider keys and disabled external network. Do not source a real `.env`.
- Preserve the completed finite-vector/count/index validators, paired vector/receipt writes, redacted logs, 30-second I/O ceilings, zero SDK retries, typed transient classification, bounded Celery retries and exact publication generation/attempt failure fences. Read `2026-10-09-embedding-contract.md` and `2026-10-09-embedding-outage-recovery.md` before editing those boundaries.
- A 1024D vector or a configuration model name does not establish compatible identity. Historical unknown receipts remain unknown. A receipt's observed revision remains null when the response does not provide revision evidence.
- Preserve exact concrete document identity, authorization revalidation, lifecycle lock ordering, source revision checks, bounded source materialization and final-publication budgets. File hashes are not permission tokens.
- Existing `TextChunk` primary keys, content and offsets are immutable once published; retired chunks remain resolvable through current document permission checks. Deleting the document still deletes its retained generations and chunks.
- At most 3 acquisition actions and 45 unique sources per turn.
- At most 250,000 materialized and 1,000,000 tokenized codepoints.
- At most 90 acquisition pairs plus 45 reserved final pairs; 6 in-flight pairs.
- At most 2 planner calls, each limited to 2 seconds and 512 output tokens.
- A 15-second retrieval deadline with a 3-second final scoring allowance inside that deadline.
- Pair context remains 1024; preserve current models, GPU allocations and provider context limits during controlled comparisons.
- Preserve existing graph branch and overall budgets. Bounded frontier work consumes those budgets; it gets no new per-generation allowance.
- No authorization, citation, source-integrity or compatibility violation is acceptable in deterministic gates.
- The old 2026-09-29 identity plan is background only. This plan replaces its source-experiment-only boundary with reviewed representation publication and does not copy private deployment observations from that plan.
- P0 is the sole allocator of migrations across I/R/G lanes. Semantic migration draft paths listed below describe exact changes; P0 assigns numbered filenames and dependencies on the pinned integration commit before registration or execution. Never renumber an applied migration.

## Ownership and sequencing

| Task | Independently reviewable deliverable | Dependencies |
|---|---|---|
| I1 | Immutable upload/member artifacts, provenance contracts and correct archived PDF storage | P0 contract freeze and migration allocation |
| I2 | Ordered native structures, faithful canonical text and working Office fallback | I1 types; may run alongside I3/I4 |
| I3 | Durable output manifest, idempotent fan-out, partial-result API and recovery | I1; uses I2 output contract with fixture extraction until I2 lands |
| I4 | Verified dense adapter and atomic vector-only generation publication/read seam | I1 frozen representation types; independent of native parsers |
| I5 | Staged structural chunks, atomic representation activation and stable historical citations | I2/I3/I4, G graph activation seam, P0 schema gate |
| I6 | Populated-DB migration, failure, cutover and rollback evidence | I1-I5, R2 dense/lexical integration, G representation binding |

I owns ingestion, parser, document model and generation services described below. R2 exclusively owns `chunk_search_candidates.py`, `chunk_search.py` and query-cache integration. G exclusively owns graph invalidation/build/projection changes. I5 owns only the source-resolution changes in `apps/documents/views/api.py` and `views/pages.py`; the coordinator serializes any other edits to those files. Execute independent tasks in isolated worktrees with a fresh implementer and separate specification/correctness reviews. No two implementers edit the same path concurrently.

## Contracts frozen before parallel implementation

I1 creates `aquillm/lib/parsers/source_contracts.py`. Frozen values use tuples, not mutable nested dictionaries:

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class SourceArtifactRefV1:
    sha256: str
    size_bytes: int
    storage_key: str
    member_path: str | None
    container_sha256: str | None

@dataclass(frozen=True)
class CellV1:
    row: int
    column: int
    address: str
    text: str | None
    formula: str | None
    value_status: Literal['literal', 'cached', 'uncached', 'blank']
    row_span: int = 1
    column_span: int = 1
    header_keys: tuple[str, ...] = ()

@dataclass(frozen=True)
class SourceElementV1:
    element_key: str
    kind: Literal['heading', 'paragraph', 'table', 'formula', 'figure', 'separator']
    start: int
    end: int
    page: int | None
    sheet: str | None
    slide: int | None
    parent_key: str | None
    cells: tuple[CellV1, ...] = ()
    bbox: tuple[float, float, float, float] | None = None

@dataclass(frozen=True)
class ExtractionV1:
    parser_key: str
    artifact: SourceArtifactRefV1
    canonical_text: str
    elements: tuple[SourceElementV1, ...]
    quality: Literal['complete', 'partial', 'unsupported']
    issue_codes: tuple[str, ...]

@dataclass(frozen=True)
class SourceChunkV1:
    start: int
    end: int
    source_text: str
    embedding_text: str
    element_keys: tuple[str, ...]
    source_sha256: str
    input_sha256: str
    modality: Literal['text', 'image', 'caption_only'] = 'text'
    media_artifact: SourceArtifactRefV1 | None = None
```

`start/end` are Unicode code-point offsets into exact canonical text. Primary elements partition that text; figure overlays have separate location metadata. Blank cells, merged spans and unknown coordinates remain explicit; no coordinate is inferred from concatenation. Source SHA-256 uses exact canonical UTF-8 bytes; media SHA-256 uses exact immutable bytes. `SourceChunkV1.input_sha256` is the representation projection digest `input_digest(embedding_text)`. The actual profile-prepared wire-input digest is separately stored in the embedding generation and response receipt using the existing `lib.embeddings.provenance.input_digest` canonical-JSON convention; it can differ after template/media preparation. Never compare either input digest to a raw-text SHA-256. I2 emits text chunks; I5 constructs image/caption-only chunks from validated media artifacts. I4 validates the supplied projection digest, applies verified profile preparation, and binds the complete exact wire-input digest to the result. Element/source locations are provenance, not authorization.

I1 also creates `aquillm/apps/documents/services/representation_contracts.py`:

```python
from dataclasses import dataclass
from uuid import UUID
from lib.parsers.source_contracts import SourceArtifactRefV1

class SourceScopeUnavailable(ValueError):
    """Exact source identity cannot be proven for the captured scope."""

@dataclass(frozen=True)
class DocumentRefV1:
    model_label: str
    pkid: int
    id: UUID
    collection_id: int
    source_hash: str

@dataclass(frozen=True)
class RepresentationSelectionV1:
    document: DocumentRefV1
    generation_key: UUID | None
    canonical_sha256: str
    source_revision_sha256: str | None

@dataclass(frozen=True)
class RepresentationSnapshotV1:
    selections: tuple[RepresentationSelectionV1, ...]

@dataclass(frozen=True)
class BuildLeaseV1:
    generation_key: UUID
    lease_token: UUID

@dataclass(frozen=True)
class CitationSourceV1:
    document: DocumentRefV1
    representation_key: UUID | None
    canonical_text: str
    start: int
    end: int
    artifact: SourceArtifactRefV1 | None
```

I4 first implements `freeze_active_representations` and `active_chunk_scope` in `services/representation_store.py` for an exact legacy-null snapshot; I5 extends the same file serially to read real active representation pointers and implement publication. This allows R2 to land the F02 dense correctness release immediately after I4, before structural reingestion or I5 activation. Exact APIs:

- `freeze_active_representations(documents: tuple[DocumentRefV1, ...], *, using: str) -> RepresentationSnapshotV1`. For legacy-null chunks it enumerates concrete owners across all supported document models and raises `SourceScopeUnavailable('source_scope_unavailable')` on absent/ambiguous ownership, including an unauthorized second owner with the same UUID/hash. I1 declares this marker in `representation_contracts.py`; I4 implements the guard. It carries no model, title, foreign PK or owner-count detail. R2 returns no evidence for that invalid scope instead of reusing an unfrozen document list.
- `active_chunk_scope(queryset, *, snapshot: RepresentationSnapshotV1)` returns a lazy queryset constrained by the disjunction of exact document UUID plus captured representation key. A null key matches only legacy-null chunks; it is never a wildcard. The caller keeps existing authorization and statement-timeout machinery.
- `begin_representation(document: DocumentRefV1, *, extraction_revision_id: int, chunker_key: str, manifest_sha256: str, expected_count: int, using: str) -> BuildLeaseV1`.
- `stage_representation(lease: BuildLeaseV1, *, chunks: tuple[SourceChunkV1, ...], using: str) -> None`.
- `activate_representation(lease: BuildLeaseV1, *, expected_active_key: UUID | None, using: str) -> Literal['activated', 'already_active', 'stale']`.
- `load_chunk_source(chunk_id: int, *, using: str) -> CitationSourceV1` supplies the immutable source snapshot; the view must authorize its exact current document before returning text/media. A deleted/moved/revoked document never acquires historical permission from the generation.

I4 creates `aquillm/lib/embeddings/document_contracts.py`:

```python
from dataclasses import dataclass, field
from time import monotonic
from typing import Callable, Literal
from uuid import UUID

@dataclass(frozen=True)
class VerifiedEmbeddingSpaceV1:
    space_key: str
    provider: str
    model: str
    verified_revision: str
    tokenizer_revision: str
    dimensions: int
    query_prep_key: str
    document_prep_key: str
    dimension_policy: str
    normalization: str
    compute_precision: Literal['fp32', 'fp16', 'bf16']
    quantization_policy: str
    pooling_policy: str
    template_policy_sha256: str
    engine_signature_sha256: str
    max_input_tokens: int
    max_batch_tokens: int
    permitted_modalities: tuple[Literal['text', 'image', 'caption_only'], ...]

@dataclass(frozen=True)
class EmbeddingBuildBudgetV1:
    deadline_at: float
    max_items: int = 32
    max_input_bytes: int = 262144
    clock: Callable[[], float] = field(default=monotonic, repr=False, compare=False)

    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline_at - self.clock())

@dataclass(frozen=True)
class QueryEmbeddingObservationV1:
    vector: tuple[float, ...]
    receipt: dict
    prepared_input_sha256: str
    compatibility_key: str | None

@dataclass(frozen=True)
class EmbeddingLeaseV1:
    generation_key: UUID
    lease_token: UUID

@dataclass(frozen=True)
class RankedChunkV1:
    chunk_id: int
    distance: float
    embedding_generation_key: UUID
    representation_key: UUID | None
    source_md5: str

@dataclass(frozen=True)
class DenseGenerationSelectionV1:
    document_id: UUID
    model_label: str
    document_pkid: int
    representation_key: UUID | None
    source_hash: str
    generation_key: UUID
    manifest_sha256: str

@dataclass(frozen=True)
class DenseGenerationSnapshotV1:
    space_key: str
    generation_keys: tuple[UUID, ...]
    manifest_sha256: str
    selections: tuple[DenseGenerationSelectionV1, ...]
```

`receipt` is a detached copy of the real `EmbeddingResult.provenance`, validated against the stored float32 vector. The verified serving binding is separate from observed response fields; declarations never fill missing observed fields. A profile is usable only after its immutable model/tokenizer/preparation/endpoint artifact binding is verified and recorded. Missing or conflicting binding yields `compatibility_key=None` or a typed contract failure; it cannot manufacture a known revision. Strict output dimension is 1024 and any vendor-supported reduction/normalization is a versioned explicit profile policy. Existing permissive dimension fitting remains only on its existing legacy API.

`engine_signature_sha256` hashes a verified immutable canonical-JSON engine manifest with exact keys: weights_revision, tokenizer_revision, compute_precision, quantization_algorithm, quantization_revision, pooling_policy, query_template_sha256, document_template_sha256, image_processor_revision, dimension_policy, normalization, engine_binary_sha256, engine_config_sha256 and response_route. Unquantized profiles explicitly say `quantization_algorithm='none'` and `quantization_revision='none'`; absence is unknown, not none. `template_policy_sha256` hashes the exact query/document template pair. `space_key` hashes the complete verified profile plus engine signature; fp16, bf16 and any quantized variant cannot share it. Immutable endpoint verification evidence records which serving engine emitted the response without pretending that a model-name echo attests precision/revision. Null response evidence stays null.

Exact I4 APIs:

- `lib/embeddings/document_provider.prepare_query(query: str, *, space: VerifiedEmbeddingSpaceV1) -> tuple[str, str]` returns exact prepared text and its SHA-256; deterministic local preparation only.
- `document_provider.observe_query_embedding(query: str, *, space: VerifiedEmbeddingSpaceV1, budget: TurnBudget) -> QueryEmbeddingObservationV1` captures the actual response before deciding compatibility.
- `document_provider.embed_document_batch(chunks: tuple[SourceChunkV1, ...], *, space: VerifiedEmbeddingSpaceV1, budget: EmbeddingBuildBudgetV1) -> tuple[EmbeddingResult, ...]` validates exact response indices/count and preparation receipts, without fallback or prefix truncation.
- `services/embedding_index_store.begin_embedding_generation(selection: RepresentationSelectionV1, *, space_key: str, manifest_sha256: str, expected_count: int, using: str) -> EmbeddingLeaseV1`.
- `embedding_index_store.stage_embedding_batch(lease: EmbeddingLeaseV1, *, chunk_ids: tuple[int, ...], results: tuple[EmbeddingResult, ...], input_sha256: tuple[str, ...], using: str) -> None`.
- `embedding_index_store.activate_embedding_generation(lease: EmbeddingLeaseV1, *, using: str) -> Literal['activated', 'already_active', 'stale']`.
- `services/embedding_index_query.freeze_dense_generations(snapshot: RepresentationSnapshotV1, *, space_key: str, using: str) -> DenseGenerationSnapshotV1`.
- `services/embedding_index_query.rank_compatible_chunks(*, snapshot: RepresentationSnapshotV1, dense_snapshot: DenseGenerationSnapshotV1, query: QueryEmbeddingObservationV1, limit: int, using: str, budget: TurnBudget) -> tuple[RankedChunkV1, ...]`.

**R2 handoff:** freeze one representation snapshot, serving space and dense-generation snapshot per turn, use the same `TurnBudget` by identity, apply `active_chunk_scope` to lexical candidates and consume `rank_compatible_chunks` for dense candidates. Dense snapshots contain an exact per-document mapping; `generation_keys` must equal the unique mapped keys. Their canonical manifest digest includes the authorized document/representation selections, selected dense generation/manifests and explicit absent-generation entries. Rank never independently chooses a newer generation. Pointer changes after capture invalidate the affected dense result; missing entries stay lexical-only for this captured acquisition. Before cache/body publication recheck captured membership, ready/current pointer, source/profile, document authorization and request lease. Preserve source MD5/generation tags and rank order through body materialization. A null query compatibility key or absent ready matching generation produces zero dense candidates and a diagnostic such as `dense_identity_unknown`/`dense_generation_unavailable`; lexical retrieval continues. Cache keys include declared serving space, exact prepared-query hash, preparation key and dense-snapshot manifest digest, alongside representation/authorization/ranking and G's authority/algorithm digests. Cached observations must revalidate actual receipt and compatibility binding; cache population records actual response identity. Never pad the result pool with legacy unknown vectors. A dense-only swap changes the dense digest without rebuilding graphs.

Strict request ranking runs inside the existing `source_runtime_scope(SourceRuntime)` installed by R1/R2. `rank_compatible_chunks` requires `current_source_runtime().budget is budget`, revalidates that runtime's authorization and exact snapshot scope, and uses `bounded_source_database` for source DB work. Missing/mismatched runtime is a typed acquisition failure, not an unguarded query. Rebuild workers use explicit exact document/source ownership and build budgets instead of borrowing a request runtime.

`TurnBudget` means the existing `lib.retrieval.turn_budget.TurnBudget`. Its 15-second request ceiling remains intact. Async indexing uses `EmbeddingBuildBudgetV1` created locally from a validated finite positive relative `timeout_ms` capped at 120000: `deadline_at=monotonic()+timeout_ms/1000`. Celery serializes only the relative integer timeout and durable generation identity, never a monotonic deadline or clock callable. Every transport timeout is `min(30.0, budget.remaining_seconds())`; an expired budget fails before dispatch. R3 proposes `TextChunk.search_vector` plus source-kind/section metadata; I is sole editor of `models/chunks.py`. After I4, the coordinator serializes an I-owned model-field patch using `SearchVectorField(null=True)`, `source_kind=CharField(max_length=32, default='unknown', choices=SOURCE_KIND_CHOICES)`, and `section_path=JSONField(default=list)`. `SOURCE_KIND_CHOICES` defines unknown/body/heading/table/formula/figure/bibliography; only validated extractor/classifier tags can replace unknown. P0 allocates the FTS migration/index; R3 owns lexical services, indexing/backfill and source-kind classifier policy. Populate these fields per representation, retain old generation metadata, and never mutate canonical chunk content during FTS backfill.

**G handoff:** G5 owns `apps.knowledge_graph.projection.representation_lifecycle.prepare_document_representation_activation(reference: DocumentRefV1, collection_ids: tuple[int, ...], *, expected_source_hash: str, old_representation_key: UUID | None, new_representation_key: UUID, using: str) -> tuple[int, ...]` under the existing lifecycle lock. I5 converts the frozen reference to the existing DocumentLifecycleRef for its outer lock; G's bridge receives DocumentRefV1. It invalidates affected graph artifacts and inserts durable graph-build/refresh intent inside the publication transaction, without deleting any citable chunk rows. Graph build keys, handoffs and projection bundles capture the exact representation key; null legacy equality is exact. G changes graph chunk enumeration to `active_chunk_scope`. Dense-only activation does not call this seam.

### Task I1: Bind every payload to immutable source/member bytes

**Files:** Create `aquillm/lib/parsers/source_contracts.py`; `aquillm/apps/documents/services/representation_contracts.py`; `aquillm/apps/ingestion/models/source_artifact.py`; `aquillm/apps/ingestion/services/source_artifacts.py`; `aquillm/apps/ingestion/migrations/source_artifacts_v1.py` (semantic draft). Modify `aquillm/apps/ingestion/models/__init__.py`, `aquillm/aquillm/ingestion/types.py`, `aquillm/aquillm/ingestion/parser_archive.py`, `aquillm/aquillm/ingestion/parsers.py`, `aquillm/aquillm/task_ingest_uploaded.py`. Tests: create `aquillm/apps/ingestion/tests/test_source_artifacts.py`, `test_archive_source_binding.py`.

**Interfaces:** Produces the frozen contracts above; `persist_artifact(data: bytes, *, member_path: str | None, container_sha256: str | None, using: str) -> SourceArtifactRefV1`; `ExtractedTextPayload.source_artifact` and `.extraction` are optional during transition. PDF persistence requires its own artifact and never uses the outer upload as a fallback when `member_path` is present.

- [ ] Write an in-memory ZIP with `a/paper.pdf` and `b/paper.pdf`; use two distinct valid PdfWriter byte streams and fake extraction. Assert each stored PDF opens with PdfReader, hashes to its own member digest, and differs from the ZIP digest. Add repeated member paths distinguished by archive ordinal and digest, cross-format members, rejected traversal names, corrupted member bytes and two users sharing identical bytes with independent authorization.

```python
def test_member_payload_keeps_member_bytes(archive_payloads, pdf_bytes):
    payload = archive_payloads({'nested/paper.pdf': pdf_bytes})[0]
    assert payload.source_artifact.sha256 == sha256(pdf_bytes).hexdigest()
    assert payload.source_artifact.member_path == 'nested/paper.pdf'
    assert payload.source_artifact.container_sha256 != payload.source_artifact.sha256
```

- [ ] Run `rtk proxy python -m pytest aquillm/apps/ingestion/tests/test_source_artifacts.py aquillm/apps/ingestion/tests/test_archive_source_binding.py -q`; expect missing source binding failures before implementation.
- [ ] Store `SourceArtifact(sha256 unique, size_bytes, immutable_storage_key)`, `SourceArtifactOccurrence(artifact FK, container FK nullable, member_path nullable, archive_ordinal nullable, original_filename, content_type)`, and immutable `ExtractionRevision(occurrence FK, parser_key, canonical_text, canonical_sha256, ordered_elements JSON, quality, issue_codes JSON, manifest_sha256)`. Extraction revision uniqueness is occurrence plus parser key plus manifest, and an empty/unavailable extraction is explicit. I2 fills these rows through a pure validated ExtractionV1 serializer; I3 can bind its output ledger to the same already-defined revision model. Blob storage uses content-addressed keys and put-if-absent semantics; existing keys are digest/size verified, never overwritten. Unsupported conditional storage operations fail explicitly. DB registration follows successful storage verification; a crash can leave an unreferenced blob, never a falsely registered artifact. Authorization resolves through output/document binding, not a public digest URL.
- [ ] Pass each member's bytes/name through its payload. Keep existing expanded-byte/depth limits, count actual archive members rather than number of derived figures, and reject unsafe normalized member paths. PDFDocument saves the member artifact's filename/bytes. Top-level files receive the same artifact contract. Preserve OCR/transcription provider fields and extracted-media digests independently from their containing document digest.
- [ ] Run the new tests plus `rtk proxy python -m pytest aquillm/apps/ingestion/tests/test_unified_ingestion_parsers.py aquillm/apps/ingestion/tests/test_multimodal_ingestion_media_storage.py aquillm/apps/ingestion/tests/test_upload_batch_source_indices.py -q`; require all pass. Review storage-crash compensation and permission tests, then commit only I1 owned files with `rtk git commit -m "feat: bind ingestion outputs to immutable member artifacts"` after an explicit owned-path `rtk git add`.

### Task I2: Preserve ordered structures and repair Office fallback

**Files:** Create `aquillm/lib/parsers/structural.py`, `aquillm/lib/parsers/canonical_projection.py`, `aquillm/lib/parsers/source_chunking.py`, `aquillm/lib/parsers/tests/test_structural_extraction.py`, `aquillm/lib/parsers/tests/test_source_chunking.py`, `aquillm/apps/ingestion/tests/test_office_conversion_fallback.py`. Modify `aquillm/lib/parsers/documents/docx.py`, `aquillm/lib/parsers/documents/pdf.py`, `aquillm/lib/parsers/spreadsheets/xlsx.py`, `aquillm/lib/parsers/spreadsheets/xls.py`, `aquillm/lib/parsers/spreadsheets/ods.py`, `aquillm/lib/parsers/spreadsheets/csv_parser.py`, `aquillm/lib/parsers/presentations/pptx.py`, `aquillm/aquillm/ingestion/parsers.py`, `aquillm/aquillm/ingestion/figure_extraction/office.py`, `aquillm/aquillm/ingestion/figure_extraction/spreadsheet.py`, `aquillm/aquillm/ingestion/figure_extraction/office_convert.py`. Existing text-extraction functions remain wrappers returning the new faithful canonical projection.

**Interfaces:** `extract_structure(filename: str, data: bytes, *, artifact: SourceArtifactRefV1) -> ExtractionV1`; `chunk_extraction(extraction: ExtractionV1, *, chunker_key: str, tokenizer, max_tokens: int, overlap_tokens: int) -> tuple[SourceChunkV1, ...]`. Tokenizer is an injected verified local adapter; no remote tokenization. `convert_figures_pdf(data, source_format, filename) -> ConversionOutcomeV1(status, pdf_bytes)` is a non-generator result, with status `unavailable`, `failed` or `converted`.

- [ ] Reuse the public audit's wrong-column and DOCX fixtures as positive tests. Add DOCX paragraph/table/paragraph order and nested tables, PPTX native tables and grouped shapes, CSV quoted commas/blanks, XLS/ODS blank/repeated/merged cells, and XLSX paired formula/cache readers. The canonical table projection includes coordinates and labels: `sheet=Sheet row=2 A2 Product=Widget B2 Domestic=7 C2 Foreign=<blank>`. An uncached formula retains `B1 formula==50+20 value=<unknown>` and `quality='partial'`; no invented result of 70.

```python
def test_empty_columns_do_not_shift_values(structure_from_workbook):
    a = structure_from_workbook([['Product','Domestic','Foreign'], ['Widget',7,None]])
    b = structure_from_workbook([['Product','Domestic','Foreign'], ['Widget',None,7]])
    assert a.canonical_text != b.canonical_text
    assert a.elements[-1].cells[1].address == 'B2'
    assert b.elements[-1].cells[2].text == '7'
```

- [ ] Write fallback tests for all four Office formats: missing LibreOffice, conversion failure, converted PDF with no figures, converted PDF with figures and direct-image failure. Patch conversion and extraction; no subprocess/model call. Define policy explicitly: unavailable/failed conversion uses direct extraction; converted-empty also attempts direct extraction; converted-nonempty deduplicates direct images by digest instead of emitting duplicates.
- [ ] Run `rtk proxy python -m pytest aquillm/lib/parsers/tests/test_structural_extraction.py aquillm/lib/parsers/tests/test_source_chunking.py aquillm/apps/ingestion/tests/test_office_conversion_fallback.py -q`; expect current omission/collapse/fallback tests to fail.
- [ ] Implement ordered element traversal, the coordinate-preserving projection and complete/partial/unsupported quality classification. Read XLSX formulas and cached values separately, preserve unknown cache status, and close both workbooks. PDF retains page identity and records unavailable layout information as null. Legacy binary `.doc`/`.ppt`, ODT and RTF must use a supported local conversion/parser path or return `unsupported`; byte-decoded garbage is never complete extraction. Malformed structure is rejected with a stable issue code.
- [ ] Implement contiguous canonical-source chunks. Heading and repeated table-header context goes into `embedding_text`, leaving `source_text == canonical_text[start:end]`. Chunk table rows with explicit column schema, preserve formula qualifiers, and split oversized elements with lossless source offsets. Use actual tokenizer counts including embedding headers; reject an unrepresentable element rather than silently truncate. `chunker_key` fingerprints token budget, overlap, tokenizer revision and projection policy. Change configured defaults only after I6 evaluation.

```python
for chunk in chunks:
    assert chunk.source_text == extraction.canonical_text[chunk.start:chunk.end]
    assert len(tokenizer(chunk.embedding_text)) <= max_tokens
assert all(chunks[i].start <= chunks[i+1].start for i in range(len(chunks)-1))
```

- [ ] Run the new modules and existing unified-parser/figure tests. Require Unicode minus signs/units, native table facts, source offsets and embedded images preserved; uncached values remain visibly unknown. Commit explicit I2 paths with `rtk git commit -m "feat: preserve ordered source structures and Office fallback"`.

### Task I3: Publish durable idempotent output results and recover fan-out

**Files:** Create `aquillm/apps/ingestion/models/output.py`, `aquillm/apps/ingestion/services/output_ledger.py`, `aquillm/apps/ingestion/services/output_manifest.py`, `aquillm/apps/ingestion/tasks/__init__.py`, `aquillm/apps/ingestion/tasks/output_recovery.py`, `aquillm/apps/ingestion/tests/test_output_ledger.py`, `aquillm/apps/ingestion/tests/test_output_recovery.py`, `aquillm/apps/ingestion/tests/test_partial_upload_status.py`; create semantic `aquillm/apps/ingestion/migrations/ingestion_output_ledger_v1.py`. Modify `aquillm/apps/ingestion/models/batch.py`, `aquillm/apps/ingestion/models/__init__.py`, `aquillm/apps/ingestion/services/upload_batches.py`, `aquillm/apps/ingestion/views/api/uploads.py`, `aquillm/aquillm/task_ingest_uploaded.py`, `aquillm/aquillm/tasks.py`, and `aquillm/aquillm/celery_schedules.py` only after coordinator serializes schedule edits.

**Interfaces:** `prepare_output_manifest(item_id: int, *, using: str) -> tuple[str, ...]` persists immutable deterministic keys; `claim_output(output_key: str, *, using: str) -> OutputLeaseV1 | None`; `publish_output(lease: OutputLeaseV1, *, using: str) -> OutputResultV1`; `recover_due_outputs(*, limit: int, using: str) -> int`. Frozen `OutputLeaseV1(output_id:int, generation:UUID, attempt:int)` and `OutputResultV1(output_key:str,status:str,document_ref:DocumentRefV1|None,error_code:str|None)` live in `output_ledger.py`. Keys cover item, source occurrence, parser key, output kind and stable element/figure key; they never depend on mutable filenames alone.

- [ ] Add real PostgreSQL transaction tests: second output fails after first commits; crash after blob persistence, after document creation, after checkpoint commit and after broker acknowledgment; concurrent duplicate Celery delivery; repeated retry after success; stale worker failure after generation reset; collection deletion/move/revocation during processing; figure creation after parent retry. Successful output IDs remain visible immediately after checkpoint. No scenario duplicates a document or falsely relabels unrelated identical text as the same source.

```python
def test_second_failure_preserves_tracked_first_output(process_two_outputs):
    result = process_two_outputs(fail_second=True)
    assert result.status == 'partial'
    assert result.document_ids == [result.outputs[0].document_ref.id]
    assert [o.status for o in result.outputs] == ['success', 'error']
    assert process_two_outputs(retry=True).document_ids[0] == result.document_ids[0]
```

- [ ] Run `rtk proxy python -m pytest aquillm/apps/ingestion/tests/test_output_ledger.py aquillm/apps/ingestion/tests/test_output_recovery.py aquillm/apps/ingestion/tests/test_partial_upload_status.py -q`; expect missing manifest/partial/checkpoint assertions to fail.
- [ ] Add `IngestionOutput` with unique `(item, output_key)`, source occurrence, extraction revision, stable parent output FK, exact document reference, generation/attempt lease, next-attempt timestamp and typed failure code. Manifest registration completes before fan-out; malformed extraction registers an explicit failure instead of hiding dropped members. Materialize bytes outside locks. In a short transaction recheck current edit authorization, claim generation/attempt, create/reuse the exact owned document, checkpoint the output reference and preserve Document.save's existing chunk-publication outbox. Foreign source/content-hash collisions become explicit `content_hash_collision`, never appended retrieval text.
- [ ] Make item `document_ids`/metadata derived from committed output rows. Item states are queued/processing/success/partial/error; success requires all expected outputs committed or explicit duplicate results, partial has at least one committed result and a failed/unavailable result. Include `index_status` per output separately from parser success. A successful extraction is not a dense-ready claim. Existing API keys remain present; counts gains `partial`, and output records expose bounded issue/error codes without raw parser text.
- [ ] Move item dispatch to atomic durable intent and on-commit publication; recovery scans expired leases with bounded keyset batches of 25 (maximum 100). Same-key completed work is a no-op. Transient errors use existing bounded retry classification; permanent errors require an explicit reset rotating generation with exact-source fence. Persist terminal states before returning. No provider call occurs under transaction locks.
- [ ] Run new tests plus upload source-index, monitor, authorization and sanitize tests. Require collected `document_ids` to survive every crash window and broker outage. Commit explicit I3 paths with `rtk git commit -m "feat: checkpoint idempotent ingestion outputs and partial results"`.

### Task I4: Serve verified dense generations without replacing evidence

**Files:** Create `aquillm/lib/embeddings/document_contracts.py`, `aquillm/lib/embeddings/document_provider.py`, `aquillm/lib/embeddings/tests/test_document_contracts.py`, `aquillm/lib/embeddings/tests/test_document_provider.py`, `aquillm/apps/documents/models/embedding_index.py`, `aquillm/apps/documents/services/embedding_index_store.py`, `aquillm/apps/documents/services/embedding_index_query.py`, `aquillm/apps/documents/services/representation_store.py`, `aquillm/apps/documents/tasks/embedding_index.py`, `aquillm/apps/documents/management/commands/reindex_document_embeddings.py`, `aquillm/apps/documents/tests/test_embedding_generation.py`, `aquillm/apps/documents/tests/test_embedding_generation_query.py`; semantic drafts `aquillm/apps/documents/migrations/embedding_generation_v1.py`, `aquillm/apps/documents/migrations/embedding_generation_hnsw_v1.py`. Modify `aquillm/apps/documents/models/__init__.py`, `aquillm/apps/documents/models/chunks.py`, and `aquillm/lib/embeddings/provenance.py` only if an additive helper is needed; existing receipt schema and old APIs stay supported. The coordinator schedules the I-owned `models/chunks.py` FTS declaration after I4 review and before R3 migration/backfill.

**Interfaces:** Produces dense contracts/APIs above. `rebuild_document_embeddings(selection: RepresentationSelectionV1, *, space_key: str, using: str, timeout_ms: int=120000) -> str` schedules/resumes vector-only work without setting `ingestion_complete=False` or calling `create_chunks`.

- [ ] Test real facade/transport seams using fake responses: shuffled indices restored, missing/duplicate/bool indices rejected, wrong model and provider rejected, unknown response model produces no compatible observation, absent observed revision stays null, no fallback on contract failure, explicit verified endpoint binding distinguished from receipt evidence, and no prefix shrink. Identical model/revision/dimensions with fp16 versus 4-bit quantization, changed pooling, or changed templates must produce incompatible keys. Query/model A must not rank a profile B or unknown row even if all vectors are 1024D. Identical text with different image bytes or preparation keys cannot reuse a vector. Figure and conversation receipts remain vector-bound; document generations never authorize a conversation index automatically. Inject a fake clock to prove a 120-second build can progress beyond 15 seconds while a query cannot exceed its TurnBudget; neither emits a request after its own deadline.

```python
def test_unknown_and_foreign_space_are_not_dense_candidates(mixed_index, query_a):
    rows = mixed_index.rank(query_a)
    assert [row.chunk_id for row in rows] == [mixed_index.compatible_chunk_id]
    assert query_a.receipt['observed_identity']['revision'] is None
```

- [ ] Run `rtk proxy python -m pytest aquillm/lib/embeddings/tests/test_document_contracts.py aquillm/lib/embeddings/tests/test_document_provider.py aquillm/apps/documents/tests/test_embedding_generation.py aquillm/apps/documents/tests/test_embedding_generation_query.py -q`; expect missing adapter/store and compatibility failures before implementation.
- [ ] Implement immutable `EmbeddingSpace` declarations plus separate verification evidence. A real response creates the receipt; verification evidence binds the configured endpoint deployment artifact to a revision and preparation policy without rewriting observed identity. Enforce validated finite 1024D storage, explicit supported dimension transform, same actual modality, exact input hashes and response order. Media is loaded by immutable artifact reference outside locks, digest checked, transformed by the registered profile policy, and included in the exact canonical-JSON prepared-input digest. Missing/corrupt pixels cannot become an image vector; an explicitly permitted caption-only generation has its own actual modality. Both strict document/query paths stop on typed outage and never call Cohere/prefix repair to fill a generation. The legacy APIs keep their existing behavior for callers outside strict retrieval.
- [ ] Store `DocumentEmbeddingState` with an active generation per exact document/representation/space; `EmbeddingGeneration` with source hash, ordered chunk/input/media manifest, lease token, expected count, state and failure fence; `ChunkEmbedding` with chunk FK, exact vector/receipt/input pair and unique `(generation,chunk)`. Provider I/O happens outside locks. Use batches <=32 items and <=256 KiB prepared text, total local build deadline 120 seconds, at most three persisted deliveries. Apply the smaller existing 30-second request ceiling and caller remaining deadline; never serialize a monotonic timestamp to Celery. Persist each successful batch, then compare full source/representation/manifest and lease in one activation transaction. Duplicate delivery can reuse only exact input+space+modality+media bindings; initially disable cross-document reuse.
- [ ] Implement `freeze_dense_generations` with metadata-first reads of the caller's exact representation scope. Record each eligible ready generation and manifest, plus explicit unavailable entries in the digest; reject duplicate/foreign document mappings. Rank only the captured generation IDs that still match active pointer, snapshot identity/source/representation and query compatibility key. Compare source MD5 against current chunk content; reject stale/missing bodies. Null compatibility or empty generation membership returns no rows. Keep the caller's authorization and shared budget, deterministic finite distances and bounded limit. Test a vector-only A-to-B pointer swap between freeze, cache lookup and body publication: A results cannot publish, B cannot appear under A's captured digest, and a new acquisition has a different digest even though representation, profile and graph identities did not change. Exact filtered search is the correctness baseline; R2 owns measured ANN tuning behind the seam. Concurrent HNSW migration is non-atomic and checks invalid indexes before retry, with no unguarded setting escaping a pooled connection.
- [ ] Add `TextChunk.representation_generation = UUIDField(null=True, db_column='representation_generation_id')` in the I4 expand migration; every existing chunk remains null. I5 later alters this same field/column to the generation FK, after its table exists and before any non-null writes. Implement the initial legacy snapshot adapter by re-reading each supplied exact concrete document/source identity and enumerating every supported concrete model for its UUID. Exactly one owner must exist, not merely exactly one authorized owner. Match model/pkid/UUID/hash/collection, then capture `generation_key=None`, canonical SHA-256 and unknown original source revision as null. Its scope predicate matches only legacy-null chunks and revalidates both owner cardinality and source digest; it does not label their vectors as compatible. Recheck cardinality before cache/body publication to close a later UUID collision. A vector-only build can publish newly verified sidecar vectors over those stable chunks, so R2's strict compatibility path can ship without waiting for I5. Add a real PostgreSQL regression with an authorized RawTextDocument and unauthorized ImageUploadDocument sharing UUID and full_text_hash: frozen scope, compatible-sidecar dense and lexical paths all fail closed, return zero evidence, and expose only `source_scope_unavailable`. R2 runs this through real candidate acquisition; C2 source revalidation consumes the same guard. Add a test that an unambiguous legacy unknown vector stays excluded while a newly built verified sidecar over the same unchanged chunk becomes eligible. Reject non-null representation values until I5 installs a validated exact owner; never treat a dangling key as active.
- [ ] Run new modules and `rtk proxy python -m pytest aquillm/lib/embeddings/tests aquillm/apps/documents/tests/test_embedding_provenance.py aquillm/apps/documents/tests/test_embedding_outage_recovery.py aquillm/apps/documents/tests/test_chunk_publication_recovery.py aquillm/apps/documents/tests/test_chunk_graph_replacement.py aquillm/apps/chat/tests/test_conversation_indexing.py aquillm/apps/chat/tests/test_conversation_embedding_outage.py -q` in P0. Require no chunk/graph/citation-ID changes for vector-only rebuild. Commit explicit I4 paths with `rtk git commit -m "feat: publish compatible response-bound dense generations"`.

### Task I5: Activate structural representations and retain old citations

**Files:** Create `aquillm/apps/documents/models/representation.py`, `aquillm/apps/documents/services/representation_activation.py`, `aquillm/apps/documents/tasks/representation.py`, `aquillm/apps/documents/management/commands/reindex_document_representations.py`, `aquillm/apps/documents/tests/test_representation_generations.py`, `aquillm/apps/documents/tests/test_representation_activation.py`, `aquillm/apps/documents/tests/test_retired_chunk_citations.py`; semantic drafts `aquillm/apps/documents/migrations/representation_generations_v1.py`, `aquillm/apps/documents/migrations/generation_chunk_constraints_v1.py`. Modify the I4-created `aquillm/apps/documents/services/representation_store.py`, `aquillm/apps/documents/models/__init__.py`, `aquillm/apps/documents/models/document.py`, `aquillm/apps/documents/models/chunks.py`, `aquillm/apps/documents/tasks/chunking.py`, `aquillm/apps/documents/services/chunk_publication.py`, `aquillm/apps/documents/services/document_meta.py`, `aquillm/apps/documents/views/api.py`, `aquillm/apps/documents/views/pages.py`. G owns `aquillm/apps/knowledge_graph/graph/invalidation.py` and graph enumeration/build changes; R2 owns acquisition/materialization callsite integration.

**Interfaces:** Implements representation APIs above. `build_document_representation(document: DocumentRefV1, *, extraction_revision_id: int, chunker_key: str, using: str) -> str`. A representation manifest covers artifact/occurrence, parser key, canonical digest, complete/partial extraction status, chunker/preparation policy, ordered source offsets/text/input digests and immutable media digests. Dense profile is excluded from that representation key and belongs to I4.

- [ ] Add transaction tests where parser-v1 and parser-v2 differ, dense profiles A/B coexist, stale source changes during extraction, activation races a move/deletion, old worker completion follows newer activation, and publication crashes before/after commit. Assert old chunks retain PK/content/offsets, active lexical scope returns only the new generation, historical chunk detail reads its own canonical text/artifact, revoked access blocks both generations, graph intent commits with the pointer and no old chunk is deleted. A legacy-null row never matches an arbitrary newer generation.

```python
def test_parser_activation_keeps_legacy_citation(retained_generation):
    old = retained_generation.old_chunk
    retained_generation.activate_corrected_table()
    assert TextChunk.objects.get(pk=old.pk).content == old.content
    source = load_chunk_source(old.pk, using='default')
    assert source.canonical_text[source.start:source.end] == old.content
    assert old.pk not in retained_generation.active_chunk_ids()
```

- [ ] Run `rtk proxy python -m pytest aquillm/apps/documents/tests/test_representation_generations.py aquillm/apps/documents/tests/test_representation_activation.py aquillm/apps/documents/tests/test_retired_chunk_citations.py -q`; expect current destructive replacement and wrong source-context tests to fail.
- [ ] Add `DocumentRepresentationGeneration` and `DocumentRepresentationState` models, consuming I1's immutable `apps.ingestion.models.ExtractionRevision`. Generation contains exact owner, immutable source revision, parser/chunker keys, canonical snapshot and expected predecessor source/active key. Alter the I4 nullable `TextChunk.representation_generation` field to a generation FK on the same physical column. Replace existing per-document uniqueness with `fields=['doc_id','representation_generation','chunk_number']` and `fields=['doc_id','representation_generation','start_position','end_position']` constraints; separate conditional `representation_generation__isnull=True` constraints preserve each legacy uniqueness rule. Draft migration rehearses duplicates and validates constraints before removal; it never silently drops conflicting rows. Retired generations are never selected for retrieval but remain citable while their document exists. `Document.chunks` returns active scope; `TextChunk.document` resolves generation-owned exact model/pkid/UUID and rejects ambiguous legacy UUID ownership instead of returning the first model match.
- [ ] Snapshot existing legacy canonical text and source artifact before its first revision transition; preserve unknown artifact identity as null rather than attach a convenient new file. Bound migration batches and verify every old offset against its saved snapshot. Generation build stages chunks invisible to active scope and leaves currently valid old generation serving until atomic activation. Existing exact source changes fail current-generation eligibility immediately; retained history is not used as fresh evidence.
- [ ] Implement activation with lifecycle -> representation-state -> generation lock order, source/member/predecessor/manifest/count/lease checks, and G's non-destructive transactional invalidation/outbox seam. Update canonical Document fields and pointer in the same transaction using a validated internal transition object; extend Document.save's lifecycle-event handling so the old post-commit cleanup cannot delete retained rows or enqueue a competing legacy replacement. A caller-supplied bypass flag without a matching locked generation is rejected. G updates all content-replacement cleanup paths to preserve history; actual document deletion still cascades all generations. Commit graph refresh/build intent before activation becomes visible.

```python
lifecycle_reference = DocumentLifecycleRef(
    concrete_model_label=reference.model_label,
    document_pkid=reference.pkid, document_id=reference.id)
with locked_document_lifecycle_row(lifecycle_reference, collections, using=using):
    with transaction.atomic(using=using):
        state, generation = lock_state_and_generation(lease, using=using)
        if not matches_source_predecessor_manifest_and_lease(
                state, generation, lease=lease, expected_active_key=expected_active_key):
            return 'stale'
        prepare_document_representation_activation(
            reference, collections, expected_source_hash=generation.predecessor_hash,
            old_representation_key=state.active_generation_id,
            new_representation_key=generation.id, using=using)
        publish_locked_representation(state, generation, using=using)
```

`lock_state_and_generation`, `matches_source_predecessor_manifest_and_lease` and `publish_locked_representation` are private functions implemented in `representation_activation.py`; the publication function checks the internal transition token, saves canonical fields/chunks/pointer and marks readiness in this transaction. It never performs provider/storage I/O.

- [ ] Route `create_chunks` through representation generations while retaining its complete exact-document/source/publication envelope and terminal-failure semantics. Remove hash-only duplicate-vector copying from strict generation writes. Input transform failure leaves lexical-capable structural chunks and an unavailable dense generation; it never claims full semantic coverage. Capture Figure parent/source relationships by durable output key and use immutable image bytes/digests. Existing conversation chunks stay under their own transcript lifecycle; R2 must exclude conversation unknown/mismatched receipts from dense search until that consumer has a verified compatible writer, retaining its lexical fallback.
- [ ] Integrate `load_chunk_source` into chunk-detail/source APIs so retired offsets use their saved generation canonical text and file. Historical PDF/image serving selects only the authorized source binding for that chunk; it cannot substitute the newest doc file. Keep bounded full-text windows and current permission checks. R2/G consume the shared active scope and graph-generation tags; no retrieval fallback to retired chunks is allowed.
- [ ] Run new tests plus document lifecycle/move/save-lock, chunk graph identity/replacement/lifecycle, multimodal position uniqueness, citation API/highlight/PDF streaming and publication failure/reset tests. Require rolled-back transactions to expose neither new chunks nor graph intents. Commit explicit I5 paths with `rtk git commit -m "feat: activate structural generations with retained citations"`.

### Task I6: Rehearse migrations, measured reindexing and safe rollback

**Files:** Create `aquillm/apps/documents/tests/test_source_index_rollout.py`, `aquillm/apps/ingestion/tests/test_ingestion_rollout.py`, `aquillm/apps/documents/evals/__init__.py`, `aquillm/apps/documents/evals/source_index_regression.py`, `aquillm/apps/documents/evals/fixtures/source_index_cases.json`; create `docs/runbooks/rag-source-index-generations.md`. P0 owns disposable environment setup and the global migration registry; R2/G own their integration tests.

**Interfaces:** `reindex_document_embeddings` accepts `--space-key` as a registered lowercase 64-character SHA-256 key, `--limit` as integer 1..100 (default 25), optional `--document-id` as RFC UUID, `--database` as an explicitly available disposable alias, `--timeout-ms` as integer 1..120000, and `--dry-run` as a boolean switch. `reindex_document_representations` accepts registered `--parser-key structure-v2`, `--chunker-key structural-v1`, the same typed limit/document/database options, and `--dry-run`. They report exact eligible/partial/unsupported/missing-artifact/unknown-identity counts without providers or mutation. Non-dry-run rebuilds require explicit registered keys and resume durable leases; they never choose today's defaults for old rows. The test obtains the real digest from its synthetic registered profile rather than passing a display label as a hash:

```python
call_command('reindex_document_embeddings', space_key=fixture_space.space_key,
             limit=25, timeout_ms=120000, database='default', dry_run=True)
call_command('reindex_document_representations', parser_key='structure-v2',
             chunker_key='structural-v1', limit=25, database='default', dry_run=True)
```

Add CLI tests rejecting a display label in `--space-key`, invalid/unknown hashes, bool/negative/oversize numeric service arguments, malformed UUIDs and unavailable DB aliases. A dry-run installs a transport spy that raises on any provider call and asserts no database changes. These examples request no real credentials or production work.

- [ ] Build populated disposable fixtures with valid legacy citations, unknown vectors, two verified synthetic dense spaces, raw files/archive members, tables/formulas, figure parents and pending/terminal publication intents. Rehearse forward migration on a populated database, not only fresh schema. Test each semantic draft's coordinator-assigned dependencies with `rtk proxy python -m pytest aquillm/apps/documents/tests/test_source_index_rollout.py aquillm/apps/ingestion/tests/test_ingestion_rollout.py -q`.
- [ ] Implement the scenario: add schema while old reads stay active; deploy dual-schema readers and fenced writers; drain old workers; shadow-build verified dense A; activate A; build corrected representation B from immutable original bytes; test failure leaves previous current-source representation intact; atomically activate B with graph invalidation/outbox; build compatible vectors for B; switch dense profile C independently; roll back C to verified A only where representation/source bindings still match, otherwise lexical-only; roll back representation B to the retained prior representation through the same graph invalidation/outbox transaction. Retired citation identifiers resolve before and after every transition.
- [ ] Require zero missing tracked outputs under every injected crash point; zero wrong-column/table omissions in the synthetic fidelity suite; no dense eligibility for unknown/mismatched/truncated generations; and no retired chunks returned by active lexical/dense/graph retrieval. Record extraction quality per output rather than averaging away incomplete artifacts. Test deletion and sharing revocation after rollback.
- [ ] Run `rtk proxy python -m pytest aquillm/apps/ingestion/tests aquillm/lib/parsers/tests aquillm/lib/embeddings/tests aquillm/apps/documents/tests -q` in P0 after narrow tests pass, plus the coordinator's named R2/G/conversation integration tests. Compare fixed-character baseline and structural windows on the same synthetic labeled fixtures: exact numerical/sign fidelity, table/header association, support-span recall, source-region accuracy, redundant tokens, latency and peak memory. Parser promotion requires 100% preservation of fixture facts/unknown statuses; model quality/latency acceptance uses the program's independently declared held-out gates.
- [ ] Document phased switches: legacy readers during additive deployment; shadow generations with no public pointer changes; compatible dense reads only after verification; structural readers only after source/citation/graph consistency gates; disable old writers before activation. Operators see per-collection/modality generation coverage and lexical-only reasons. Old binaries unable to enforce active scopes cannot run after activation.
- [ ] Rehearse rollback without dropping new tables, relabeling receipts or repopulating unknown dense vectors. Pointer rollback selects a complete retained representation only for the same immutable source revision and re-invalidates graph artifacts; an actual source revision change requires a new generation instead. Dense rollback selects an actually verified matching profile/generation or disables dense. Broken historical artifacts remain explicitly unavailable, not repaired by an invented attachment. Deleting unused blobs uses a reference/retention scan; default retention is indefinite for citable generations while a document exists, and seven days for abandoned unreferenced staged blobs. No destructive cleanup runs as a migration.
- [ ] Commit explicit I6 paths with `rtk git commit -m "test: rehearse source index cutover and rollback"`; hand the coordinator the red/green transaction, migration, source fidelity and degraded-retrieval evidence before any real rollout is proposed.

## Self-review and integration gates

- F15 -> I1 artifact/member fixtures; F01 -> I2 native fidelity and I5 new parser/chunker generations; F16 -> I3 checkpoints, partial status and recovery; F02 -> I4 compatible dense identity and I5 representation manifests; F17 -> I2 explicit conversion/fallback outcomes.
- R2 must integrate both dense compatibility and active lexical scope before strict activation. G must remove destructive content-replacement deletion and bind graph artifacts to the captured representation before I5 activation. Neither a passed unit test nor a valid receipt authorizes skipping those gates.
- Migration draft names are assigned exactly once by P0; reviewers verify the resulting graph before any Django migration command. Provider/model defaults, unknown receipts, legacy-null selection and source artifact absence are covered explicitly.
- All plan examples use public code/audit or synthetic documents. No private deployment snapshot, internal record, study identifier or credential is needed.
- This lane does not start implementation or request another execution decision. The coordinator integrates this plan into the new comprehensive parallel program and presents the single user-facing execution handoff.
