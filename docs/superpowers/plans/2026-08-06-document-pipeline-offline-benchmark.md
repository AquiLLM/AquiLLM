# Offline Document-Pipeline Benchmark Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a reproducible local preprocessing benchmark over a frozen 17-PDF astronomy corpus and deterministic 1/10/50/100-page synthetic PDFs, producing paper-ready absolute counts and carefully scoped local timing data without database, embedding, or inference services.

**Architecture:** Refactor the existing PDF parser and character-window chunk construction into production-owned pure boundaries used unchanged by ingestion and the benchmark. A dedicated offline document runner owns corpus validation, deterministic fixtures, metrics, 30 repeated timing sweeps, separate memory passes, artifact validation, comparison, and provenance while reusing the established canonical hashing and process-local network guard.

**Tech Stack:** Python 3.13, Django 5.2, pypdf, PyYAML, pytest, `time.perf_counter_ns`, `tracemalloc`, standard-library `hashlib/json/csv/statistics`, and the existing offline evaluation helpers.

---

## File map

- Modify `aquillm/aquillm/ingestion/parsers.py`: expose a production primary-text extraction boundary and ensure normal PDF ingestion still detects once and appends figures by default.
- Create `aquillm/apps/documents/services/text_chunk_plan.py`: pure, validated production chunk specification helper.
- Modify `aquillm/apps/documents/tasks/chunking.py`: build `TextChunk` instances from the pure helper without changing historical behavior.
- Create `aquillm/apps/ingestion/tests/test_primary_text_extraction.py`: parser fidelity and one-detection/no-figure tests.
- Create `aquillm/apps/documents/tests/test_text_chunk_plan.py`: historical chunk-boundary equivalence tests.
- Create `aquillm/apps/documents/tests/test_text_chunk_task_plan.py`: task-level multi-window persistence, batching, progress, and image-position equivalence.
- Create `aquillm/apps/chat/evals/offline/document_pipeline_schema.py`: inventory schema, deterministic minimal-PDF generator, absolute metrics, canonical case records, and safe diagnostics.
- Create `aquillm/apps/chat/evals/offline/document_pipeline_runner.py`: corpus matching, production pipeline execution, rotating timing sweeps, memory passes, aggregation, and manifest creation.
- Create `aquillm/apps/chat/evals/offline/document_pipeline_artifacts.py`: exact artifact writing, validation, normalized comparison, table/report/CSV rendering, and provenance.
- Modify `aquillm/apps/chat/evals/run_offline_evidence.py`: add Windows-compatible `document-*` commands to the existing no-side-effect offline Django bootstrap.
- Create `aquillm/apps/chat/evals/offline/document_corpus_inventory.yaml`: frozen 17-member hash/size inventory without paths or contents.
- Create `aquillm/apps/chat/evals/offline/document_corpus_review.yaml`: independent inventory/synthetic-protocol review record.
- Create `aquillm/apps/chat/tests/test_offline_document_pipeline_schema.py`: inventory, generator, count, failure, and aggregation tests.
- Create `aquillm/apps/chat/tests/test_offline_document_pipeline_runner.py`: production-call, timing, memory, network, and corpus-runner tests.
- Create `aquillm/apps/chat/tests/test_offline_document_pipeline_artifacts.py`: artifacts, privacy, comparison, table, CLI, and provenance tests.
- Generate `docs/evaluation/offline/document-pipeline/2026-08-06-canonical-a/` and `2026-08-06-canonical-b/` only after external comparison passes.
- Generate `docs/evaluation/offline/document-pipeline/2026-08-06-results-summary.md` and a follow-up provenance JSON.

## Public interfaces

```python
# apps/documents/services/text_chunk_plan.py
@dataclass(frozen=True)
class TextChunkSpec:
    content: str
    start_position: int
    end_position: int
    chunk_number: int

def plan_text_chunks(text: str, *, chunk_size: int, overlap: int) -> list[TextChunkSpec]: ...

# aquillm/ingestion/parsers.py
def extract_primary_text_payload(
    filename: str,
    data: bytes,
    *,
    content_type: str | None = None,
    ingest_type: str | None = None,
) -> ExtractedTextPayload: ...

# document_pipeline_schema.py
def load_document_inventory(path: Path) -> dict: ...
def validate_document_inventory(data: dict) -> None: ...
def load_document_review(path: Path, inventory_path: Path, *, allow_pending: bool = False) -> dict: ...
def validate_document_review(data: dict, inventory: dict, *, allow_pending: bool = False) -> None: ...
def freeze_document_inventory(
    corpus_dir: Path, *, expected_count: int, expected_total_bytes: int
) -> dict: ...
def generate_synthetic_pdf(page_count: int) -> tuple[bytes, str]: ...
def build_pending_document_review(inventory_path: Path) -> dict: ...
def build_document_record(
    *, arm: str, case_id: str, input_bytes: int, page_count: int | None,
    success: bool, diagnostic_code: str, sanitized_text: str | None,
    chunk_specs: Sequence[TextChunkSpec] | None,
) -> dict[str, object]: ...
def aggregate_document_results(
    static_records: Sequence[Mapping[str, object]],
    timing_sweeps: Sequence[Mapping[str, object]],
    memory_records: Sequence[Mapping[str, object]],
    *, network_audit: Mapping[str, object],
) -> dict[str, object]: ...

# document_pipeline_runner.py
def resolve_real_corpus(corpus_dir: Path, inventory: dict) -> list[dict]: ...
def run_document_case(case: dict, *, chunk_size: int, overlap: int) -> dict: ...
def run_document_benchmark(...) -> dict: ...

# document_pipeline_artifacts.py
def write_document_artifacts(result: dict, output_dir: Path) -> None: ...
def validate_document_artifacts(output_dir: Path) -> None: ...
def normalized_document_result(output_dir: Path) -> bytes: ...
def compare_document_results(first: Path, second: Path) -> None: ...
def regenerate_document_table(aggregate_path: Path) -> str: ...
def write_document_provenance(aggregate_path: Path, artifact_commit: str, output_path: Path) -> None: ...
def validate_document_provenance(path: Path, repository: Path) -> None: ...
```

## Chunk 1: Production fidelity and frozen inputs

### Task 1: Extract production-owned text and chunk boundaries

**Files:**
- Modify: `aquillm/aquillm/ingestion/parsers.py`
- Create: `aquillm/apps/documents/services/text_chunk_plan.py`
- Modify: `aquillm/apps/documents/tasks/chunking.py`
- Create: `aquillm/apps/ingestion/tests/test_primary_text_extraction.py`
- Create: `aquillm/apps/documents/tests/test_text_chunk_plan.py`
- Create: `aquillm/apps/documents/tests/test_text_chunk_task_plan.py`

- [ ] **Step 1: Write RED parser-fidelity tests**

Patch only call counters. Add a small test-owned valid one-page PDF byte fixture directly in `test_primary_text_extraction.py`; it is independent of the production synthetic generator created in Task 2. Require `extract_text_payloads("paper.pdf", ...)` to call type detection once, call the new PDF-only primary helper once, and append figures once under the default path. Require a benchmark-style call to `extract_primary_text_payload(..., ingest_type="document")` to execute the real `extract_pdf_text` over that fixture without invoking the figure hook. Assert primary payload fields and text are identical, and prove that supplying `ingest_type` prevents re-detection.

In the same new test file, explicitly parameterize every existing non-PDF dispatch group and patch its selected extractor: archive; image; audio; video; raw-text `.txt/.md/.doc/.rtf`; HTML `.html/.htm`; DOCX; raw ODT; EPUB; CSV; TSV; XLSX; XLS; ODS; PPTX; raw PPT; ODP; JSON; JSONL; XML; YAML/YML; VTT; SRT; MIME-only `text/*` fallback; and unsupported fallback. Assert the current figure hook executes only for DOCX, EPUB, XLSX, ODS, and PPTX, and does not execute for every other non-PDF group. Assert the same normalized types, singleton/list shapes, archive depth, and unsupported error behavior.

- [ ] **Step 2: Verify parser tests fail for the missing boundary**

Run from the worktree root:

```powershell
Set-Location aquillm
python -m pytest apps/ingestion/tests/test_primary_text_extraction.py -q
Set-Location ..
```

Expected: import failure for `extract_primary_text_payload`.

- [ ] **Step 3: Implement the minimal primary-text helper**

Scope the new helper to the PDF primary payload only. Accept a precomputed type so the caller does not detect twice. `extract_text_payloads` detects once, calls the helper only for `.pdf`, and then appends PDF figures exactly as before. Leave archive, image, audio/video, text, DOCX, spreadsheet, presentation, EPUB, transcript, and structured-format branches behaviorally unchanged, including their existing per-format figure policies.

- [ ] **Step 4: Run the parser suite to prove GREEN before writing chunk tests**

```powershell
Set-Location aquillm
python -m pytest apps/ingestion/tests/test_primary_text_extraction.py apps/ingestion/tests/test_unified_ingestion_parsers.py -q
Set-Location ..
```

- [ ] **Step 5: Write RED chunk-planning tests**

Use a test-only historical slice implementation as the oracle. Cover invalid configuration and lengths 0, 1, 1663, 1664, 2047, 2048, 2049, and 10,000. Assert exact content, half-open spans, numbering, final span, and coverage.

Add task-level RED tests for empty and 10,000-character documents. With model/database boundaries patched as existing task tests do, require exact persisted contents/spans/numbers, planner invocation, embedding batch input order, progress completion, and unchanged image-chunk position after the last text span.

- [ ] **Step 6: Verify both chunk suites fail for the missing planner/adoption**

```powershell
Set-Location aquillm
python -m pytest apps/documents/tests/test_text_chunk_plan.py apps/documents/tests/test_text_chunk_task_plan.py -q
Set-Location ..
```

Expected: FAIL because the planner is missing and the task has not adopted it.

- [ ] **Step 7: Implement the pure helper and adopt it in the task**

The helper returns immutable specs only. `create_chunks` converts specs to `TextChunk` model instances and preserves modality, positions, numbers, embedding batching, progress, and image-chunk behavior.

- [ ] **Step 8: Run focused and adjacent tests to prove GREEN**

```powershell
Set-Location aquillm
python -m pytest apps/ingestion/tests/test_primary_text_extraction.py apps/ingestion/tests/test_unified_ingestion_parsers.py apps/documents/tests/test_text_chunk_plan.py apps/documents/tests/test_text_chunk_task_plan.py apps/documents/tests/test_multimodal_chunk_position_uniqueness.py -q
Set-Location ..
```

- [ ] **Step 9: Commit**

```powershell
git add aquillm/aquillm/ingestion/parsers.py aquillm/apps/documents/services/text_chunk_plan.py aquillm/apps/documents/tasks/chunking.py aquillm/apps/ingestion/tests/test_primary_text_extraction.py aquillm/apps/documents/tests/test_text_chunk_plan.py aquillm/apps/documents/tests/test_text_chunk_task_plan.py
git commit -m "refactor(ingest): expose offline preprocessing boundaries"
```

### Task 2: Freeze inventory and deterministic scaling fixtures

**Files:**
- Create: `aquillm/apps/chat/evals/offline/document_pipeline_schema.py`
- Create: `aquillm/apps/chat/evals/offline/document_corpus_inventory.yaml`
- Create: `aquillm/apps/chat/evals/offline/document_corpus_review.yaml`
- Create: `aquillm/apps/chat/tests/test_offline_document_pipeline_schema.py`

- [ ] **Step 1: Write RED inventory-schema tests**

Require schema/version, exactly 17 unique `real-NNN` cases, raw PDF SHA-256, positive size, fixed rationale/lineage/sensitivity/license fields, no basename/path/title/content fields, and no absolute path. Define a sibling review schema containing canonical-text hash algorithm `sha256-utf8-lf-v1`, canonical inventory/protocol hashes, a deliberate stable agent identifier, reviewer role/date/decisions, and status. Prove pending review is rejected by default, `allow_pending=True` accepts only structurally valid pending input, and independently approved review is accepted. Privacy validation covers both YAML files.

- [ ] **Step 2: Run inventory tests to prove RED**

```powershell
Set-Location aquillm
python -m pytest apps/chat/tests/test_offline_document_pipeline_schema.py -q
Set-Location ..
```

Expected: FAIL because the inventory/review schema module is missing.

- [ ] **Step 3: Implement inventory and review validation**

Implement `load_document_inventory`, `validate_document_inventory`, `load_document_review`, `validate_document_review`, and `freeze_document_inventory`. Use raw-byte SHA-256 only for PDF inventory members and newline-canonical hashes for YAML/review lineage. `freeze_document_inventory` discovers direct-child `.pdf` members case-insensitively, rejects duplicate hashes, sorts by raw hash, assigns `real-NNN`, verifies the caller-supplied exact count and byte total, and emits no filename/path. It is the only Task 2 filesystem discovery helper; Task 3 still owns runtime matching and diagnostics.

Reuse the repository's existing `sha256-utf8-lf-v1` implementation exactly for the committed inventory source file: decode UTF-8, replace CRLF and lone CR with LF, preserve all other decoded code points and bytes including a leading `U+FEFF` and the presence/absence of a final newline, encode UTF-8, then SHA-256. Add fixed LF/CRLF/lone-CR/no-final-newline/leading-BOM test vectors against `sha256_canonical_text` so no second interpretation can emerge. Separately, the document inventory/review loaders reject a leading BOM as invalid fixture syntax before calling the hash helper; this does not redefine the shared algorithm. Inventory member hashes remain hashes of unmodified raw PDF bytes.

The review file contains a dedicated `protocol` mapping. Its hash algorithm is separately named `sha256-canonical-json-v1`: serialize only that mapping using the existing `canonical_json_bytes` helper (sorted keys, compact separators, UTF-8, exactly one final LF), then SHA-256. Add a fixed protocol test vector with a literal expected digest. Because only `protocol` is hashed, `protocol_hash`, `status`, reviewer identity/date/decisions, and all mutable approval fields are excluded by construction.

- [ ] **Step 4: Run inventory tests to prove GREEN**

```powershell
Set-Location aquillm
python -m pytest apps/chat/tests/test_offline_document_pipeline_schema.py -q
Set-Location ..
```

- [ ] **Step 5: Generate and freeze the 17-member inventory before measuring outputs**

Operator precondition: set the process-local `AQUILLM_LOCAL_ASTRO_CORPUS` environment variable to the private selected corpus directory without writing its value to any committed file or captured diagnostic. Run the exact API below from `aquillm/`; it hashes only direct-child PDFs and writes canonical YAML containing no path or filename. The function must abort before writing unless it observes exactly 17 PDFs totaling exactly 97,006,698 bytes. Do not execute production extraction yet.

```powershell
Set-Location aquillm
python -c "import os; from pathlib import Path; import yaml; from apps.chat.evals.offline.document_pipeline_schema import freeze_document_inventory; p=Path(os.environ['AQUILLM_LOCAL_ASTRO_CORPUS']); d=freeze_document_inventory(p, expected_count=17, expected_total_bytes=97006698); print(yaml.safe_dump(d, sort_keys=True, allow_unicode=True, line_break='\n'), end='')"
Set-Location ..
```

Capture the printed mapping, verify its reported count/total, and add those exact path-free bytes with `apply_patch`; never redirect private-path diagnostics or generated output into the repository.

- [ ] **Step 6: Write RED deterministic minimal-PDF tests**

Require exact byte identity across two generation calls, exact page count, authored ASCII page strings in order, and exact expected post-sanitize/strip text `"\n\n".join(page_strings).strip()` at 1, 10, 50, and 100 pages. To test corruption, truncate the returned bytes before passing them to the parser/fixture validator; generation itself accepts only a positive page count and has no corruption mode.

- [ ] **Step 7: Run deterministic-PDF tests to prove RED**

```powershell
Set-Location aquillm
python -m pytest apps/chat/tests/test_offline_document_pipeline_schema.py::test_synthetic_pdf_is_byte_deterministic apps/chat/tests/test_offline_document_pipeline_schema.py::test_synthetic_pdf_page_count_and_expected_text apps/chat/tests/test_offline_document_pipeline_schema.py::test_synthetic_pdf_rejects_nonpositive_page_count apps/chat/tests/test_offline_document_pipeline_schema.py::test_truncated_synthetic_pdf_fails_validation -q
Set-Location ..
```

Expected: FAIL because `generate_synthetic_pdf` is missing.

- [ ] **Step 8: Implement the minimal PDF builder and frozen expected hashes**

Build this exact ASCII PDF 1.4 grammar, joining every displayed line with literal `b"\n"` and ending with exactly one LF. Object 1 is `<< /Type /Catalog /Pages 2 0 R >>`; object 2 is `<< /Type /Pages /Count N /Kids [4 0 R 6 0 R ...] >>`; object 3 is `<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>`. For zero-based page `i`, its page object is `4 + 2*i`, its content object is `5 + 2*i`, and objects are emitted in numeric order. Each page dictionary is exactly `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents C 0 R >>`. Each content object is exactly `<< /Length L >>\nstream\nBT /F1 12 Tf 72 720 Td (TEXT) Tj ET\nendstream`, where `L` counts only the ASCII content line bytes and `TEXT` escapes backslash and parentheses. Every indirect object is `O 0 obj\nBODY\nendobj\n`. The header is `%PDF-1.4\n%AquiLLM\n`. The xref is `xref\n0 SIZE\n`, followed by `0000000000 65535 f \n` and one live line per object formatted as ten-digit byte offset plus ` 00000 n \n`. Finish exactly with `trailer\n<< /Size SIZE /Root 1 0 R >>\nstartxref\nXREF_OFFSET\n%%EOF\n`. Page strings use four-digit numbering. Use no current time, random ID, hostname, path, downloaded font, or external generator. The tests construct authored expected page strings and normalized text independently of the generator return value. Store expected generated-byte and normalized-output hashes in the pending protocol.

- [ ] **Step 9: Run deterministic-PDF tests to prove GREEN**

Run the exact Step 7 command and require PASS before adding metric tests.

- [ ] **Step 10: Write RED absolute-metric tests**

Assert Unicode code points, UTF-8 bytes, whitespace words, actual production estimated-token function call, chunk counts, union coverage, excess overlap, ratio denominator, chunk summaries, exact output preimage hash, and empty/failure conventions.

- [ ] **Step 11: Run absolute-metric tests to prove RED**

```powershell
Set-Location aquillm
python -m pytest apps/chat/tests/test_offline_document_pipeline_schema.py::test_build_document_record_absolute_units apps/chat/tests/test_offline_document_pipeline_schema.py::test_build_document_record_uses_production_token_estimator apps/chat/tests/test_offline_document_pipeline_schema.py::test_build_document_record_failure_conventions apps/chat/tests/test_offline_document_pipeline_schema.py::test_aggregate_document_results_ratio_denominators -q
Set-Location ..
```

Expected: FAIL because `build_document_record` and `aggregate_document_results` are missing.

- [ ] **Step 12: Implement and prove GREEN for absolute metrics**

Implement the exact public signatures above. `build_document_record` returns precisely the static-record keys/types enumerated under **Versioned artifacts and schemas** in the design; successful metrics derive from `sanitized_text` and `chunk_specs`, while every output-conditioned field is null on failure. `aggregate_document_results` returns precisely the eight top-level mappings and nested arm/timing/memory/failure contracts enumerated in that same section. Use integer absolute units, the production token estimator, explicit null failure conventions, union-of-half-open-spans coverage, excess overlap, and ratio-of-sums denominators. Re-run the complete schema test file and require PASS.

- [ ] **Step 13: Write and verify RED pending-review construction tests**

Specify this exact mapping shape: `schema_version`, `review_id`, `status`, `source_hash_algorithm`, `inventory_hash`, `protocol_hash_algorithm`, `protocol_hash`, `protocol`, `reviewer_identity`, `reviewer_role`, `review_date`, and `decisions`. `protocol` contains `generator_version`, `page_counts: [1, 10, 50, 100]`, the exact authored string template, and four entries containing page count, generated-PDF raw SHA-256, expected-normalized-text SHA-256, and expected page count. In pending state the three reviewer fields are null and `decisions` is empty.

```powershell
Set-Location aquillm
python -m pytest apps/chat/tests/test_offline_document_pipeline_schema.py::test_build_pending_review_exact_shape_and_hashes -q
Set-Location ..
```

Expected: FAIL because `build_pending_document_review` is missing.

- [ ] **Step 14: Implement pending-review construction and prove GREEN**

Implement `build_pending_document_review(inventory_path)` exactly as specified and rerun Step 13 to PASS. Then generate the canonical pending YAML into a temporary file outside the repository and add its exact path-free bytes to the committed fixture using `apply_patch`:

```powershell
Set-Location aquillm
python -c "import os; from pathlib import Path; import yaml; from apps.chat.evals.offline.document_pipeline_schema import build_pending_document_review; p=Path('apps/chat/evals/offline/document_corpus_inventory.yaml'); d=build_pending_document_review(p); Path(os.environ['TEMP']).joinpath('document_corpus_review.yaml').write_text(yaml.safe_dump(d, sort_keys=True, allow_unicode=True, line_break='\n'), encoding='utf-8', newline='\n')"
python -m pytest apps/chat/tests/test_offline_document_pipeline_schema.py::test_build_pending_review_exact_shape_and_hashes apps/chat/tests/test_offline_document_pipeline_schema.py::test_pending_review_requires_opt_in apps/chat/tests/test_offline_document_pipeline_schema.py::test_review_hash_fixed_vectors -q
Set-Location ..
```

Require PASS with `allow_pending=True` and explicit rejection without it. No production parser or benchmark run is allowed yet.

- [ ] **Step 15: Commit pending frozen inputs before review**

Validate the generated review explicitly in pending mode, prove default loading still rejects it, then commit the inventory, pending review/protocol, schema implementation, and tests as a distinct checkpoint before dispatching the reviewer:

```powershell
Set-Location aquillm
python -m pytest apps/chat/tests/test_offline_document_pipeline_schema.py::test_pending_review_requires_opt_in apps/chat/tests/test_offline_document_pipeline_schema.py::test_build_pending_review_exact_shape_and_hashes -q
Set-Location ..
git add aquillm/apps/chat/evals/offline/document_pipeline_schema.py aquillm/apps/chat/evals/offline/document_corpus_inventory.yaml aquillm/apps/chat/evals/offline/document_corpus_review.yaml aquillm/apps/chat/tests/test_offline_document_pipeline_schema.py
git commit -m "test(eval): freeze pending document benchmark inputs"
```

- [ ] **Step 16: Obtain and record genuinely independent approval**

Dispatch a fresh reviewer that did not author the inventory. Give it read-only access to the 17 local PDFs, pending inventory/protocol, generator tests, and approved spec. It verifies exact PDF membership/total, raw hashes/sizes, absence of private fields, authored strings, generated hashes, sensitivity/license wording, and that production functions have not been measured. The reviewer returns a stable task identifier and decisions. The coordinator records that returned identity/decision and `status: approved`, then re-dispatches the same reviewer to verify the exact approved bytes and canonical hashes. The implementer must not self-approve.

- [ ] **Step 17: Run tests and commit approved frozen inputs**

```powershell
Set-Location aquillm
python -m pytest apps/chat/tests/test_offline_document_pipeline_schema.py -q
Set-Location ..
git add aquillm/apps/chat/evals/offline/document_pipeline_schema.py aquillm/apps/chat/evals/offline/document_corpus_inventory.yaml aquillm/apps/chat/evals/offline/document_corpus_review.yaml aquillm/apps/chat/tests/test_offline_document_pipeline_schema.py
git commit -m "test(eval): freeze offline document benchmark inputs"
```

Do not run the canonical benchmark before this commit and its independent review.

## Chunk 2: Measurement runner and artifact contracts

### Task 3: Implement timing, memory, failure, and aggregate measurements

**Files:**
- Create: `aquillm/apps/chat/evals/offline/document_pipeline_runner.py`
- Create: `aquillm/apps/chat/tests/test_offline_document_pipeline_runner.py`

- [ ] **Step 1: Write and verify RED corpus-resolution/production-path tests**

Test `resolve_real_corpus`: discover direct-child files only, match `.pdf` case-insensitively, ignore non-PDF sidecars, default `allow_unlisted_pdfs=False`, reject unlisted PDFs, missing inventory hashes, duplicate file hashes, and size/hash mismatch, and return only stable case IDs plus preloaded bytes with path-free diagnostic codes. Patch call counters only and require one-case execution to call detection, production primary extraction, exact sanitize/strip, and the production chunk planner. Assert figures, database, embeddings, and network are never called; token estimation is explicitly not part of the timed pipeline.

```powershell
Set-Location aquillm
python -m pytest apps/chat/tests/test_offline_document_pipeline_runner.py -k "resolve_real_corpus or production_path" -q
Set-Location ..
```

Expected: FAIL because the resolver/runner module is missing.

- [ ] **Step 2: Implement corpus resolution and one-case production execution, then prove GREEN**

Implement both APIs. The case object contains only `arm`, stable `case_id`, preloaded `pdf_bytes`, precomputed raw hash, and precomputed page count. Keep page counting, hashing, validation, metric construction, and token estimation outside the direct outer timer. Return a static record and timing observation without document content or raw exception strings. Re-run the exact Step 1 command and require PASS.

- [ ] **Step 3: Write and verify RED stage/failure tests**

Use a deterministic fake clock to prove nested stage units, direct outer `combined_ns`, nullable unexecuted stages, terminal-stage enum, populated input/page denominators on failures, safe diagnostic mapping, and synthetic failure as an integrity error.

```powershell
Set-Location aquillm
python -m pytest apps/chat/tests/test_offline_document_pipeline_runner.py -k "stage or failure or combined" -q
Set-Location ..
```

Expected: FAIL on missing stage timing/failure semantics.

- [ ] **Step 4: Implement stage/failure handling and prove GREEN**

Use `time.perf_counter_ns` for the direct interval and each nested stage. The direct interval starts immediately before detection and ends immediately after chunk planning or the terminal failure. Token estimation and record/metric validation occur only after the interval. Re-run Step 3 and require PASS.

- [ ] **Step 5: Write and verify RED sweep tests**

Assert dependencies and the production token estimator are initialized before warm-up. Run one full-arm unreported warm-up for the real arm and separately one full-arm unreported warm-up for the synthetic arm (two warm-ups total), immediately followed by exactly 30 measured sweeps for that arm. Each arm starts from case-ID-sorted order and rotates by `sweep_index mod arm_case_count`. Require 630 timing-case rows, 60 timing-sweep rows, exact copied static work-unit fields, `case_combined_sum_ns`, successful-only denominator, effective and success-conditioned rates, nearest-rank p95 over sweep values, and no pooled pseudo-replication. Input loading/hashing/page counting, synthetic generation, token estimation, metric calculation, and validation must never occur inside an observation.

```powershell
Set-Location aquillm
python -m pytest apps/chat/tests/test_offline_document_pipeline_runner.py -k "warmup or sweep or rotation or rate or p95" -q
Set-Location ..
```

Expected: FAIL because sweep execution/aggregation is missing.

- [ ] **Step 6: Implement sweeps and prove GREEN**

Use raw integer nanoseconds. Regenerate each sweep row from case rows and static records. Every rate is ratio-of-sums and null on zero denominator. Re-run Step 5 and require PASS.

- [ ] **Step 7: Write and verify RED memory/manifest tests**

Patch `tracemalloc` and GC to prove this exact lifecycle for each of three separate memory-only passes per case: preload input bytes and precompute page count outside tracing; call `gc.collect()`; call fresh `tracemalloc.start()`; call `tracemalloc.reset_peak()`; execute the same detect→extract→sanitize→chunk pipeline; read the peak; and call `tracemalloc.stop()` in `finally` on both success and failure. Assert input allocation, page counting, token estimation, metric construction, and validation remain outside tracing; timing code never traces; failed peaks remain separate; and per-case/arm maxima follow the specification. Require non-secret machine context, fixed chunk config, source/corpus/config hashes, timer resolution, repetitions, dependency versions, and the exact process-local network-audit wording.

```powershell
Set-Location aquillm
python -m pytest apps/chat/tests/test_offline_document_pipeline_runner.py -k "memory or manifest" -q
Set-Location ..
```

Expected: FAIL because memory passes/manifest construction are missing.

- [ ] **Step 8: Implement memory passes and manifest, then prove GREEN**

The runner accepts a network-audit result but does not own the guard; Task 4's CLI owns the one enclosing guard scope. Implement memory/manifest behavior, re-run Step 7, then run the complete runner/schema tests.

- [ ] **Step 9: Commit**

```powershell
Set-Location aquillm
python -m pytest apps/chat/tests/test_offline_document_pipeline_runner.py apps/chat/tests/test_offline_document_pipeline_schema.py -q
Set-Location ..
git add aquillm/apps/chat/evals/offline/document_pipeline_runner.py aquillm/apps/chat/tests/test_offline_document_pipeline_runner.py
git commit -m "feat(eval): measure offline document preprocessing"
```

### Task 4: Implement exact artifacts, CLI, comparison, and provenance

**Files:**
- Create: `aquillm/apps/chat/evals/offline/document_pipeline_artifacts.py`
- Modify: `aquillm/apps/chat/evals/run_offline_evidence.py`
- Create: `aquillm/apps/chat/tests/test_offline_document_pipeline_artifacts.py`
- Modify: `README.md`

- [ ] **Step 1: Write and verify RED artifact membership/schema/privacy tests**

Require the 13 files in the specification, row cardinalities 17/4/630/60/63, exact required keys/types/units, valid foreign keys, canonical JSONL ordering, raw `COMPLETE` hashes, and immutable non-overwriting output. Reject usernames, hostnames, absolute paths, basenames, document titles/content, raw exceptions, credential fields/values, missing/extra files, inconsistent aggregates, and non-regenerating CSV/report/table output.

```powershell
Set-Location aquillm
python -m pytest apps/chat/tests/test_offline_document_pipeline_artifacts.py -k "membership or schema or privacy or regenerate or overwrite" -q
Set-Location ..
```

Expected: FAIL because artifact writing/validation is missing.

- [ ] **Step 2: Implement atomic artifact writing/validation and prove GREEN**

Write to a sibling temporary directory, validate it, then atomically rename. CSVs derive only from static JSONL. Markdown derives only from aggregate/static sources. Failure-conditioned values render as `N/A`, never zero.

Re-run Step 1 and require PASS.

- [ ] **Step 3: Write and verify RED normalized-comparison tests**

Assert the exact exclusion paths from the specification, require all identity/config/count/diagnostic/network fields identical, and prove a one-count mutation fails comparison while a timing-only mutation passes.

```powershell
Set-Location aquillm
python -m pytest apps/chat/tests/test_offline_document_pipeline_artifacts.py -k "normalized or compare" -q
Set-Location ..
```

Expected: FAIL because normalization/comparison is missing.

- [ ] **Step 4: Implement normalized comparison and prove GREEN**

Implement `normalized_document_result` and `compare_document_results` over parsed canonical sources with only the design's exact exclusions. Re-run Step 3 and require PASS.

- [ ] **Step 5: Write RED table/provenance tests, implement, and prove GREEN**

Require title `Local document preprocessing measurements`, separate real/synthetic results, fixed-convenience-corpus qualification, “estimated tokens” terminology, warm single-process in-memory scope, exact process-local guard wording, and complete caption exclusions. Provenance records evaluated source, artifact commit, exact artifact hashes, corpus inventory hash, source/config hashes, and the source hash algorithm without claiming its own commit. Run tests selected by `-k "table or provenance"` before implementation to confirm RED, implement, then rerun to PASS.

- [ ] **Step 6: Write and verify RED Windows CLI/network subprocess tests**

Test `--help`, validation/comparison/table/provenance, existing-output rejection, corrupt synthetic failure, dirty-source rejection for canonical real runs, and startup with no external services or ambient credentials. The one-page subprocess uses an explicitly noncanonical `--noncanonical-smoke --synthetic-pages 1` mode that executes the pipeline and writes only `smoke-result.json`; it cannot write the 13-file schema, cannot pass `document-validate`, and is rejected if combined with canonical 30/3 settings. Canonical mode exclusively enforces 17/4/630/60/63 and 30/3.

The CLI must enter the existing `deny_network` scope before any run-specific dependency initialization or fixture generation, keep it active through parsing, measurement, aggregation, atomic artifact writing, and validation, then fail if the audit observed any attempt. Add a RED subprocess/unit case where production catches the guard's blocked-socket exception but the command still fails; persisted audit details contain operation names only and never IP addresses, ports, or hostnames.

```powershell
Set-Location aquillm
python -m pytest apps/chat/tests/test_offline_document_pipeline_artifacts.py -k "cli or subprocess or network or noncanonical" -q
Set-Location ..
```

Expected: FAIL because the `document-*` commands are missing.

- [ ] **Step 7: Implement CLI and README instructions, then prove GREEN**

Extend the existing `python -m apps.chat.evals.run_offline_evidence` bootstrap with subcommands `document-run`, `document-validate`, `document-compare`, `document-table`, and `document-provenance`. `document-run` accepts `--real-corpus`, `--inventory`, `--output`, `--sweeps`, and `--memory-repeats`, plus the isolated test-only smoke arguments above. Canonical mode requires 30/3 and a clean source. Re-run Step 6 and require PASS.

- [ ] **Step 8: Run all new and adjacent tests and commit**

```powershell
Set-Location aquillm
python -m pytest apps/ingestion/tests/test_primary_text_extraction.py apps/documents/tests/test_text_chunk_plan.py apps/chat/tests/test_offline_document_pipeline_schema.py apps/chat/tests/test_offline_document_pipeline_runner.py apps/chat/tests/test_offline_document_pipeline_artifacts.py -q
python -m pytest apps/ingestion/tests/test_unified_ingestion_parsers.py apps/documents/tests/test_multimodal_chunk_position_uniqueness.py apps/chat/tests/test_offline_evidence_metrics.py apps/chat/tests/test_offline_evidence_schema.py apps/chat/tests/test_offline_evidence_runner.py -q
Set-Location ..
git diff --check
git add aquillm/apps/chat/evals/offline/document_pipeline_artifacts.py aquillm/apps/chat/evals/run_offline_evidence.py aquillm/apps/chat/tests/test_offline_document_pipeline_artifacts.py README.md
git commit -m "feat(eval): add reproducible document benchmark artifacts"
```

## Chunk 3: Canonical data and publication

### Task 5: Execute, inspect, publish, and push the canonical measurements

**Files:**
- Generate outside worktree first: canonical A/B temporary directories.
- Generate after comparison: `docs/evaluation/offline/document-pipeline/2026-08-06-canonical-a/**`
- Generate after comparison: `docs/evaluation/offline/document-pipeline/2026-08-06-canonical-b/**`
- Create: `docs/evaluation/offline/document-pipeline/2026-08-06-results-summary.md`
- Create after artifact commit: `docs/evaluation/offline/document-pipeline/2026-08-06-PROVENANCE.json`

- [ ] **Step 1: Independently review the frozen inputs and implementation**

Confirm the inventory matches exactly 17 selected PDF hashes totaling 97,006,698 bytes, the synthetic strings/hashes were frozen before production measurement, no content/path leaked, and all specification requirements map to tests.

- [ ] **Step 2: Run fresh focused and adjacent verification**

Run every new test plus the existing 146-test offline evaluation suite. Commit any source fixes before measurement. Require clean status and record the source SHA.

- [ ] **Step 3: Generate canonical A outside the worktree**

From `aquillm/`, use a new explicit system temporary directory and the quoted local corpus path:

```powershell
python -m apps.chat.evals.run_offline_evidence document-run --real-corpus "$env:AQUILLM_LOCAL_ASTRO_CORPUS" --inventory apps/chat/evals/offline/document_corpus_inventory.yaml --output "<TEMP>\canonical-a" --sweeps 30 --memory-repeats 3
python -m apps.chat.evals.run_offline_evidence document-validate "<TEMP>\canonical-a"
```

Require exit 0, 17 real rows, 4 synthetic rows, zero observed connection attempts, and exact artifact membership.

- [ ] **Step 4: Generate B and compare deterministic results**

Run from the identical source/config into a distinct directory. Validate B, run CLI comparison, and independently compare normalized bytes/hashes. Copy neither directory unless comparison succeeds.

- [ ] **Step 5: Inspect all failures and scaling behavior**

Retain real-corpus parser failures and diagnose by safe enum without altering corpus membership. Do not tune fixtures from outputs. If a runner bug is found, add a RED regression, create a new source commit, and regenerate both runs from scratch.

- [ ] **Step 6: Copy immutable artifacts byte-for-byte and generate the paper summary**

Require both repository destinations not to exist, then copy A and B non-overwriting into `docs/evaluation/offline/document-pipeline/2026-08-06-canonical-a/` and `...-b/`. Re-run `document-validate` on both copied directories and `document-compare` between each copied directory and its corresponding temporary source; compare the relative-file/raw-SHA-256 maps to prove all 13 files are byte-identical to the validated temporary outputs.

Generate `2026-08-06-results-summary.md` exclusively from the copied canonical sources. Report fixed-corpus totals, success rate, pages, MiB, code points, estimated tokens, chunks, coverage/overlap, local real-corpus median/p95, effective and success-conditioned rates, synthetic scaling, memory, environment, and excluded claims. Use observed totals, not representative/generalized language. The table title must be exactly `Local document preprocessing measurements`; its caption and summary must explicitly say fixed convenience corpus, estimated tokens, warm single-process in-memory local preprocessing, and `zero connection attempts observed through the configured process-local socket guard`, and must exclude full ingestion/indexing/retrieval, concurrency, end-to-end response latency, RSS, and GPU memory.

- [ ] **Step 7: Validate and commit the copied artifacts**

Validate both copied directories, compare normalized results, regenerate `paper-table.md` and the paper-summary table byte-for-byte, mechanically assert every required title/qualification/guard phrase and every caption exclusion, recompute all totals/rates/nearest-rank quantiles from raw rows, verify `COMPLETE`, and run privacy scans plus all focused tests. Then commit the artifact directories and summary and capture the resulting exact artifact SHA:

```powershell
git add docs/evaluation/offline/document-pipeline .gitattributes
git commit -m "docs(eval): publish document preprocessing evidence"
$artifactCommit = git rev-parse HEAD
```

- [ ] **Step 8: Archive-verify the artifact commit, then generate provenance**

Export `git archive $artifactCommit` into a new explicit system temporary directory, extract it, and from that archive re-run both validators, normalized A/B comparison, table/summary byte regeneration, raw `COMPLETE` checks, total/rate/quantile recomputation, and privacy scans. The archived files—not the mutable worktree—are authoritative for provenance.

From the worktree's `aquillm/` directory, generate provenance using the canonical-A aggregate and repository root inside the extracted `$artifactCommit` archive; only the resulting provenance file is written back into the worktree. Validate lineage, artifact hashes, source/config/corpus hashes, and privacy against that same immutable archive:

```powershell
python -m apps.chat.evals.run_offline_evidence document-provenance --aggregate "$archiveRoot\repo\docs\evaluation\offline\document-pipeline\2026-08-06-canonical-a\aggregate.json" --repository "$archiveRoot\repo" --artifact-commit $artifactCommit --output ..\docs\evaluation\offline\document-pipeline\2026-08-06-PROVENANCE.json
python -m apps.chat.evals.run_offline_evidence document-provenance --validate ..\docs\evaluation\offline\document-pipeline\2026-08-06-PROVENANCE.json --repository "$archiveRoot\repo"
```

- [ ] **Step 9: Commit provenance and push from a clean worktree**

```powershell
git add docs/evaluation/offline/document-pipeline/2026-08-06-PROVENANCE.json
git commit -m "docs(eval): record document benchmark provenance"
git diff --check
git status --short
git push origin feat/aquillm-evaluation-framework
```

Require `git status --short` to be empty before push and confirm local and remote heads match exactly afterward.
