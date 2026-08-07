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
- Create `aquillm/apps/chat/evals/run_document_pipeline.py`: Windows-compatible standalone CLI with offline Django bootstrap.
- Create `aquillm/apps/chat/evals/offline/fixtures/document_corpus_inventory.yaml`: frozen 17-member hash/size inventory without paths or contents.
- Create `aquillm/apps/chat/evals/offline/fixtures/document_corpus_review.yaml`: independent inventory/synthetic-protocol review record.
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
def build_document_record(case: dict, outcome: dict) -> dict: ...
def aggregate_document_results(...) -> dict: ...

# document_pipeline_runner.py
def resolve_real_corpus(corpus_dir: Path, inventory: dict) -> list[dict]: ...
def run_document_case(case: dict, *, chunk_size: int, overlap: int) -> dict: ...
def run_document_benchmark(...) -> dict: ...

# document_pipeline_artifacts.py
def write_document_artifacts(result: dict, output_dir: Path) -> None: ...
def validate_document_artifacts(output_dir: Path) -> None: ...
def normalized_document_result(output_dir: Path) -> bytes: ...
def regenerate_document_table(aggregate_path: Path) -> str: ...
def write_document_provenance(aggregate_path: Path, artifact_commit: str, output_path: Path) -> None: ...
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

Patch only call counters. Require `extract_text_payloads("paper.pdf", ...)` to call type detection once, call the new PDF-only primary helper once, and append figures once under the default path. Require a benchmark-style call to `extract_primary_text_payload(..., ingest_type="document")` to execute the real `extract_pdf_text` without invoking the figure hook. Assert primary payload fields and text are identical, and prove that supplying `ingest_type` prevents re-detection. Keep existing parameterized tests over every non-PDF extension/type to prove their dispatch and figure policies do not change.

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
- Create: `aquillm/apps/chat/evals/offline/fixtures/document_corpus_inventory.yaml`
- Create: `aquillm/apps/chat/evals/offline/fixtures/document_corpus_review.yaml`
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

Define `sha256-utf8-lf-v1` exactly: decode source YAML as UTF-8 with an optional leading BOM removed, parse it with `yaml.safe_load`, serialize the selected semantic mapping with `yaml.safe_dump(sort_keys=True, default_flow_style=False, allow_unicode=True, line_break="\n")`, remove any trailing line breaks, append exactly one ASCII LF, encode UTF-8 without a BOM, then SHA-256 those bytes. The inventory hash covers the complete inventory mapping. The review file contains a dedicated `protocol` mapping, and the protocol hash covers only that mapping; it therefore excludes `protocol_hash`, `status`, reviewer identity/date/decisions, and other approval fields by construction. Inventory member hashes remain hashes of unmodified raw PDF bytes.

- [ ] **Step 4: Run inventory tests to prove GREEN**

```powershell
Set-Location aquillm
python -m pytest apps/chat/tests/test_offline_document_pipeline_schema.py -q
Set-Location ..
```

- [ ] **Step 5: Generate and freeze the 17-member inventory before measuring outputs**

Operator precondition: the private corpus must exist at `C:\Users\jackj\Github\Semantic Extraction Experiment\data\raw_docs\astro_test`. Run the exact API below from `aquillm/`; it hashes only direct-child PDFs and writes canonical YAML containing no path or filename. The function must abort before writing unless it observes exactly 17 PDFs totaling exactly 97,006,698 bytes. Do not execute production extraction yet.

```powershell
Set-Location aquillm
python -c "from pathlib import Path; import yaml; from apps.chat.evals.offline.document_pipeline_schema import freeze_document_inventory; p=Path(r'C:\Users\jackj\Github\Semantic Extraction Experiment\data\raw_docs\astro_test'); d=freeze_document_inventory(p, expected_count=17, expected_total_bytes=97006698); Path('apps/chat/evals/offline/fixtures/document_corpus_inventory.yaml').write_text(yaml.safe_dump(d, sort_keys=True, allow_unicode=True, line_break='\n'), encoding='utf-8', newline='\n')"
Set-Location ..
```

- [ ] **Step 6: Write RED deterministic minimal-PDF tests**

Require exact byte identity across two generation calls, exact page count, authored ASCII page strings in order, and exact expected post-sanitize/strip text `"\n\n".join(page_strings).strip()` at 1, 10, 50, and 100 pages. To test corruption, truncate the returned bytes before passing them to the parser/fixture validator; generation itself accepts only a positive page count and has no corruption mode.

- [ ] **Step 7: Run deterministic-PDF tests to prove RED**

Run the four generator/parser tests explicitly. Expected: FAIL because `generate_synthetic_pdf` is missing.

- [ ] **Step 8: Implement the minimal PDF builder and frozen expected hashes**

Build a PDF 1.4 file deterministically with ASCII only: object 1 catalog, object 2 pages tree, page objects in ascending order, one shared built-in Helvetica font object, and one content-stream object per page. Content streams use ASCII `BT /F1 12 Tf 72 720 Td (...) Tj ET`, byte-exact `/Length`, escaped literal delimiters, ten-digit zero-padded xref offsets, one free xref entry, `trailer /Size /Root`, exact `startxref`, and `%%EOF`. Headers, objects, stream separators, xref, trailer, `startxref`, and EOF use literal ASCII `b"\n"` only, never platform line endings. Page strings use four-digit numbering. Use no current time, random ID, hostname, path, downloaded font, or external generator. The tests construct authored expected page strings and expected normalized text independently of the generator return value. Store expected generated-byte and normalized-output hashes in the pending protocol.

- [ ] **Step 9: Run deterministic-PDF tests to prove GREEN**

Run the same four tests and require PASS before adding metric tests.

- [ ] **Step 10: Write RED absolute-metric tests**

Assert Unicode code points, UTF-8 bytes, whitespace words, actual production estimated-token function call, chunk counts, union coverage, excess overlap, ratio denominator, chunk summaries, exact output preimage hash, and empty/failure conventions.

- [ ] **Step 11: Run absolute-metric tests to prove RED**

Run the named metric and aggregation tests. Expected: FAIL because `build_document_record` and `aggregate_document_results` are missing.

- [ ] **Step 12: Implement and prove GREEN for absolute metrics**

Implement `build_document_record` and `aggregate_document_results` using integer absolute units, the production token estimator, explicit null failure conventions, union-of-half-open-spans coverage, excess overlap, and ratio-of-sums denominators. Re-run the complete schema test file and require PASS.

- [ ] **Step 13: Commit pending frozen inputs before review**

Write `document_corpus_review.yaml` with `status: pending_independent_review` and no claimed reviewer approval. Run schema tests with `allow_pending=True`, confirm normal loading rejects it, and commit the inventory, pending protocol, schema implementation, and tests. No production parser or benchmark run is allowed yet.

- [ ] **Step 14: Obtain and record genuinely independent approval**

Dispatch a fresh reviewer that did not author the inventory. Give it read-only access to the 17 local PDFs, pending inventory/protocol, generator tests, and approved spec. It verifies exact PDF membership/total, raw hashes/sizes, absence of private fields, authored strings, generated hashes, sensitivity/license wording, and that production functions have not been measured. The reviewer returns a stable task identifier and decisions. The coordinator records that returned identity/decision and `status: approved`, then re-dispatches the same reviewer to verify the exact approved bytes and canonical hashes. The implementer must not self-approve.

- [ ] **Step 15: Run tests and commit approved frozen inputs**

```powershell
Set-Location aquillm
python -m pytest apps/chat/tests/test_offline_document_pipeline_schema.py -q
Set-Location ..
git add aquillm/apps/chat/evals/offline/document_pipeline_schema.py aquillm/apps/chat/evals/offline/fixtures/document_corpus_inventory.yaml aquillm/apps/chat/evals/offline/fixtures/document_corpus_review.yaml aquillm/apps/chat/tests/test_offline_document_pipeline_schema.py
git commit -m "test(eval): freeze offline document benchmark inputs"
```

Do not run the canonical benchmark before this commit and its independent review.

## Chunk 2: Measurement runner and artifact contracts

### Task 3: Implement timing, memory, failure, and aggregate measurements

**Files:**
- Create: `aquillm/apps/chat/evals/offline/document_pipeline_runner.py`
- Create: `aquillm/apps/chat/tests/test_offline_document_pipeline_runner.py`

- [ ] **Step 1: Write RED production-execution tests**

First test `resolve_real_corpus`: discover direct-child files only, match `.pdf` case-insensitively, ignore non-PDF sidecars, default `allow_unlisted_pdfs=False`, reject unlisted PDFs, missing inventory hashes, duplicate file hashes, and size/hash mismatch, and return only stable case IDs plus in-memory bytes with path-free diagnostic codes. Patch call counters only and require the runner to execute detection, production primary extraction, exact sanitize/strip, production chunk planner, and production token estimator. Assert figures, database, embeddings, and network are never called.

- [ ] **Step 2: Write RED stage/failure tests**

Use a deterministic fake clock to prove nested stage units, direct outer `combined_ns`, nullable unexecuted stages, terminal-stage enum, populated input/page denominators on failures, safe diagnostic mapping, and synthetic failure as an integrity error.

- [ ] **Step 3: Implement one-case execution**

Keep metric calculation outside the direct outer timer. Return a static record and timing observation from one execution without document content or raw exception strings.

- [ ] **Step 4: Write RED 30-sweep tests**

Assert one warm-up, exactly 30 observations per case, rotating order, 630 timing-case rows, 60 timing-sweep rows, `case_combined_sum_ns`, successful-only denominator, effective and success-conditioned rates, nearest-rank p95 over sweep values, and no pooled pseudo-replication.

- [ ] **Step 5: Implement sweeps and aggregation**

Use integer nanoseconds as the source. Regenerate sweep rows from case rows and static records. Every rate is ratio-of-sums and null on zero denominator.

- [ ] **Step 6: Write RED memory tests**

Patch `tracemalloc` and GC to prove three separate memory-only passes per case, tracing starts after input allocation, timing code never traces, failed peaks remain separate, and per-case/arm maxima follow the specification.

- [ ] **Step 7: Implement memory passes and manifest**

Record non-secret machine context, fixed chunk config, source/corpus/config hashes, timer resolution, repetitions, dependency versions, and process-local network audit wording. Activate the guard around generation through validation and fail on any observed attempt.

- [ ] **Step 8: Run focused tests and commit**

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
- Create: `aquillm/apps/chat/evals/run_document_pipeline.py`
- Create: `aquillm/apps/chat/tests/test_offline_document_pipeline_artifacts.py`
- Modify: `README.md`

- [ ] **Step 1: Write RED exact-membership and schema tests**

Require the 13 files in the specification, row cardinalities 17/4/630/60/63, exact required keys/types/units, valid foreign keys, canonical JSONL ordering, raw `COMPLETE` hashes, and immutable non-overwriting output.

- [ ] **Step 2: Write RED privacy and regeneration tests**

Reject usernames, hostnames, absolute paths, basenames, document titles/content, raw exceptions, credential fields/values, missing/extra files, inconsistent aggregates, and non-regenerating CSV/report/table output.

- [ ] **Step 3: Implement atomic artifact writing and validation**

Write to a sibling temporary directory, validate it, then atomically rename. CSVs derive only from static JSONL. Markdown derives only from aggregate/static sources. Failure-conditioned values render as `N/A`, never zero.

- [ ] **Step 4: Write RED normalized-comparison tests**

Assert the exact exclusion paths from the specification, require all identity/config/count/diagnostic/network fields identical, and prove a one-count mutation fails comparison while a timing-only mutation passes.

- [ ] **Step 5: Implement table and provenance contracts**

The table reports real/synthetic results separately and includes exclusions in its caption. Provenance records evaluated source, artifact commit, exact artifact hashes, corpus inventory hash, source/config hashes, and the source hash algorithm without claiming its own commit.

- [ ] **Step 6: Write RED Windows CLI subprocess tests**

Test `--help`, a one-page synthetic-only temporary run, validation/comparison/table/provenance, existing-output rejection, corrupt synthetic failure, dirty-source rejection for canonical real runs, and startup with no external services or ambient credentials.

- [ ] **Step 7: Implement CLI and README instructions**

Subcommands: `run`, `validate`, `compare`, `table`, and `provenance`. `run` accepts `--real-corpus`, `--inventory`, `--output`, `--sweeps`, `--memory-repeats`, and test-only `--synthetic-only`. Canonical mode requires 30/3 and a clean source.

- [ ] **Step 8: Run all new and adjacent tests and commit**

```powershell
Set-Location aquillm
python -m pytest apps/ingestion/tests/test_primary_text_extraction.py apps/documents/tests/test_text_chunk_plan.py apps/chat/tests/test_offline_document_pipeline_schema.py apps/chat/tests/test_offline_document_pipeline_runner.py apps/chat/tests/test_offline_document_pipeline_artifacts.py -q
python -m pytest apps/ingestion/tests/test_unified_ingestion_parsers.py apps/documents/tests/test_multimodal_chunk_position_uniqueness.py apps/chat/tests/test_offline_evidence_metrics.py apps/chat/tests/test_offline_evidence_schema.py apps/chat/tests/test_offline_evidence_runner.py -q
Set-Location ..
git diff --check
git add aquillm/apps/chat/evals/offline/document_pipeline_artifacts.py aquillm/apps/chat/evals/run_document_pipeline.py aquillm/apps/chat/tests/test_offline_document_pipeline_artifacts.py README.md
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
python -m apps.chat.evals.run_document_pipeline run --real-corpus "<LOCAL_ASTRO_CORPUS>" --inventory apps/chat/evals/offline/fixtures/document_corpus_inventory.yaml --output "<TEMP>\canonical-a" --sweeps 30 --memory-repeats 3
python -m apps.chat.evals.run_document_pipeline validate "<TEMP>\canonical-a"
```

Require exit 0, 17 real rows, 4 synthetic rows, zero observed connection attempts, and exact artifact membership.

- [ ] **Step 4: Generate B and compare deterministic results**

Run from the identical source/config into a distinct directory. Validate B, run CLI comparison, and independently compare normalized bytes/hashes. Copy neither directory unless comparison succeeds.

- [ ] **Step 5: Inspect all failures and scaling behavior**

Retain real-corpus parser failures and diagnose by safe enum without altering corpus membership. Do not tune fixtures from outputs. If a runner bug is found, add a RED regression, create a new source commit, and regenerate both runs from scratch.

- [ ] **Step 6: Copy immutable artifacts and generate the paper summary**

Report fixed-corpus totals, success rate, pages, MiB, code points, estimated tokens, chunks, coverage/overlap, local real-corpus median/p95, effective and success-conditioned rates, synthetic scaling, memory, environment, and excluded claims. Use observed totals, not representative/generalized language.

- [ ] **Step 7: Fresh commit-level verification**

Validate both copied directories, compare normalized results, regenerate the paper table byte-for-byte, recompute all totals/rates/quantiles from raw rows, verify `COMPLETE`, run privacy scans, run all focused tests, export `git archive HEAD`, and repeat validation/hash checks from the archive.

- [ ] **Step 8: Commit artifacts, then provenance**

```powershell
git add docs/evaluation/offline/document-pipeline .gitattributes
git commit -m "docs(eval): publish document preprocessing evidence"
```

Generate provenance with that artifact commit, validate its lineage, then commit separately:

```powershell
git add docs/evaluation/offline/document-pipeline/2026-08-06-PROVENANCE.json
git commit -m "docs(eval): record document benchmark provenance"
```

- [ ] **Step 9: Push the requested branch**

```powershell
git push origin feat/aquillm-evaluation-framework
```

Confirm local and remote heads match exactly.
