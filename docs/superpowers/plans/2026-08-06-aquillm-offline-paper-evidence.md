# AquiLLM Offline Paper Evidence Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a network-blocked offline benchmark that produces traceable component-level data and paper-ready tables from actual deterministic AquiLLM code.

**Architecture:** A small evaluation package owns schema validation, metric computation, controlled baseline policies, artifact generation, and network denial. Frozen YAML fixtures remain separate from production code. The runner calls existing routing, query, evidence-packing, and memory-normalization functions; raw JSONL is the source of aggregate JSON, CSV, and Markdown artifacts.

**Tech Stack:** Python 3.12, pytest, PyYAML, standard-library `json`, `hashlib`, `statistics`, `time`, `socket`, and existing AquiLLM services.

---

## File Map

- Create `aquillm/apps/chat/evals/offline/__init__.py`: package exports only.
- Create `aquillm/apps/chat/evals/offline/schema.py`: fixture loading, validation, canonical serialization, and SHA-256 helpers.
- Create `aquillm/apps/chat/evals/offline/metrics.py`: confusion counts, exact-set metrics, evidence metrics, and aggregation with explicit applicability.
- Create `aquillm/apps/chat/evals/offline/policies.py`: sequential evidence baseline using production-equivalent token and stopping semantics.
- Create `aquillm/apps/chat/evals/offline/network.py`: scoped socket-denial guard and attempt counter.
- Create `aquillm/apps/chat/evals/offline/runner.py`: production-function execution, timing, test-manifest execution, aggregation, and artifact writing.
- Create `aquillm/apps/chat/evals/run_offline_evidence.py`: CLI entry point.
- Create `aquillm/apps/chat/evals/offline/fixtures/routing.yaml`: frozen routing/query/orchestration cases.
- Create `aquillm/apps/chat/evals/offline/fixtures/evidence.yaml`: controlled evidence-packing cases.
- Create `aquillm/apps/chat/evals/offline/fixtures/memory.yaml`: heuristic canonical-output cases.
- Create `aquillm/apps/chat/evals/offline/fixtures/ANNOTATION_RUBRIC.md`: label ontology, ambiguity rules, and examples written before scoring.
- Create `aquillm/apps/chat/evals/offline/fixtures/review.yaml`: independent-review identity/role, date, decisions, and adjudication record.
- Create `aquillm/apps/chat/evals/offline/test_manifest.yaml`: exact offline test paths and exclusions/prerequisites.
- Create `aquillm/apps/chat/tests/test_offline_evidence_metrics.py`: scorer and baseline tests.
- Create `aquillm/apps/chat/tests/test_offline_evidence_schema.py`: fixture and canonicalization tests.
- Create `aquillm/apps/chat/tests/test_offline_evidence_runner.py`: network guard, production calls, artifact and reproducibility tests.
- Generate `docs/evaluation/offline/<run-id>/`: manifest, JSONL, aggregate JSON, CSV, report, and mechanically generated paper table.

## Public Interfaces and Data Contracts

The implementation must expose these small, testable interfaces:

```python
# schema.py
def load_dataset(path: Path, kind: str) -> dict: ...
def validate_dataset(data: dict, kind: str) -> None: ...
def canonical_json_bytes(value: object) -> bytes: ...
def sha256_file(path: Path) -> str: ...

# metrics.py
def binary_metrics(expected: list[bool], actual: list[bool]) -> dict: ...
def exact_set_metrics(expected: list[list[str]], actual: list[list[str]]) -> dict: ...
def score_evidence_case(case: dict, selected: list[dict]) -> dict: ...
def aggregate_evidence(records: list[dict]) -> dict: ...
def compare_policies(records: list[dict], metric: str) -> dict: ...
def citation_diagnostics(selected: list[dict]) -> dict: ...
def memory_stratum_errors(records: list[dict]) -> dict: ...

# policies.py
def sequential_select(chunks: list[dict], token_budget: int) -> dict: ...

# network.py
@contextmanager
def deny_network() -> Iterator[NetworkAttempts]: ...

# runner.py
def run_component_evaluation(fixture_dir: Path, timing_repeats: int) -> dict: ...
def run_test_manifest(path: Path, project_root: Path) -> dict: ...
def write_artifacts(result: dict, output_dir: Path) -> None: ...
def validate_artifacts(output_dir: Path) -> None: ...
def normalized_reproducibility_bytes(result_path: Path) -> bytes: ...
def regenerate_paper_table(aggregate_path: Path) -> str: ...
def write_provenance(aggregate_path: Path, artifact_commit: str, output_path: Path) -> None: ...
```

Fixture envelopes use this schema:

```yaml
schema_version: "1.0"
dataset_id: routing-v1
frozen_at: "2026-08-06T00:00:00Z"
provenance: synthetic_public
rubric_version: "1.0"
review:
  status: approved
  record: review.yaml
cases: []
```

Every case contains `id`, `stratum`, `rationale`, and `gold`. Evidence candidates contain `evidence_id`, `doc_id`, `chunk_id`, `rank`, `text`, `citation`, optional `image_url`, and `relevant`. Test-manifest entries contain an exact pytest node ID, `status` (`included` or `prerequisite_blocked`), prerequisite, and reason.

Each completed output directory contains exactly these required artifacts:

```text
manifest.json
routing.jsonl
evidence.jsonl
memory.jsonl
timings.jsonl
tests.json
aggregate.json
routing.csv
evidence.csv
memory.csv
report.md
paper-table.md
COMPLETE
```

`manifest.json` requires `schema_version`, `run_id`, `timestamp_utc`, clean `source_commit`, `source_dirty=false`, fixture/code/config SHA-256 maps, canonical non-secret configuration, redacted environment, network-attempt count, and test-manifest hash. Each JSONL record requires `schema_version`, module, case ID, stratum, expected, actual, passed/conformant flag, and diagnostics. `aggregate.json` requires separate `routing`, `action`, `query`, `evidence`, `memory`, `tests`, `timing`, and `excluded_claims` objects. `tests.json` requires every manifest node with status and outcome. CSV files are flat projections of their JSONL sources. `paper-table.md` is a pure rendering of `aggregate.json`. `COMPLETE` contains the SHA-256 of every other required file. Unknown secret-bearing fields and missing/extra required files fail validation.

## Chunk 1: Scoring and Controlled Baseline

### Task 1: Implement executable metric definitions

**Files:**
- Create: `aquillm/apps/chat/evals/offline/__init__.py`
- Create: `aquillm/apps/chat/evals/offline/metrics.py`
- Create: `aquillm/apps/chat/evals/offline/policies.py`
- Test: `aquillm/apps/chat/tests/test_offline_evidence_metrics.py`

- [ ] **Step 1: Write failing tests for binary confusion metrics**

Test true/false positives and negatives, accuracy, precision, recall, F1, explicit numerator/denominator fields, and `not_applicable` when a denominator is zero.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `cd aquillm; python -m pytest apps/chat/tests/test_offline_evidence_metrics.py -q`

Expected: collection/import failure because the offline metrics package does not exist.

The initial test must use the intended public API:

```python
def test_binary_metrics_reports_counts_denominators_and_f1():
    result = binary_metrics([True, True, False, False], [True, False, True, False])
    assert result["confusion"] == {"tp": 1, "fn": 1, "fp": 1, "tn": 1}
    assert result["recall"] == {"status": "ok", "value": 0.5, "numerator": 1, "denominator": 2}
    assert result["f1"]["value"] == 0.5
```

- [ ] **Step 3: Implement minimal metric result types and binary scoring**

Use plain dictionaries suitable for canonical JSON. Never silently omit undefined metrics.

- [ ] **Step 4: Write failing tests for all case and aggregate scorers**

Test per-reason/action/query conformance with support counts; exact fact identity; duplicate removal and duplicate rate; memory false positives/negatives by stratum; case-level exact-set conformance; evidence macro/micro recall; zero-gold applicability; relevant-document coverage; distinct selected documents; estimated token use and overrun for both policies; citation syntax and doc/chunk consistency; duplicate/conflicting citations before packet deduplication; image-prefix behavior; and metric-specific win/tie/loss.

- [ ] **Step 5: Run and verify the new tests fail for missing behavior**

- [ ] **Step 6: Implement the minimal scorers**

Evidence identity is `evidence_id`; fact identity is the production-normalized exact string. Every aggregate includes support counts. Query comparison strips leading/trailing whitespace only. Classifier fields are multilabel binary metrics; `reason` and production action are separate single-label conformance matrices.

- [ ] **Step 7: Write failing tests for sequential policy equivalence**

Assert identical `_estimate_tokens` and `_chunk_text` behavior, candidate eligibility, stable order/tie handling, citation/image overhead accounting, first-oversized-chunk admission, later stop-at-budget behavior, selected metadata preservation, and overrun reporting.

- [ ] **Step 8: Implement the sequential baseline and verify GREEN**

Import the production `_estimate_tokens` and chunk-text helper rather than copying their formulas. Return an `EvidencePacket`-compatible record and explicit overrun tokens.

- [ ] **Step 9: Run focused tests and commit**

Run: `cd aquillm; python -m pytest apps/chat/tests/test_offline_evidence_metrics.py -q`

Commit: `feat(eval): add offline component scorers`

## Chunk 2: Frozen Fixtures and Validation

### Task 2: Create independently auditable case sets

**Files:**
- Create: `aquillm/apps/chat/evals/offline/schema.py`
- Create: `aquillm/apps/chat/evals/offline/fixtures/routing.yaml`
- Create: `aquillm/apps/chat/evals/offline/fixtures/evidence.yaml`
- Create: `aquillm/apps/chat/evals/offline/fixtures/memory.yaml`
- Create: `aquillm/apps/chat/evals/offline/fixtures/ANNOTATION_RUBRIC.md`
- Create: `aquillm/apps/chat/evals/offline/fixtures/review.yaml`
- Create: `aquillm/apps/chat/evals/offline/test_manifest.yaml`
- Test: `aquillm/apps/chat/tests/test_offline_evidence_schema.py`

- [ ] **Step 1: Write failing schema tests**

Require dataset version, frozen timestamp, provenance, author role, independent-review status, rubric version, unique case IDs, valid strata, required gold fields, evidence identity consistency, public/synthetic sensitivity, and exact test-manifest status/reason fields.

```python
def test_fixture_envelope_requires_approved_review_record(tmp_path):
    path = write_fixture(tmp_path, review={"status": "pending"})
    with pytest.raises(ValueError, match="approved independent review"):
        load_dataset(path, "routing")

def test_manifest_rejects_file_only_selector(tmp_path):
    manifest = manifest_with_node("apps/chat/tests/test_rag_intent.py")
    with pytest.raises(ValueError, match="exact pytest node id"):
        validate_test_manifest(manifest)
```

- [ ] **Step 2: Verify RED**

Run: `cd aquillm; python -m pytest apps/chat/tests/test_offline_evidence_schema.py -q`

- [ ] **Step 3: Implement validation and canonical hashing**

Canonical JSON uses UTF-8, sorted keys, compact separators, and newline termination. Hash fixture contents and directly exercised implementation files with SHA-256.

- [ ] **Step 4: Add fixtures before measuring production outputs**

Create at least:

- 60 routing cases balanced across retrieve/prompt-select/skip/local-tool actions and including paraphrase, ambiguous, and adversarial boundary strata;
- 24 evidence cases balanced across distributed evidence, single-source evidence, redundancy, distractors, tight/relaxed budgets, multimodal metadata, duplicates, and empty retrieval;
- 40 memory-helper cases spanning explicit remember directives, stable preferences/project facts, transient requests, vague references, duplicates, and prompt-like noise.

Gold labels must follow `ANNOTATION_RUBRIC.md` and include rationales. `review.yaml` records the second reviewer's role, every changed label, retained ambiguity, adjudication, approval, and the hashes reviewed. Do not run the canonical benchmark until fixtures and the review record are committed.

- [ ] **Step 5: Add the exact test manifest**

The initial manifest must enumerate exact nodes, including all tests in `apps/chat/tests/test_rag_intent.py`, `apps/chat/tests/test_rag_query.py`, `apps/chat/tests/test_rag_evidence.py`, `apps/chat/tests/test_rag_eval_runner.py`, and these exact nodes:

- `lib/tools/search/tests/test_vector_search_pack.py::VectorSearchPackTests::test_pack_includes_image_url_when_storage_has_image`
- `lib/tools/search/tests/test_vector_search_pack.py::VectorSearchPackTests::test_pack_omits_image_url_when_storage_missing_file`
- `lib/tools/search/tests/test_vector_search_pack.py::VectorSearchPackTests::test_pack_empty_results_explains_no_relevant_passages`
- `lib/memory/tests/test_stable_facts_quality.py::test_explicit_remember_directive_normalizes_to_durable_fact`
- `lib/memory/tests/test_stable_facts_quality.py::test_durable_project_tooling_statement_is_retained`
- `lib/memory/tests/test_stable_facts_quality.py::test_transient_tactical_turn_is_not_promoted_to_memory`
- `lib/memory/tests/test_stable_facts_quality.py::test_vague_self_referential_remember_text_is_filtered`
- `lib/memory/tests/test_mem0_search_isolation.py::test_parse_mem0_search_items_requires_matching_user_and_excludes_current_session`
- `tests/integration/test_architecture_import_boundaries.py::test_no_direct_aquillm_models_imports_in_runtime_modules`
- `tests/integration/test_settings_security_flags.py::test_celery_accept_content_excludes_pickle`
- `tests/integration/test_settings_security_flags.py::test_celery_tasks_do_not_force_pickle_serializer`

The fixture author expands the four whole-file groups into collected exact node IDs before commit. PostgreSQL nodes such as `apps/documents/tests/test_citation_api.py::test_citation_sources_groups_and_enforces_access` are explicit `prerequisite_blocked` entries unless a connection/preflight succeeds. Manifest execution invokes pytest once with the included node list and parses a generated JUnit XML file for exact counts; subprocess stdout/stderr are retained.

- [ ] **Step 6: Validate fixtures and commit**

Run: `cd aquillm; python -m pytest apps/chat/tests/test_offline_evidence_schema.py -q`

Commit: `test(eval): add frozen offline evidence fixtures`

## Chunk 3: Reproducible Runner and Artifacts

### Task 3: Execute production components without network access

**Files:**
- Create: `aquillm/apps/chat/evals/offline/network.py`
- Create: `aquillm/apps/chat/evals/offline/runner.py`
- Create: `aquillm/apps/chat/evals/run_offline_evidence.py`
- Test: `aquillm/apps/chat/tests/test_offline_evidence_runner.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing network-guard tests**

Prove outbound `socket.connect`, `socket.connect_ex`, and `socket.create_connection` attempts fail and increment a counter while local pure computation remains available. Restore patched functions on exit.

```python
def test_deny_network_blocks_and_counts_create_connection():
    with deny_network() as attempts:
        with pytest.raises(NetworkAccessError):
            socket.create_connection(("example.com", 443))
    assert attempts.total == 1
```

- [ ] **Step 2: Verify RED, implement guard, verify GREEN**

- [ ] **Step 3: Write failing runner tests for production-function execution**

Patch wrappers only to count calls; require actual `classify_chat_message`, `build_retrieval_query`, `build_evidence_packet`, memory normalizer/helper, `run_direct_rag_turn`, and `apps.chat.consumers.chat_receive._configure_append_tools` to execute. `run_direct_rag_turn` scores retrieve/prompt-select/skip reachability; `_configure_append_tools` scores retry tool reuse and local-tool selection. The memory fallback test names and calls `aquillm.memory.promote_profile_facts_for_turn`, replaces `requests.post` with an immediate controlled failure, replaces user lookup and `_promote_profile_facts` persistence with recorders, asserts the chosen explicit-remember or heuristic branch, and reports orchestration failure/fallback latency separately.

- [ ] **Step 4: Implement case runners and timing**

Reset `RAG_DIRECT_ENABLED`, `RAG_DIRECT_TOP_K`, `RAG_QUERY_REWRITE_ENABLED`, `RAG_EVIDENCE_TOKEN_BUDGET`, `RAG_MAX_SNIPPETS_PER_DOC`, `RAG_ATTACH_TOOLS_WHEN_COLLECTIONS_SELECTED`, and `TOOL_SEARCH_COMPACT_PAYLOAD` to declared canonical values and restore them on exit. Record the values in the manifest. Conformance runs execute once. Timing runs warm up, retain raw samples, and report median, nearest-rank p95, throughput, input size, and timer resolution. Never mix timing samples into conformance denominators.

- [ ] **Step 5: Write failing artifact tests**

Require atomic non-overwriting output, canonical JSONL, manifest hashes, redacted environment data, JUnit count parsing, schema validation, secret/path scanning, explicit unavailable counts, and a paper table generated only from aggregate JSON. The artifact schema names required files and fields; the secret scan rejects common credential patterns, the current username/hostname, drive-qualified absolute paths, and fixture sensitivity other than `synthetic_public`.

- [ ] **Step 6: Implement artifacts and CLI**

The CLI uses subcommands:

- `run --fixtures PATH --test-manifest PATH --output PATH --timing-repeats N [--skip-tests]`;
- `validate OUTPUT`;
- `compare OUTPUT_A OUTPUT_B`;
- `table AGGREGATE_JSON --output PAPER_TABLE`;
- `provenance AGGREGATE_JSON --artifact-commit SHA --output PROVENANCE_JSON`.

A completed output directory is immutable. Test execution uses `python -m pytest <exact nodes> --junitxml <temporary path>`. Frozen-case conformance misses are data and never change the process exit status. Only integrity failures—schema validation, network attempts, included test failures, secret/path scan, artifact validation, dirty source, or reproducibility mismatch—produce nonzero status.

- [ ] **Step 7: Write and pass two-run reproducibility test**

`normalized_reproducibility_bytes` removes only `run.timestamp_utc`, raw timing samples, and timing aggregates, then canonicalizes the remainder. Create two temporary output directories, compare normalized bytes, and verify the paper table regenerates byte-for-byte from aggregate JSON.

- [ ] **Step 8: Document the command and limitations, run focused tests, and commit**

Run: `cd aquillm; python -m pytest apps/chat/tests/test_offline_evidence_runner.py apps/chat/tests/test_offline_evidence_metrics.py apps/chat/tests/test_offline_evidence_schema.py -q`

Commit: `feat(eval): add reproducible offline evidence runner`

## Chunk 4: Canonical Run and Paper Data

### Task 4: Freeze source, execute twice, and publish traceable results

**Files:**
- Generate first outside worktree: two temporary canonical run directories
- Copy after comparison: `docs/evaluation/offline/2026-08-06-canonical-a/**`
- Copy after comparison: `docs/evaluation/offline/2026-08-06-canonical-b/**`
- Create: `docs/evaluation/offline/2026-08-06-results-summary.md`

- [ ] **Step 1: Run all new tests plus the declared offline manifest**

Confirm exact collected/passed/failed/skipped/unavailable counts and preserve output.

- [ ] **Step 2: Commit the clean evaluated source**

Commit any final code/test fixes before generating canonical data. Record this clean source SHA in both runs.

Exact preparation commands from the worktree root:

```powershell
git status --short
git rev-parse HEAD
$evalRunRoot = Join-Path ([System.IO.Path]::GetTempPath()) "aquillm-offline-20260806"
New-Item -ItemType Directory -Path $evalRunRoot -ErrorAction Stop
```

Expected: clean status, one source SHA, and a new empty temporary directory.

- [ ] **Step 3: Run canonical A under network denial**

Run from `aquillm/` to a new temporary directory outside the worktree with production-affecting settings reset by the runner:

```powershell
Set-Location aquillm
python -m apps.chat.evals.run_offline_evidence run --fixtures apps/chat/evals/offline/fixtures --test-manifest apps/chat/evals/offline/test_manifest.yaml --output "$evalRunRoot/canonical-a" --timing-repeats 200
python -m apps.chat.evals.run_offline_evidence validate "$evalRunRoot/canonical-a"
```

Expected: both commands exit 0, zero network attempts, and all required files plus `COMPLETE`.

- [ ] **Step 4: Run canonical B and compare deterministic outputs**

Use the same clean source commit and configuration in a second temporary directory outside the worktree. Compare normalized outputs byte-for-byte and report timing separately:

```powershell
python -m apps.chat.evals.run_offline_evidence run --fixtures apps/chat/evals/offline/fixtures --test-manifest apps/chat/evals/offline/test_manifest.yaml --output "$evalRunRoot/canonical-b" --timing-repeats 200
python -m apps.chat.evals.run_offline_evidence compare "$evalRunRoot/canonical-a" "$evalRunRoot/canonical-b"
Set-Location ..
New-Item -ItemType Directory -Path docs/evaluation/offline -Force
Copy-Item -Recurse -LiteralPath "$evalRunRoot/canonical-a" -Destination docs/evaluation/offline/2026-08-06-canonical-a
Copy-Item -Recurse -LiteralPath "$evalRunRoot/canonical-b" -Destination docs/evaluation/offline/2026-08-06-canonical-b
```

Expected: run and comparison exit 0 with `deterministic comparison: identical`; copy occurs only after that result.

- [ ] **Step 5: Inspect every failure and limitation**

Do not tune fixtures or hide negative cases. If implementation bugs are found, fix them with a failing test, create a new source commit, and rerun both canonical directories with a new run version.

- [ ] **Step 6: Write the results summary from generated artifacts**

Include routing/action/query conformance, evidence-policy macro/micro results and win/tie/loss, memory-helper canonical-output metrics, focused test counts, local timing, environment/configuration, and the exact excluded claims.

- [ ] **Step 7: Fresh verification**

Run all focused tests, artifact validators, two-run deterministic comparison, `git diff --check`, and secret/path scan. Read full output and report any unavailable prerequisite-dependent tests.

```powershell
Set-Location aquillm
python -m pytest apps/chat/tests/test_offline_evidence_metrics.py apps/chat/tests/test_offline_evidence_schema.py apps/chat/tests/test_offline_evidence_runner.py -q
python -m apps.chat.evals.run_offline_evidence validate ../docs/evaluation/offline/2026-08-06-canonical-a
python -m apps.chat.evals.run_offline_evidence validate ../docs/evaluation/offline/2026-08-06-canonical-b
python -m apps.chat.evals.run_offline_evidence compare ../docs/evaluation/offline/2026-08-06-canonical-a ../docs/evaluation/offline/2026-08-06-canonical-b
python -m apps.chat.evals.run_offline_evidence table ../docs/evaluation/offline/2026-08-06-canonical-a/aggregate.json --output ../docs/evaluation/offline/2026-08-06-canonical-a/paper-table.regenerated.md
Set-Location ..
git diff --check
```

Expected: tests and validators exit 0, comparison is identical, regenerated table is byte-identical to `paper-table.md`, and `git diff --check` prints nothing.

- [ ] **Step 8: Commit artifacts and push branch**

First commit code and copied artifacts with `docs(eval): publish offline component evidence`. Then record its SHA and generate provenance:

```powershell
git add docs/evaluation/offline
git commit -m "docs(eval): publish offline component evidence"
$artifactCommit = git rev-parse HEAD
Set-Location aquillm
python -m apps.chat.evals.run_offline_evidence provenance ../docs/evaluation/offline/2026-08-06-canonical-a/aggregate.json --artifact-commit $artifactCommit --output ../docs/evaluation/offline/2026-08-06-PROVENANCE.json
Set-Location ..
git add docs/evaluation/offline/2026-08-06-PROVENANCE.json
git commit -m "docs(eval): record offline evidence artifact provenance"
git push origin feat/aquillm-evaluation-framework
```

Expected: provenance records the already-existing artifact commit and source/hash lineage, never its own follow-up commit SHA; push exits 0.

Push: `git push origin feat/aquillm-evaluation-framework`
