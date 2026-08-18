# GLiNER2 Knowledge Graph and Hybrid Retrieval Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a provenance-first, incrementally maintained collection knowledge graph with GLiNER2 extraction, conservative entity resolution, permission-safe cross-collection canonical links, and fail-open graph expansion inside AquiLLM's existing hybrid RAG retrieval.

**Architecture:** Add a dedicated `apps.knowledge_graph` Django app backed by the existing PostgreSQL/pgvector database. GLiNER2 runs only in a dedicated Celery worker and emits immutable entity/relation mentions tied to `TextChunk` spans; deterministic document and collection resolvers promote those mentions into versioned collection graphs, while deployment-wide canonical entities connect equivalent collection nodes without merging their claims. Existing vector/trigram/exact retrieval remains primary: graph traversal expands its seed chunks within the caller's already-authorized document set, and the existing reranker produces the final chunk order.

**Tech Stack:** Python 3.12, Django 5, PostgreSQL 17 + pgvector, Celery + Redis, GLiNER2 local inference, existing AquiLLM embedding/reranking services, structlog, pytest/pytest-django, Docker Compose.

---

## Requirements and Source of Truth

This plan supersedes the implementation details in `docs/documents/architecture/2026-04-09-knowledge-graph-index-overlay-implementation-plan.md` while preserving the architectural principles in `docs/documents/architecture/2026-04-09-knowledge-graph-index-overlay-design.md`:

- vector retrieval remains primary;
- graph state is an asynchronous, rebuildable overlay;
- every promoted entity and relation retains chunk-level evidence;
- stale or failed graph state never blocks ingestion or chat;
- collection permissions constrain graph traversal before expansion;
- Mem0 remains conversation-memory infrastructure, not corpus-graph storage.

Additional requirements approved in the design discussion:

- GLiNER2 performs schema-driven entity and relation-mention extraction;
- coreference and entity resolution are separate, versioned AquiLLM stages;
- the first release resolves identifiers, names, aliases, and defined acronyms but does not automatically merge pronoun-only references;
- raw extraction evidence is retained even when an entity is suppressed or rejected;
- cross-collection deduplication uses persistent canonical links; any dictionary is a rebuildable cache, never the source of truth;
- claims stay collection-owned even when their endpoint identities are canonicalized;
- ontology definitions and type/relation constraints are versioned;
- LLM-generated ontology changes are proposals only and require explicit activation;
- the feature ships disabled by default.

GLiNER2 version note: pin the exact package and checkpoint validated during Chunk 2. The linked v1 paper does not establish production relation-extraction or coreference quality; do not use unpinned `latest` behavior as an acceptance criterion.

---

## Subagent Execution Strategy

Use one fresh implementation subagent per task. After each task, dispatch a requirements reviewer and then a code-quality reviewer before starting a dependent task. Do not give two active agents ownership of the same file.

| Gate | Subagent ownership | Prerequisites | May run in parallel with |
|---|---|---|---|
| 0 | Baseline/evaluation contracts | None | Nothing; contracts define later behavior |
| 1A | Django app and graph persistence | Gate 0 | 1B after interfaces are agreed |
| 1B | Optional GLiNER2 runtime packaging | Gate 0 | 1A |
| 2 | Extraction adapter and mention persistence | Gates 1A, 1B | None |
| 3A | Coreference and entity resolution | Gate 2 | 3B only after shared types freeze |
| 3B | Filtering and collection graph assembly | Gate 2 | 3A only after shared types freeze |
| 4 | Idempotent async build lifecycle | Gates 3A, 3B | None |
| 5 | Canonical links and cross-collection projection | Gate 4 | Operator docs after APIs freeze |
| 6 | Hybrid retrieval integration | Gate 5 | Eval fixture authoring |
| 7 | Quality gates, operations, and rollout | Gate 6 | None |

Every subagent brief must include:

1. the exact task section from this plan;
2. the current branch/worktree path;
3. the files it owns and must not modify outside that list without approval;
4. the focused verification command;
5. an instruction to preserve unrelated user changes;
6. an instruction not to enable the feature by default.

---

## Planned File Structure

### New knowledge-graph app

```text
aquillm/apps/knowledge_graph/
  __init__.py                    # App package marker; no optional runtime imports
  apps.py                         # Django app registration only
  models/
    artifacts.py                 # GraphArtifact and GraphBuildRun lifecycle
    ontology.py                  # OntologyVersion and activation state
    entities.py                  # EntityMention, DocumentEntity, CollectionEntity, CanonicalEntity
    relations.py                 # RelationMention, CollectionRelation, evidence links
    associations.py              # mention/entity/canonical resolution links
  extraction/
    windows.py                   # TextChunk windows, global span mapping, overlap dedupe
    pipeline.py                  # Extract and persist raw mention evidence
  resolution/
    normalization.py             # Deterministic label/identifier normalization
    coreference.py               # Conservative within-document mention clustering
    collection.py                # DocumentEntity -> CollectionEntity resolution
    canonical.py                 # CollectionEntity -> CanonicalEntity candidate/link logic
    scoring.py                   # Separate resolution and retrieval-utility scores
  graph/
    filtering.py                 # Active/suppressed/rejected decisions
    assembly.py                  # Versioned collection nodes/relations from resolved evidence
    invalidation.py              # Content, move, delete, ontology/model invalidation
  retrieval/
    types.py                     # GraphExpansionRequest/Result/Diagnostics
    expansion.py                 # Bounded, permission-scoped graph traversal
  tasks.py                       # Thin Celery entry points; no model import at module load
  services/builds.py             # Idempotent staged build orchestration
  services/inspection.py         # Bounded artifact/build health summaries
  services/pruning.py            # Retention and safe superseded-artifact pruning
  management/commands/
    rebuild_knowledge_graph.py   # Targeted document/collection/all rebuild enqueueing
    inspect_knowledge_graph.py   # Build/artifact health and counts
    check_knowledge_graph_extractor.py # Explicit pinned-runtime smoke check
    prune_knowledge_graph.py     # Dry-run-first retention enforcement
  ontologies/research-v1.yaml    # Checked-in initial schema and constraints
  evals/
    extraction_cases.yaml        # Gold mention/relation/resolution cases
    retrieval_cases.yaml         # Graph-specific retrieval cases
    run_kg_eval.py               # Offline metrics runner
  tests/                         # Focused unit/integration tests by responsibility

aquillm/lib/knowledge_graph/
  types.py                       # ORM-free immutable extraction contracts
  config.py                      # Environment-backed provider settings
  extractors/
    base.py                      # ExtractionBackend protocol
    factory.py                   # Lazy provider selection
    gliner2_local.py             # Lazy GLiNER2 import/model cache/result normalization
  tests/                         # Import-isolated provider tests
```

### Existing files intentionally modified

- `aquillm/aquillm/settings.py`: disabled-by-default graph settings and Celery task route.
- `aquillm/apps/documents/tasks/chunking.py`: one post-success enqueue seam covering normal and duplicate-content chunk paths.
- `aquillm/apps/documents/models/document.py`: targeted move/delete invalidation without synchronous graph work.
- `aquillm/apps/documents/services/chunk_search.py`: graph candidate expansion between candidate dedupe and existing reranking.
- `aquillm/apps/chat/services/rag_metrics.py`: optional graph retrieval diagnostics in structured events.
- `pyproject.toml` and `uv.lock`: pinned optional/runtime dependency decision.
- `pytest.ini`: include the new app and library test directories in default discovery.
- `deploy/docker/knowledge-graph/Dockerfile`: graph-worker-only GLiNER2 installation.
- `deploy/compose/development.yml` and `deploy/compose/production.yml`: dedicated `knowledge-graph` queue worker.
- `.env.example`: feature, model, queue, thresholds, and traversal limits.
- `docs/documents/operations/knowledge-graph-overlay-runbook.md`: rebuild, inspection, rollback, and rollout.

No graph visualization UI, external graph database, pronoun-level neural coreference, automatic LLM ontology activation, or Mem0 storage convergence belongs in this plan.

---

## Chunk 0: Freeze Contracts and Baselines

### Task 1: Add provider-neutral extraction and retrieval contracts

**Subagent:** `kg-contracts`

**Files:**
- Create: `aquillm/lib/knowledge_graph/__init__.py`
- Create: `aquillm/lib/knowledge_graph/types.py`
- Create: `aquillm/apps/knowledge_graph/retrieval/__init__.py`
- Create: `aquillm/apps/knowledge_graph/retrieval/types.py`
- Create: `aquillm/lib/knowledge_graph/tests/__init__.py`
- Create: `aquillm/lib/knowledge_graph/tests/test_contracts.py`

- [ ] **Step 1: Write failing dataclass contract tests**

Create immutable dataclasses with equality tests for the minimum stable boundary:

```python
EntityCandidate(
    entity_type="model",
    text="Qwen3",
    start=14,
    end=19,
    confidence=0.94,
)

RelationCandidate(
    relation_type="evaluates_on",
    head_text="Qwen3",
    tail_text="MMLU",
    head_start=14,
    head_end=19,
    tail_start=33,
    tail_end=37,
    confidence=0.88,
)

diagnostic = ExtractionDiagnostic(
    code="ambiguous_relation_endpoint",
    candidate_kind="relation",
    input_index=0,
    details=(("relation_type", "evaluates_on"), ("head_text", "Qwen3")),
)
ExtractionBatchResult(
    entities=(entity_candidate,),
    relations=(relation_candidate,),
    diagnostics=(diagnostic,),
)
GraphExpansionRequest(
    query="Which model uses MMLU?",
    seed_chunk_ids=(1,),
    allowed_doc_ids=(uuid_value,),
    allowed_collection_ids=(collection_uuid,),
)
GraphExpansionResult(chunk_ids=(2,), diagnostics=GraphExpansionDiagnostics(status="hit"))
```

`ExtractionBatchResult` includes `diagnostics: tuple[ExtractionDiagnostic, ...]`. Diagnostics must be immutable, provider-neutral rejected-output evidence with a stable code, candidate kind, input index, and a tuple of scalar key/value details; do not store a provider object or mutable mapping. This is the contract Task 6 uses for malformed spans, unknown types, and ambiguous raw relation endpoints. It may retain extracted private surface text for build audit, but callers must never copy those details into operational logs or public retrieval diagnostics.

Require offsets to be half-open (`start <= offset < end`), confidence in `[0, 1]`, tuples rather than mutable lists, scalar-only diagnostic details, and no Django imports in either types module.

- [ ] **Step 2: Run the contract test and confirm failure**

Run: `python -m pytest aquillm/lib/knowledge_graph/tests/test_contracts.py -q`

Expected: FAIL because the contract modules do not exist.

- [ ] **Step 3: Implement only the immutable dataclasses and validation**

Use `@dataclass(frozen=True, slots=True)` and `__post_init__` validation. Do not add database or GLiNER2 imports.

- [ ] **Step 4: Run the contract test**

Run: `python -m pytest aquillm/lib/knowledge_graph/tests/test_contracts.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aquillm/lib/knowledge_graph aquillm/apps/knowledge_graph/retrieval
git commit -m "test(kg): define extraction and retrieval contracts"
```

### Task 2: Establish gold fixtures and a baseline eval runner

**Subagent:** `kg-eval-baseline`

**Files:**
- Create: `aquillm/apps/knowledge_graph/evals/__init__.py`
- Create: `aquillm/apps/knowledge_graph/evals/extraction_cases.yaml`
- Create: `aquillm/apps/knowledge_graph/evals/retrieval_cases.yaml`
- Create: `aquillm/apps/knowledge_graph/evals/run_kg_eval.py`
- Create: `aquillm/apps/knowledge_graph/tests/test_eval_runner.py`
- Reference: `aquillm/apps/chat/evals/rag_cases.yaml`
- Reference: `aquillm/apps/chat/evals/run_rag_eval.py`

- [ ] **Step 1: Write a failing eval-loader test**

Require at least these adversarial fixtures:

- acronym definition and later full-name mention;
- same acronym with two meanings;
- same model name across two documents;
- two similar but distinct model versions;
- relation endpoint at a chunk overlap;
- publisher/boilerplate entity retained as suppressed evidence;
- conflicting claims in two collections;
- inaccessible collection evidence excluded from retrieval;
- vector seed in collection A expanding through a canonical entity to collection B.

- [ ] **Step 2: Run the loader test and confirm failure**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_eval_runner.py -q`

Expected: FAIL because the eval loader does not exist.

- [ ] **Step 3: Implement deterministic fixture loading and metric shells**

The runner must report, without a live LLM:

```text
entity_precision entity_recall relation_precision relation_recall
auto_link_precision suppression_precision retrieval_recall_at_10
```

At this stage, accept injected/mock predictions and existing retrieval outputs. Do not call GLiNER2 or the database.

- [ ] **Step 4: Record the vector-only retrieval baseline**

Add a `--baseline-only` mode that records the existing chunk retrieval result IDs for graph-specific retrieval fixtures where database fixtures are available. Missing database fixtures must produce `SKIP`, not fabricated scores.

- [ ] **Step 5: Run the eval tests**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_eval_runner.py aquillm/apps/chat/tests/test_rag_eval_runner.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add aquillm/apps/knowledge_graph/evals aquillm/apps/knowledge_graph/tests/test_eval_runner.py
git commit -m "test(kg): add extraction and retrieval eval fixtures"
```

---

## Chunk 1: Persistence, Ontology, and Runtime Isolation

### Task 3: Add the Django app and versioned graph persistence

**Subagent:** `kg-persistence`

**Files:**
- Create: `aquillm/apps/knowledge_graph/__init__.py`
- Create: `aquillm/apps/knowledge_graph/apps.py`
- Create: `aquillm/apps/knowledge_graph/models/__init__.py`
- Create: `aquillm/apps/knowledge_graph/models/artifacts.py`
- Create: `aquillm/apps/knowledge_graph/models/ontology.py`
- Create: `aquillm/apps/knowledge_graph/models/entities.py`
- Create: `aquillm/apps/knowledge_graph/models/relations.py`
- Create: `aquillm/apps/knowledge_graph/models/associations.py`
- Create: `aquillm/apps/knowledge_graph/migrations/__init__.py`
- Create: `aquillm/apps/knowledge_graph/migrations/0001_initial.py`
- Create: `aquillm/apps/knowledge_graph/tests/test_models.py`
- Modify: `aquillm/aquillm/settings.py`

- [ ] **Step 1: Write failing app and model tests**

Test these invariants:

- `apps.knowledge_graph` is registered;
- at most one active `GraphArtifact` per logical `(scope_type, scope_id)` via a conditional uniqueness constraint on `status="active"`;
- `(scope_type, scope_id, source_hash, ontology_version, extractor_version, resolver_version, filter_policy_version)` is the immutable build/idempotency identity, not an additional active-artifact allowance;
- `EntityMention` always has `document_id`, `chunk`, `start`, `end`, explicit `position_basis`, raw text, type, and extraction confidence; text evidence requires document-global offsets, while image/figure evidence requires chunk-local offsets plus figure/content-object provenance;
- `DocumentEntityMention` links mentions to one document entity without deleting raw mentions;
- `CollectionEntityDocumentLink` links document entities to collection entities with score, method, resolver version, and status;
- `CanonicalEntityLink` links collection entities to canonical entities with the same audit fields;
- `RelationMention` points to head/tail mentions and evidence chunk;
- `CollectionRelationEvidence` preserves each supporting relation mention;
- active/suppressed/rejected/superseded states are explicit choices, not row deletion;
- relations are unique per `(artifact, source_entity, relation_type, target_entity)`;
- graph evidence cascades with a deleted `TextChunk`, while build runs retain terminal statistics.

- [ ] **Step 2: Run model tests and confirm failure**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_models.py aquillm/tests/integration/test_architecture_import_boundaries.py -q`

Expected: FAIL because the app/models are absent.

- [ ] **Step 3: Implement focused model files and indexes**

Use existing database types and `pgvector.django.VectorField(dimensions=1024)` only for resolved collection/canonical entity embeddings. Index:

- artifact scope/status/source hash;
- mention `(document_id, chunk_id)` and normalized text/type;
- collection entity `(collection_id, entity_type, normalized_label)`;
- canonical entity `(entity_type, normalized_label)`;
- relation artifact/source/target/type;
- association status and target IDs.

Do not add a generic JSON-only node table. Keep free-form extractor details in `metadata`, but store lifecycle, identity, confidence, and provenance in typed columns.

Enforce a conditional database uniqueness constraint for one `active` artifact per logical scope. Multiple `building`, `failed`, `stale`, and `superseded` artifacts may coexist for audit/retry purposes, but activation must replace—not coexist with—the prior active artifact.

- [ ] **Step 4: Generate and inspect the migration**

Run from `aquillm/`:

```bash
python manage.py makemigrations apps_knowledge_graph
python manage.py sqlmigrate apps_knowledge_graph 0001
```

Expected: bounded indexed tables with no external graph extension and no destructive operation on document tables.

- [ ] **Step 5: Run focused tests and Django checks**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_models.py aquillm/tests/integration/test_architecture_import_boundaries.py -q`

Run from `aquillm/`: `python manage.py check`

Expected: PASS and no Django check errors.

- [ ] **Step 6: Commit**

```bash
git add aquillm/apps/knowledge_graph aquillm/aquillm/settings.py
git commit -m "feat(kg): add versioned graph persistence"
```

### Task 4: Add a versioned research ontology and policy loader

**Subagent:** `kg-ontology`

**Files:**
- Create: `aquillm/apps/knowledge_graph/ontologies/research-v1.yaml`
- Create: `aquillm/apps/knowledge_graph/services/ontology.py`
- Create: `aquillm/apps/knowledge_graph/tests/test_ontology.py`
- Modify: `aquillm/apps/knowledge_graph/models/ontology.py`

- [ ] **Step 1: Write failing ontology validation tests**

The checked-in schema must define entity types, descriptions, aliases, default retrieval weights, default suppression policy, relation descriptions, direction, allowed head types, and allowed tail types. Start with:

```text
paper author institution method model dataset metric task
claim finding figure software
```

and relations:

```text
authored_by cites uses_method uses_model uses_dataset evaluates_on
measures_with reports_metric supports contradicts compares_with
shown_in_figure implemented_by affiliated_with
```

Reject duplicate names, undefined endpoint types, directionless relations, invalid thresholds, and schema mutation after activation.

- [ ] **Step 2: Run ontology tests and confirm failure**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_ontology.py -q`

Expected: FAIL because the loader does not exist.

- [ ] **Step 3: Implement checksum-based loading and activation**

`load_ontology(path) -> OntologyDefinition` must return immutable provider-neutral types. `activate_ontology(definition)` persists the YAML, semantic version, and SHA-256 checksum in `OntologyVersion`; it must not call an LLM.

- [ ] **Step 4: Test collection-specific extension merging**

Support a future extension as a versioned delta, but require it to preserve core type/relation meanings. Conflicting redefinitions must fail validation rather than silently replace the core.

- [ ] **Step 5: Run tests**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_ontology.py aquillm/apps/knowledge_graph/tests/test_models.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add aquillm/apps/knowledge_graph/ontologies aquillm/apps/knowledge_graph/services/ontology.py aquillm/apps/knowledge_graph/models/ontology.py aquillm/apps/knowledge_graph/tests/test_ontology.py
git commit -m "feat(kg): add versioned research ontology"
```

### Task 5: Package GLiNER2 as an optional, import-isolated worker dependency

**Subagent:** `kg-runtime-isolation`

**Files:**
- Create: `aquillm/lib/knowledge_graph/config.py`
- Create: `aquillm/lib/knowledge_graph/extractors/__init__.py`
- Create: `aquillm/lib/knowledge_graph/extractors/base.py`
- Create: `aquillm/lib/knowledge_graph/extractors/factory.py`
- Create: `aquillm/lib/knowledge_graph/tests/test_factory.py`
- Create: `aquillm/tests/integration/test_knowledge_graph_import_isolation.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `pytest.ini`

- [ ] **Step 1: Write failing configuration and import-isolation tests**

The provider-neutral `ExtractionBackend` protocol must expose:

```python
def extract_batch(
    self,
    texts: tuple[str, ...],
    *,
    ontology: OntologyDefinition,
) -> tuple[ExtractionBatchResult, ...]:
    raise NotImplementedError
```

In a subprocess, install an import hook that raises for `gliner2`, `torch`, `transformers`, and `peft`, then import Django settings, ASGI, KG models, and task registration. All imports must succeed when graph builds are disabled.

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `python -m pytest aquillm/lib/knowledge_graph/tests/test_factory.py aquillm/tests/integration/test_knowledge_graph_import_isolation.py -q`

Expected: FAIL because config/factory modules and dependency separation do not exist.

- [ ] **Step 3: Add the optional dependency group**

Add an exact package pin under `[project.optional-dependencies]`, using the GLiNER2 version verified when this task executes:

```toml
knowledge-graph-local = [
    "gliner2[local]==1.3.2",
]
```

Verify `gliner2[local]==1.3.2` against AquiLLM's locked `torch`, `transformers`, `tokenizers`, and `peft`. Pin checkpoint `fastino/gliner2-base-v1` to Hugging Face commit `8437ba583a733d87f56ae902f3b197934eedd58e`; record both identities in the commit body and runbook. If compatibility fails, stop this task and revise the plan rather than silently selecting another version. Regenerate `uv.lock`; do not hand-edit it and do not add the heavy extra to `requirements.txt` or default web installs.

- [ ] **Step 4: Implement lazy backend selection**

`factory.get_extraction_backend()` must import the configured provider with `importlib.import_module()` only when called. No app, model, task, or package `__init__.py` may re-export the GLiNER2 provider class.

- [ ] **Step 5: Add default-off settings parsing**

Expose provider-neutral getters for `KG_BUILD_ENABLED`, provider name, model/checkpoint revision, device, batch size, cache directory, local-files-only, and fail-open. Validate unsafe missing model revisions when build enablement is true.

- [ ] **Step 6: Add new test directories to pytest discovery**

Add `aquillm/apps/knowledge_graph/tests` and `aquillm/lib/knowledge_graph/tests` to `pytest.ini::testpaths` so plain `pytest` cannot silently skip the feature.

- [ ] **Step 7: Run tests and dependency validation**

Run: `python -m pytest aquillm/lib/knowledge_graph/tests/test_factory.py aquillm/tests/integration/test_knowledge_graph_import_isolation.py -q`

Run: `uv lock --check`

Expected: PASS; default environment imports without the optional extra.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock pytest.ini aquillm/lib/knowledge_graph aquillm/tests/integration/test_knowledge_graph_import_isolation.py
git commit -m "build(kg): isolate optional gliner2 runtime"
```

---

## Chunk 2: GLiNER2 Extraction and Provenance

### Task 6: Implement the pinned GLiNER2 backend behind the protocol

**Subagent:** `kg-gliner2-backend`

**Files:**
- Create: `aquillm/lib/knowledge_graph/extractors/gliner2_local.py`
- Create: `aquillm/lib/knowledge_graph/tests/test_gliner2_local.py`
- Modify: `aquillm/lib/knowledge_graph/extractors/factory.py`
- Modify: `aquillm/lib/knowledge_graph/config.py`
- Reference: `aquillm/lib/llm/optimizations/lm_lingua2_adapter.py`

- [ ] **Step 1: Write failing backend tests with a fake GLiNER2 module**

Verify:

- `huggingface_hub.snapshot_download` receives the exact model ID, immutable revision, cache directory, and `local_files_only` configuration;
- `GLiNER2.from_pretrained` receives the resolved local snapshot path and only arguments supported by pinned GLiNER2 `1.3.2`;
- one process loads the model at most once;
- entity and relation outputs normalize into the provider-neutral dataclasses with confidence and half-open spans;
- malformed spans, NaN confidence, unknown relation types, and disallowed endpoint types are rejected into diagnostics rather than persisted;
- empty inputs return empty results without loading the model;
- provider exceptions become `ExtractionBackendError` without importing Django.

The fake modules must assert the pinned production contract exactly: package `gliner2[local]==1.3.2`, model `fastino/gliner2-base-v1`, and revision `8437ba583a733d87f56ae902f3b197934eedd58e`. Tests must not contact Hugging Face.

- [ ] **Step 2: Run backend tests and confirm failure**

Run: `python -m pytest aquillm/lib/knowledge_graph/tests/test_gliner2_local.py -q`

Expected: FAIL because the provider is absent.

- [ ] **Step 3: Implement process-local lazy loading**

Import `snapshot_download` and `GLiNER2` inside `_load_model()`. Resolve the immutable checkpoint with `snapshot_download(repo_id=..., revision=..., cache_dir=..., local_files_only=...)`, then call `GLiNER2.from_pretrained(local_snapshot_path, ...)` with only arguments verified against GLiNER2 `1.3.2` (for example its supported device/map-location option). Do not pass Hugging Face `revision`, `cache_dir`, or `local_files_only` through to GLiNER2 if its public loader does not accept them. Protect first load with a process-local lock, cache the model, and never download or load it in a Django web request. Use the pinned public entity/relation extraction methods and normalize their outputs behind the provider-neutral backend contract.

- [ ] **Step 4: Normalize relation endpoints to spans**

Prefer upstream head/tail spans. When the provider returns text-only endpoints, match only an unambiguous extracted mention of a compatible type in the same input window. If multiple spans match, retain the raw relation in diagnostics but do not promote it.

- [ ] **Step 5: Run tests**

Run: `python -m pytest aquillm/lib/knowledge_graph/tests/test_gliner2_local.py aquillm/lib/knowledge_graph/tests/test_factory.py -q`

Expected: PASS without downloading a real checkpoint.

- [ ] **Step 6: Commit**

```bash
git add aquillm/lib/knowledge_graph
git commit -m "feat(kg): add lazy gliner2 extraction backend"
```

### Task 7: Map overlapping TextChunks into immutable mention evidence

**Subagent:** `kg-mention-extraction`

**Files:**
- Create: `aquillm/apps/knowledge_graph/extraction/__init__.py`
- Create: `aquillm/apps/knowledge_graph/extraction/windows.py`
- Create: `aquillm/apps/knowledge_graph/extraction/pipeline.py`
- Create: `aquillm/apps/knowledge_graph/tests/test_span_mapping.py`
- Create: `aquillm/apps/knowledge_graph/tests/test_mention_extraction.py`
- Reference: `aquillm/apps/documents/models/chunks.py`

- [ ] **Step 1: Write failing global-span and overlap-dedupe tests**

Cover:

- `global_start = chunk.start_position + local_start` and the analogous end;
- for text chunks, validation that the document `full_text` source slice equals the extracted surface text after only documented Unicode normalization;
- for image/figure chunks, offsets remain chunk-local with `position_basis="chunk_content"`; never validate synthetic image positions against `Document.full_text`;
- the same mention extracted from overlapping chunks becomes one `EntityMention` with two optional evidence observations or the highest-confidence observation plus provenance metadata;
- identical text at genuinely different document positions remains distinct;
- text and image chunks retain modality and figure/document identity;
- a real `DocumentFigure`/image `TextChunk` successfully extracts and persists entity plus relation evidence using the chunk-content position basis;
- relation endpoints map to the exact persisted mention rows;
- stale source hashes abort before writes.

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_span_mapping.py aquillm/apps/knowledge_graph/tests/test_mention_extraction.py -q`

Expected: FAIL because extraction orchestration is absent.

- [ ] **Step 3: Implement bounded batch window construction**

Use persisted `TextChunk` content and positions as the unit of work. Preserve existing chunk overlap; do not materialize an entire large document or invoke GLiNER2's whole-document API. Batch by configured count and character/token guard, retaining `(chunk_id, doc_id, start_position, modality, content_object_type, content_object_id)` for remapping. Text mentions use document-global positions; image/figure mentions use chunk-local positions plus figure provenance so the two coordinate systems cannot be confused.

- [ ] **Step 4: Persist raw extraction inside the building artifact**

`extract_document_mentions(document_id, expected_source_hash, ontology_version)` must:

1. re-read the concrete document and verify its hash;
2. select ordered chunks;
3. obtain the provider lazily;
4. batch extraction;
5. map/dedupe spans;
6. bulk-create immutable `EntityMention` and valid `RelationMention` rows attached to a `building` artifact;
7. record counts, timings, filtered malformed outputs, model/checkpoint identity, and ontology checksum in `GraphBuildRun.stats`.

Do not create resolved graph nodes in this task.

- [ ] **Step 5: Test rollback and fail-open behavior**

Provider failure must mark the build run failed, leave any prior active artifact untouched, and never change `Document.ingestion_complete`.

- [ ] **Step 6: Run tests**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_span_mapping.py aquillm/apps/knowledge_graph/tests/test_mention_extraction.py aquillm/apps/documents/tests/test_text_chunk_document.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add aquillm/apps/knowledge_graph/extraction aquillm/apps/knowledge_graph/tests/test_span_mapping.py aquillm/apps/knowledge_graph/tests/test_mention_extraction.py
git commit -m "feat(kg): persist gliner2 mention evidence"
```

---

## Chunk 3: Conservative Resolution, Filtering, and Graph Assembly

### Task 8: Resolve within-document names, identifiers, aliases, and acronyms

**Subagent:** `kg-coreference`

**Files:**
- Create: `aquillm/apps/knowledge_graph/resolution/__init__.py`
- Create: `aquillm/apps/knowledge_graph/resolution/normalization.py`
- Create: `aquillm/apps/knowledge_graph/resolution/coreference.py`
- Create: `aquillm/apps/knowledge_graph/tests/test_normalization.py`
- Create: `aquillm/apps/knowledge_graph/tests/test_coreference.py`

- [ ] **Step 1: Write failing normalization tests**

Test Unicode normalization, whitespace/punctuation folding, case-preserving display labels, version suffix preservation, DOI/arXiv/ORCID/repository identifier parsing, and explicit negative cases where normalization must not collapse distinct values.

- [ ] **Step 2: Write failing conservative coreference tests**

Require automatic clustering for:

- identical normalized names with compatible types;
- a full form followed by a parenthetical acronym, then later acronym mentions;
- exact stable identifier agreement;
- ontology-declared aliases.

Require separate candidate clusters for:

- an undefined acronym;
- an acronym with two candidate expansions;
- similar names with incompatible entity types;
- distinct versioned models/datasets;
- pronoun-only mentions (`it`, `they`, `this method`) in v1.

- [ ] **Step 3: Run tests and confirm failure**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_normalization.py aquillm/apps/knowledge_graph/tests/test_coreference.py -q`

Expected: FAIL because the resolver is absent.

- [ ] **Step 4: Implement deterministic clustering with an audit explanation**

`resolve_document_mentions(mentions, ontology) -> ResolutionResult` must return clusters plus an explanation for every accepted/rejected pair. Do not call an embedding model or LLM in this stage. Persist `DocumentEntity` and `DocumentEntityMention` links with resolver version and method.

- [ ] **Step 5: Run tests**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_normalization.py aquillm/apps/knowledge_graph/tests/test_coreference.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add aquillm/apps/knowledge_graph/resolution aquillm/apps/knowledge_graph/tests/test_normalization.py aquillm/apps/knowledge_graph/tests/test_coreference.py
git commit -m "feat(kg): resolve document entity mentions conservatively"
```

### Task 9: Resolve collection identities and apply transparent filtering

**Subagent:** `kg-collection-resolution`

**Files:**
- Create: `aquillm/apps/knowledge_graph/resolution/scoring.py`
- Create: `aquillm/apps/knowledge_graph/resolution/collection.py`
- Create: `aquillm/apps/knowledge_graph/graph/__init__.py`
- Create: `aquillm/apps/knowledge_graph/graph/filtering.py`
- Create: `aquillm/apps/knowledge_graph/tests/test_collection_resolution.py`
- Create: `aquillm/apps/knowledge_graph/tests/test_filtering.py`
- Reference: `aquillm/aquillm/utils.py`
- Reference: `aquillm/lib/embeddings/config.py`

- [ ] **Step 1: Write failing resolution-tier tests**

All `CollectionEntity` rows and `CollectionEntityDocumentLink` rows created here belong to the new `building` collection artifact. They must not be queried as current graph state until that exact artifact passes Task 10 validation and is atomically activated. Candidate generation must compare only document entities in the same collection build snapshot and compatible ontology type. Resolve in this order:

1. stable identifier equality;
2. exact normalized label/known alias;
3. type-constrained embedding candidates;
4. neighborhood agreement using already-supported relations.

Test three outcomes with independently configurable thresholds: `automatic`, `candidate`, and `rejected`. Automatic identity merges require a stricter threshold than retrieval similarity. Never make insertion order determine identity.

- [ ] **Step 2: Write failing filtering tests**

Keep separate values for:

```text
extraction_confidence
resolution_confidence
retrieval_utility
promotion_confidence
```

Test that frequency is log-capped, document dispersion is more useful than repeated boilerplate frequency, relation participation and title/abstract/caption position can raise utility, and ontology rejection/suppression policy never deletes raw mentions. Publishers are suppressed by default but can be enabled by an activated ontology extension.

- [ ] **Step 3: Run tests and confirm failure**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_collection_resolution.py aquillm/apps/knowledge_graph/tests/test_filtering.py -q`

Expected: FAIL.

- [ ] **Step 4: Implement deterministic blocking before embeddings**

Use indexed `(artifact_id, collection_id, entity_type, normalized_label)` blocking and stable identifiers first. Embed only unresolved candidate labels/descriptions via AquiLLM's existing embedding interface, store 1024-dimensional vectors, cap candidate fan-out, and record model signature plus scores. Do not invoke the chat LLM. Never mutate collection entities in the currently active artifact in place.

- [ ] **Step 5: Implement status-only filtering**

Filtering returns and persists `active`, `suppressed`, or `rejected` with reason codes. Re-running a policy version may change status without re-running GLiNER2.

- [ ] **Step 6: Run tests**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_collection_resolution.py aquillm/apps/knowledge_graph/tests/test_filtering.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add aquillm/apps/knowledge_graph/resolution aquillm/apps/knowledge_graph/graph/filtering.py aquillm/apps/knowledge_graph/tests/test_collection_resolution.py aquillm/apps/knowledge_graph/tests/test_filtering.py
git commit -m "feat(kg): resolve and filter collection entities"
```

### Task 10: Assemble evidence-backed collection relations into a shadow artifact

**Subagent:** `kg-graph-assembly`

**Files:**
- Create: `aquillm/apps/knowledge_graph/graph/assembly.py`
- Create: `aquillm/apps/knowledge_graph/tests/test_graph_assembly.py`
- Modify: `aquillm/apps/knowledge_graph/models/artifacts.py`
- Modify: `aquillm/apps/knowledge_graph/models/relations.py`

- [ ] **Step 1: Write failing assembly tests**

Require that:

- relation mentions are lifted from mention endpoints to resolved collection entities;
- ontology-invalid endpoint types remain rejected evidence;
- self-loops and generic identity edges are suppressed;
- repeated supporting mentions add evidence/support counts rather than duplicate edges;
- contradictory relations remain distinct assertions with independent evidence;
- every active collection entity and relation has at least one source chunk;
- building artifacts are invisible to retrieval;
- validation failure leaves the prior active artifact unchanged.
- the collection artifact records an aggregate source signature derived from the sorted contributing active document artifact IDs and their source/build signatures;
- activation is rejected when any contributing document artifact has changed, become stale, or been superseded since the aggregate snapshot was taken.

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_graph_assembly.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement shadow assembly and validation**

`assemble_collection_graph(collection_id, build_run_id, aggregate_source_signature)` writes `CollectionEntity`, association, relation, and evidence rows only to the new building collection artifact. The aggregate signature is a deterministic hash of the sorted active contributing document artifact IDs plus each document artifact's source hash, extractor/checkpoint, ontology, resolver, and filter-policy versions. Validate provenance, endpoint membership, active statuses, aggregate snapshot consistency, node/edge caps, and orphan counts before activation.

- [ ] **Step 4: Atomically activate the validated artifact**

Inside `transaction.atomic()`, lock competing artifacts/build runs and the collection row, recompute/recheck the contributing active document-artifact snapshot, verify that no newer source/version build won, mark the previous active collection artifact superseded, and activate the new artifact. The conditional uniqueness constraint from Task 3 is the final guard. An older task or a build assembled from a stale aggregate signature must never supersede a newer build.

- [ ] **Step 5: Run tests**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_graph_assembly.py aquillm/apps/knowledge_graph/tests/test_models.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add aquillm/apps/knowledge_graph/graph/assembly.py aquillm/apps/knowledge_graph/models aquillm/apps/knowledge_graph/tests/test_graph_assembly.py
git commit -m "feat(kg): assemble versioned collection graphs"
```

---

## Chunk 4: Idempotent Async Build and Document Lifecycle

### Task 11: Implement separate idempotent document builds and collection refreshes

**Subagent:** `kg-build-orchestration`

**Files:**
- Create: `aquillm/apps/knowledge_graph/services/__init__.py`
- Create: `aquillm/apps/knowledge_graph/services/builds.py`
- Create: `aquillm/apps/knowledge_graph/tests/test_build_idempotency.py`
- Create: `aquillm/apps/knowledge_graph/migrations/0002_graph_build_run_stages.py`
- Modify: `aquillm/apps/knowledge_graph/models/artifacts.py`
- Modify: `aquillm/apps/knowledge_graph/extraction/pipeline.py`
- Modify: `aquillm/apps/knowledge_graph/graph/assembly.py`

- [ ] **Step 1: Write failing document-build idempotency/race tests**

The document-build idempotency key is derived from:

```text
document UUID + full_text_hash + ordered chunk signature
+ extractor package/checkpoint revision + ontology checksum
+ coreference/resolver/filter policy versions
```

Test duplicate delivery, retry after provider failure, concurrent older/newer builds, document hash change during extraction, and failure after mention persistence but before document-artifact activation. Only the document build owns raw mentions, document entities, and its active document artifact.

- [ ] **Step 2: Write failing collection-refresh idempotency/race tests**

The collection refresh snapshots the sorted set of currently active contributing document artifacts. Its idempotency key is:

```text
collection UUID + aggregate document-artifact source signature
+ collection resolver/filter/assembly versions + ontology checksum
```

Test duplicate refresh delivery, two document builds finishing in either order, collection refresh racing a newer document activation, a document move between snapshot and activation, and failure after collection rows are assembled but before collection-artifact activation. Only the collection refresh owns `CollectionEntity`, collection association, `CollectionRelation`, and collection evidence rows in its building artifact.

- [ ] **Step 3: Run tests and confirm failure**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_build_idempotency.py -q`

Expected: FAIL.

- [ ] **Step 4: Implement the document-build state machine**

Use explicit states:

```text
document: queued -> extracting -> resolving -> validating -> active
                                                    \-> failed
          active -> superseded | stale
```

`build_document_graph(document_id, expected_source_hash, document_build_key)` must recheck the concrete document hash and ordered chunk signature before extraction and immediately before activation. In one transaction it locks the logical document scope, rechecks freshness, supersedes the prior active document artifact, activates the validated replacement, and then schedules collection refresh on `transaction.on_commit`. Retrying the same key resumes or safely replaces incomplete building state; it never duplicates active evidence.

- [ ] **Step 5: Implement the collection-refresh state machine**

```text
collection: queued -> snapshotting -> resolving -> assembling -> validating -> active
                                                               \-> failed
            active -> superseded | stale
```

`refresh_collection_graph(collection_id, aggregate_source_signature, collection_build_key)` must resolve and assemble exclusively inside a new building collection artifact. At activation it locks the collection scope, recomputes the sorted active document-artifact snapshot, and abandons/reschedules itself if the aggregate signature changed. A newer aggregate snapshot always wins regardless of Celery delivery order. No document build may directly mutate the active collection artifact.

Add the document and collection stage choices to `GraphBuildRun` in `models/artifacts.py`; migrations, transition validation, and terminal-state tests belong to this task. Do not encode stages as unvalidated free-form strings.

- [ ] **Step 6: Record structured build metrics**

Log `obs.kg.build_started`, `obs.kg.build_stage`, `obs.kg.build_failed`, and `obs.kg.build_completed` with IDs, versions, counts, and timings—but not raw document text or extracted private entity labels.

- [ ] **Step 7: Run tests**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_build_idempotency.py aquillm/apps/knowledge_graph/tests/test_mention_extraction.py aquillm/apps/knowledge_graph/tests/test_graph_assembly.py -q`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add aquillm/apps/knowledge_graph/services aquillm/apps/knowledge_graph/migrations/0002_graph_build_run_stages.py aquillm/apps/knowledge_graph/models/artifacts.py aquillm/apps/knowledge_graph/extraction/pipeline.py aquillm/apps/knowledge_graph/graph/assembly.py aquillm/apps/knowledge_graph/tests/test_build_idempotency.py
git commit -m "feat(kg): orchestrate idempotent graph builds"
```

### Task 12: Route graph builds to a dedicated Celery worker

**Subagent:** `kg-worker-deployment`

**Files:**
- Create: `aquillm/apps/knowledge_graph/tasks.py`
- Create: `aquillm/apps/knowledge_graph/tests/test_tasks.py`
- Create: `deploy/docker/knowledge-graph/Dockerfile`
- Create: `aquillm/tests/integration/test_knowledge_graph_compose.py`
- Modify: `aquillm/aquillm/settings.py`
- Modify: `deploy/compose/base.yml`
- Modify: `deploy/compose/development.yml`
- Modify: `deploy/compose/production.yml`
- Modify: `deploy/compose/no_gpu_dev.yml`

- [ ] **Step 1: Write failing task and static deployment tests**

Assert:

- extraction tasks route to `knowledge-graph-extraction`;
- task module import does not import ML packages;
- dedicated worker uses `--queues=knowledge-graph-extraction --concurrency=1 --prefetch-multiplier=1`;
- graph-worker image declares `WORKDIR /app/aquillm`, matching the final `manage.py` smoke commands;
- graph worker receives `DJANGO_DEBUG`, `KG_BUILD_ENABLED`, `KG_OVERLAY_ENABLED`, and fail-closed `KG_EVAL_BYPASS_ALLOWED` as explicit Compose environment overrides, and depends on healthy `db` and `redis` services;
- worker is under an optional `knowledge-graph` profile;
- worker installs `--extra knowledge-graph-local` while web/default worker Dockerfiles do not;
- Hugging Face cache is persistent and configurable;
- web does not depend on graph-worker readiness.

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_tasks.py aquillm/tests/integration/test_knowledge_graph_compose.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement thin task entry points**

Tasks accept scalar IDs and expected hashes only, call build services lazily, use bounded retries for transient broker/storage/provider failures, and record terminal failure. No GLiNER2 type appears in a task signature or top-level import.

Provide distinct task entry points for `build_document_graph_task` and `refresh_collection_graph_task`; task chaining is not the source of correctness. Each entry point calls the independently idempotent service above, so redelivery and out-of-order completion remain safe. Add a low-priority maintenance entry point for artifact pruning, but do not schedule it until Task 18 defines retention policy.

- [ ] **Step 4: Add dedicated worker image/profile**

Build from the same application source with `uv sync --frozen --no-dev --extra knowledge-graph-local`, then set `WORKDIR /app/aquillm`. Configure CPU initially and a persistent `HF_HOME`; GPU support is explicitly deferred. In each Compose variant, pass the four graph/debug settings through with defaults `DJANGO_DEBUG=0`, `KG_BUILD_ENABLED=0`, `KG_OVERLAY_ENABLED=0`, and `KG_EVAL_BYPASS_ALLOWED=0`, and require healthy `db`/`redis` before the dedicated worker starts.

- [ ] **Step 5: Run tests and Compose validation**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_tasks.py aquillm/tests/integration/test_knowledge_graph_compose.py aquillm/tests/integration/test_knowledge_graph_import_isolation.py -q`

Run: `docker compose --env-file .env -f deploy/compose/development.yml --profile knowledge-graph config --quiet`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add aquillm/apps/knowledge_graph/tasks.py aquillm/apps/knowledge_graph/tests/test_tasks.py aquillm/aquillm/settings.py deploy/docker/knowledge-graph deploy/compose aquillm/tests/integration/test_knowledge_graph_compose.py
git commit -m "build(kg): add isolated extraction worker"
```

### Task 13: Enqueue after both successful chunking paths and handle lifecycle invalidation

**Subagent:** `kg-document-lifecycle`

**Files:**
- Create: `aquillm/apps/knowledge_graph/graph/invalidation.py`
- Create: `aquillm/apps/knowledge_graph/tests/test_graph_enqueue.py`
- Create: `aquillm/apps/knowledge_graph/tests/test_document_lifecycle.py`
- Modify: `aquillm/apps/documents/tasks/chunking.py`
- Modify: `aquillm/apps/documents/models/document.py`
- Modify: `aquillm/apps/knowledge_graph/apps.py`
- Modify: `aquillm/apps/ingestion/tests/test_multimodal_ingestion_media_storage.py`
- Modify: `aquillm/apps/documents/tests/test_multimodal_chunk_position_uniqueness.py`

- [ ] **Step 1: Write failing enqueue tests for both success branches**

Cover normal chunk creation, a successful `DocumentFigure`/image chunk creation, and the duplicate-content chunk-copy early return. Queue only after chunks and `ingestion_complete=True` commit successfully. Queue failure logs and leaves ingestion successful.

- [ ] **Step 2: Write failing move/delete/rechunk/figure tests**

Require:

- content change marks the old document artifact stale before replacement activation;
- move captures old/new collection IDs, reuses document extraction, and refreshes both collection graphs;
- instance deletion and queryset/cascade deletion cannot leave active graph evidence;
- parent-document deletion behavior for `DocumentFigure` is explicit and tested despite its `GenericForeignKey`;
- all invalidation work is queued with `transaction.on_commit`, never performed synchronously in a model method.

- [ ] **Step 3: Run tests and confirm failure**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_graph_enqueue.py aquillm/apps/knowledge_graph/tests/test_document_lifecycle.py -q`

Expected: FAIL.

- [ ] **Step 4: Centralize post-chunk success enqueueing**

Add one helper called from both success exits of `create_chunks()`. Pass `document_id` and the expected `full_text_hash`; never copy another document's graph rows because evidence chunk IDs and collection resolution differ.

- [ ] **Step 5: Register deletion safety for every concrete document model**

Because `TextChunk.doc_id` is not a foreign key and queryset cascades can bypass `Document.delete()`, register narrowly scoped pre/post-delete handlers from `KnowledgeGraphConfig.ready()` for concrete document types. Handlers enqueue cleanup/refresh only; protect imports from cycles and test idempotency.

- [ ] **Step 6: Run focused and ingestion regressions**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_graph_enqueue.py aquillm/apps/knowledge_graph/tests/test_document_lifecycle.py aquillm/apps/documents/tests/test_multimodal_chunk_position_uniqueness.py aquillm/apps/ingestion/tests/test_multimodal_ingestion_media_storage.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add aquillm/apps/knowledge_graph aquillm/apps/documents/tasks/chunking.py aquillm/apps/documents/models/document.py aquillm/apps/documents/tests/test_multimodal_chunk_position_uniqueness.py aquillm/apps/ingestion/tests/test_multimodal_ingestion_media_storage.py
git commit -m "feat(kg): maintain graphs across document lifecycle"
```

---

## Chunk 5: Cross-Collection Identity Without Cross-Collection Claims

### Task 14: Build conservative canonical identity links and a rebuildable lookup index

**Subagent:** `kg-canonical-links`

**Files:**
- Create: `aquillm/apps/knowledge_graph/resolution/canonical.py`
- Create: `aquillm/apps/knowledge_graph/tests/test_canonical_resolution.py`
- Create: `aquillm/apps/knowledge_graph/tests/test_canonical_permissions.py`
- Modify: `aquillm/apps/knowledge_graph/models/associations.py`
- Modify: `aquillm/apps/knowledge_graph/services/builds.py`

- [ ] **Step 1: Write failing canonical-resolution tests**

Automatic cross-collection links are allowed only for:

- identical stable identifiers and compatible types;
- exact normalized names/declared aliases with no conflicting identifiers;
- document-defined acronyms already resolved to the same full form.

Embedding similarity may generate a reviewable candidate but cannot automatically merge cross-collection identities in v1. Test model-version distinctions, ambiguous acronyms, type mismatches, unlink/rebuild after a corrected resolution, and deterministic results independent of collection processing order.

- [ ] **Step 2: Write failing privacy-boundary tests**

AquiLLM has no workspace/tenant model. Therefore:

- `CanonicalEntity` is an internal identity registry, not a user-enumerable graph;
- it stores no independent claim, edge, chunk, or access grant;
- collection-local links remain the permission-bearing endpoints;
- a user-visible traversal must filter candidate collection entities and evidence by the authorized document/collection allowlist before crossing the link;
- counts, labels, or neighbor existence from inaccessible collections never appear in API output or diagnostics.

- [ ] **Step 3: Run tests and confirm failure**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_canonical_resolution.py aquillm/apps/knowledge_graph/tests/test_canonical_permissions.py -q`

Expected: FAIL.

- [ ] **Step 4: Implement audited canonical links**

Persist the source collection entity, canonical identity, score, method, resolver version, and status. Keep all substantive relations and evidence on `CollectionEntity`/`CollectionRelation`; never promote a collection relation into an unqualified canonical fact.

- [ ] **Step 5: Implement the dictionary as a derived cache only**

Expose a rebuildable lookup equivalent to:

```text
canonical_entity_id -> [(collection_a_id, entity_a_id), (collection_b_id, entity_b_id)]
```

If stored in Django cache, include canonical resolver version and active artifact IDs in the key and treat a miss/error as a database lookup. Do not create the reverse source-of-truth mapping proposed earlier.

Invalidate or version-bust derived canonical lookup entries whenever a collection artifact is activated/superseded, a collection entity/canonical link is deleted, or the canonical resolver version changes. Cache invalidation failure must fail open to a permission-filtered database lookup; stale cache entries must never authorize traversal.

- [ ] **Step 6: Run tests**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_canonical_resolution.py aquillm/apps/knowledge_graph/tests/test_canonical_permissions.py aquillm/apps/knowledge_graph/tests/test_collection_resolution.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add aquillm/apps/knowledge_graph/resolution/canonical.py aquillm/apps/knowledge_graph/models/associations.py aquillm/apps/knowledge_graph/services/builds.py aquillm/apps/knowledge_graph/tests/test_canonical_resolution.py aquillm/apps/knowledge_graph/tests/test_canonical_permissions.py
git commit -m "feat(kg): link collection entities conservatively"
```

---

## Chunk 6: Permission-Safe Hybrid Retrieval

### Task 15: Implement bounded graph candidate expansion

**Subagent:** `kg-retrieval-expansion`

**Files:**
- Create: `aquillm/apps/knowledge_graph/retrieval/expansion.py`
- Create: `aquillm/apps/knowledge_graph/tests/test_retrieval_expansion.py`
- Create: `aquillm/apps/knowledge_graph/tests/test_retrieval_overlay_permissions.py`
- Reference: `aquillm/apps/collections/models/collection.py`
- Reference: `aquillm/apps/documents/models/chunks.py`

- [ ] **Step 1: Write failing local and cross-collection expansion tests**

Given real seed chunk IDs and explicit authorized document/collection IDs, test:

```text
seed chunk
 -> entity mention
 -> active collection entity
 -> one semantic relation edge
 -> supporting real chunk
```

and:

```text
seed collection entity
 -> canonical identity link (identity bridge, not a semantic hop)
 -> authorized peer collection entity
 -> one semantic relation edge
 -> supporting real chunk
```

Only actual `TextChunk` IDs may be returned. Building, failed, stale, or superseded artifacts are ignored.

- [ ] **Step 2: Write failing permission and bound tests**

Test that:

- both `allowed_doc_ids` and `allowed_collection_ids` are required nonempty fields on `GraphExpansionRequest`, and every allowed document is proven to belong to an allowed collection before traversal;
- unselected or inaccessible documents are excluded in the first ORM query, not filtered after traversal;
- raw `CollectionsRef` IDs are never accepted by this service;
- parent/child collection selection tests pin the current `Collection.get_user_accessible_documents` behavior, including the existing difference between recursive `user_can_view` and direct-row `filter_by_user_perm`; this KG feature must neither widen nor silently "fix" that separate permission contract;
- document-access cache/revocation tests prove graph traversal receives exactly the same authorized document snapshot as baseline retrieval and cannot bypass that resolver;
- one semantic hop, per-node fan-out, total candidate, and per-document caps are enforced;
- self-loops and duplicate evidence do not consume multiple result slots;
- deterministic support/rank ordering breaks ties by chunk PK;
- timeout/database error returns an empty result with status, never raises into chunk search;
- diagnostics contain only counts/timing/status and non-enumerating version signatures, with no artifact IDs, query text, entity labels, collection names, or inaccessible counts.

- [ ] **Step 3: Run tests and confirm failure**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_retrieval_expansion.py aquillm/apps/knowledge_graph/tests/test_retrieval_overlay_permissions.py -q`

Expected: FAIL.

- [ ] **Step 4: Implement one bounded PostgreSQL expansion**

`expand_chunk_candidates(request) -> GraphExpansionResult` must require the explicit document and collection allowlists frozen in Task 1, verify their relationship in the first query, and query only active artifacts inside that intersection. Use a transaction-local PostgreSQL statement timeout when supported, plus hard row limits. Initial defaults:

```text
KG_OVERLAY_MAX_HOPS=1
KG_OVERLAY_MAX_FANOUT=10
KG_OVERLAY_MAX_CANDIDATES=20
KG_OVERLAY_TIMEOUT_MS=150
```

Do not invoke GLiNER2, embeddings, rerankers, LLMs, or a network service on this path. Do not add a retrieval cache in v1.

- [ ] **Step 5: Run tests**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_retrieval_expansion.py aquillm/apps/knowledge_graph/tests/test_retrieval_overlay_permissions.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add aquillm/apps/knowledge_graph/retrieval/expansion.py aquillm/apps/knowledge_graph/tests/test_retrieval_expansion.py aquillm/apps/knowledge_graph/tests/test_retrieval_overlay_permissions.py
git commit -m "feat(kg): add bounded graph candidate expansion"
```

### Task 16: Integrate graph candidates before the existing reranker

**Subagent:** `kg-chunk-search-integration`

**Files:**
- Create: `aquillm/apps/documents/tests/test_chunk_search_graph_overlay.py`
- Modify: `aquillm/apps/documents/services/chunk_search.py`
- Modify: `aquillm/aquillm/settings.py`
- Reference: `aquillm/apps/documents/services/chunk_rerank.py`
- Reference: `aquillm/apps/chat/services/tool_wiring/documents.py`

- [ ] **Step 1: Write failing chunk-search integration tests**

Cover:

- disabled feature returns the exact existing candidate/result order and diagnostics shape, with no graph-specific keys added;
- enabled expansion receives allowlists derived from the already-authorized `docs` argument, never raw chat collection IDs;
- graph candidates are real ORM `TextChunk` rows;
- duplicate graph/vector/trigram/exact chunks appear once;
- graph candidates join the candidate pool before the existing reranker;
- graph timeout/error produces the exact vector/trigram/exact baseline rather than raising;
- vector failure with trigram/exact seeds may still expand;
- candidate cap and deterministic ordering are honored;
- existing 4-tuple return `(vector_results, trigram_results, reranked_results, diagnostics)` remains unchanged.

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest aquillm/apps/documents/tests/test_chunk_search_graph_overlay.py -q`

Expected: FAIL.

- [ ] **Step 3: Refactor candidate dedupe into a small helper**

Materialize vector/trigram/exact candidates, dedupe baseline seeds, call graph expansion, append graph-supported rows, and dedupe the combined pool before the existing fallback/reranker branch. When the overlay is disabled or misses, preserve baseline behavior byte-for-byte where observable.

- [ ] **Step 4: Add structured graph diagnostics**

Extend `obs.rag.search` and the returned diagnostics with:

```text
graph_ms graph_seed_count graph_candidate_count
graph_status graph_version_signature
```

Add these fields only when `KG_OVERLAY_ENABLED=1`; the disabled path must preserve the pre-feature diagnostics object exactly. Use statuses `miss|hit|timeout|error`. `graph_version_signature` is a one-way aggregate of only the authorized active artifact versions and must not expose artifact IDs, collection counts, or inaccessible state. Never log query text or graph labels. The public result rows and no-result payload remain unchanged; graph diagnostics stay inside the existing diagnostics channel and may not add graph triples or pseudo-evidence.

- [ ] **Step 5: Run focused regression suite**

Run: `python -m pytest aquillm/apps/documents/tests/test_chunk_search_graph_overlay.py aquillm/apps/documents/tests/test_chunk_search_candidate_tuning.py aquillm/apps/documents/tests/test_chunk_search_diagnostics.py aquillm/apps/documents/tests/test_chunk_search_query_cache.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add aquillm/apps/documents/services/chunk_search.py aquillm/apps/documents/tests/test_chunk_search_graph_overlay.py aquillm/aquillm/settings.py
git commit -m "feat(rag): expand hybrid candidates through collection graph"
```

### Task 17: Preserve tool payloads, evidence budgets, citations, and direct RAG

**Subagent:** `kg-rag-compatibility`

**Files:**
- Modify: `aquillm/lib/tools/search/vector_search.py`
- Modify: `aquillm/lib/tools/search/tests/test_vector_search_pack.py`
- Modify: `aquillm/apps/chat/tests/test_rag_evidence.py`
- Modify: `aquillm/apps/chat/tests/test_direct_rag_pipeline.py`
- Create: `aquillm/apps/chat/tests/test_single_document_graph_overlay.py`
- Modify: `aquillm/lib/llm/tests/test_rag_citations.py`
- Modify: `aquillm/apps/chat/services/rag_metrics.py`
- Reference: `aquillm/lib/tools/search/vector_search.py`
- Reference: `aquillm/apps/chat/services/rag_evidence.py`
- Reference: `aquillm/lib/llm/providers/rag_citations.py`

- [ ] **Step 1: Add failing compatibility regressions**

Prove that a graph-expanded chunk:

- uses the unchanged verbose and compact tool row shapes;
- produces only its real `[doc:<uuid> chunk:<numeric-pk>]` citation;
- never exposes canonical nodes, triples, scores, or pseudo-evidence as citable rows;
- strips every `graph_*` field from verbose, compact, and no-result public tool payload diagnostics, even though enabled `chunk_search` retains those fields internally for structured metrics;
- obeys existing per-document and token evidence caps;
- is available to both normal `vector_search` tool calls and direct RAG;
- leaves `search_single_document` payload/citation behavior unchanged and never expands outside its one already-authorized document;
- cannot authorize a citation from an inaccessible/unreturned chunk;
- leaves direct RAG handled when graph expansion fails internally and vector results remain.

- [ ] **Step 2: Run tests and confirm any missing coverage fails**

Run: `python -m pytest aquillm/lib/tools/search/tests/test_vector_search_pack.py aquillm/apps/chat/tests/test_rag_evidence.py aquillm/apps/chat/tests/test_direct_rag_pipeline.py aquillm/apps/chat/tests/test_single_document_graph_overlay.py aquillm/lib/llm/tests/test_rag_citations.py -q`

Expected: New assertions FAIL only where compatibility wiring/metrics are incomplete.

- [ ] **Step 3: Make the minimum compatibility changes**

Do not change public payload keys. Add one explicit diagnostics-sanitization helper at the `pack_chunk_search_results()` boundary that removes `graph_*` keys before any verbose, compact, or no-result tool payload is returned; do not rely on callers to remember to filter. Extend internal metrics with optional graph fields using defaults so existing callers/tests remain valid. The direct RAG pipeline should require no new retrieval call because it already delegates to the existing vector-search tool.

- [ ] **Step 4: Run tests**

Run: `python -m pytest aquillm/lib/tools/search/tests/test_vector_search_pack.py aquillm/apps/chat/tests/test_rag_evidence.py aquillm/apps/chat/tests/test_direct_rag_pipeline.py aquillm/apps/chat/tests/test_single_document_graph_overlay.py aquillm/lib/llm/tests/test_rag_citations.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aquillm/lib/tools/search/vector_search.py aquillm/lib/tools/search/tests/test_vector_search_pack.py aquillm/apps/chat/tests/test_rag_evidence.py aquillm/apps/chat/tests/test_direct_rag_pipeline.py aquillm/apps/chat/tests/test_single_document_graph_overlay.py aquillm/lib/llm/tests/test_rag_citations.py aquillm/apps/chat/services/rag_metrics.py
git commit -m "test(rag): preserve citations with graph expansion"
```

---

## Chunk 7: Quality Gates, Operations, and Controlled Rollout

### Task 18: Add rebuild, inspection, extractor-check, and retention operations

**Subagent:** `kg-operations`

**Files:**
- Create: `aquillm/apps/knowledge_graph/management/__init__.py`
- Create: `aquillm/apps/knowledge_graph/management/commands/__init__.py`
- Create: `aquillm/apps/knowledge_graph/management/commands/rebuild_knowledge_graph.py`
- Create: `aquillm/apps/knowledge_graph/management/commands/inspect_knowledge_graph.py`
- Create: `aquillm/apps/knowledge_graph/management/commands/check_knowledge_graph_extractor.py`
- Create: `aquillm/apps/knowledge_graph/management/commands/prune_knowledge_graph.py`
- Create: `aquillm/apps/knowledge_graph/services/inspection.py`
- Create: `aquillm/apps/knowledge_graph/services/pruning.py`
- Create: `aquillm/apps/knowledge_graph/tests/test_management_commands.py`
- Create: `aquillm/apps/knowledge_graph/tests/test_pruning.py`
- Create: `aquillm/apps/knowledge_graph/migrations/0003_graph_rebuild_request.py`
- Modify: `aquillm/apps/knowledge_graph/tasks.py`
- Modify: `aquillm/apps/knowledge_graph/services/builds.py`
- Modify: `aquillm/apps/knowledge_graph/models/artifacts.py`
- Modify: `aquillm/apps/knowledge_graph/tests/test_tasks.py`
- Modify: `aquillm/apps/knowledge_graph/tests/test_build_idempotency.py`
- Modify: `aquillm/lib/knowledge_graph/config.py`
- Modify: `aquillm/aquillm/settings.py`

- [ ] **Step 1: Write failing command tests**

Require:

- rebuild accepts exactly one of `--document <uuid>`, `--collection <uuid>`, or operator-wide `--all`, and enqueues rather than extracting synchronously;
- every rebuild creates a durable `GraphRebuildRequest` with caller-supplied-or-generated UUID, scope, requested document/source snapshot, expected aggregate signature, status, timestamps, and terminal failure counts; the command prints the request UUID and supports `--request-id <uuid>` for deterministic automation;
- `--all` means every eligible document/collection in the local deployment database; it is not user-access scoped and requires `--yes` unless combined with `--dry-run`;
- `--eval-only` bypasses `KG_BUILD_ENABLED` only when Django is in an explicit test/debug environment with `KG_EVAL_BYPASS_ALLOWED=1`, requires one concrete `--collection`, rejects `--all`, marks the build/task payload/build run as evaluation-only, and is rejected in production settings;
- dry-run reports counts without mutations;
- inspection delegates to `services/inspection.py`, accepts optional `--document`/`--collection` filters plus `--request-id <uuid> --wait --timeout-seconds`, and reports artifact/build status, versions, stale counts, active evidence coverage, and failures without raw private text;
- extractor check loads the pinned checkpoint only when explicitly invoked, runs one entity/relation fixture, verifies spans, and prints package/checkpoint identity;
- commands exit nonzero on invalid scope, missing worker/runtime, incompatible checkpoint, or failed extraction fixture.

- [ ] **Step 2: Write failing retention/pruning tests**

Require a configurable policy with `KG_ARTIFACT_RETENTION_DAYS` and `KG_ARTIFACT_KEEP_SUPERSEDED`. `prune_knowledge_graph` and its low-priority Celery task must:

- default to dry-run unless `--execute` is explicit;
- delete in bounded batches only terminal `superseded`, `stale`, or `failed` artifacts/build runs older than the retention boundary;
- preserve every `active` or `building` artifact, the newest configured superseded artifacts per logical scope, and anything referenced by a live build/canonical link;
- remove dependent evidence through declared database cascades without touching `TextChunk` or document rows;
- tolerate retry/duplicate delivery and report IDs/counts only, never entity labels or document text.

At the asynchronous boundary, test direct task invocation and direct build-service invocation with forged `eval_only=True`. Both `tasks.py` and `services/builds.py` must independently re-read settings and reject the bypass unless `DJANGO_DEBUG=1` (or the test-runner equivalent) and `KG_EVAL_BYPASS_ALLOWED=1`. The service-level check is authoritative and runs before creating a build run or reading document content; the management-command check is only early feedback.

Test collection rebuild orchestration from (a) no graph artifacts and (b) a pre-existing active collection artifact. For one immutable request snapshot, enqueue/idempotently complete a document build for every current member document—even when a matching active document artifact already exists—then compute the aggregate signature from the request's resulting active document artifacts and enqueue exactly one final collection refresh. The old active collection artifact remains queryable until that request's new collection artifact activates. Membership/source changes invalidate and resnapshot the request; partial document failure leaves the prior collection artifact active and marks the request failed/partial according to policy rather than activating an incomplete graph.

Test that `inspect --request-id <uuid> --wait` cannot return on an older active artifact or before the rebuild command's transaction commits. It waits on that durable request through all document builds and the correlated collection activation, exits nonzero on request failure/timeout, and confirms the activated collection artifact records the same request UUID and expected aggregate signature.

- [ ] **Step 3: Run tests and confirm failure**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_management_commands.py aquillm/apps/knowledge_graph/tests/test_pruning.py aquillm/apps/knowledge_graph/tests/test_tasks.py aquillm/apps/knowledge_graph/tests/test_build_idempotency.py -q`

Expected: FAIL.

- [ ] **Step 4: Implement thin commands over services**

Keep business logic in build, inspection, and pruning services. Commands validate arguments, create the durable rebuild request and snapshot atomically, invoke services, and format bounded output. The operator-wide rebuild command must enumerate eligible scopes in deterministic pages and create child requests/enqueue scalar IDs/source signatures; it must not load GLiNER2 in the command process. Add fail-closed parsing for `KG_EVAL_BYPASS_ALLOWED` with default `0`. Propagate the request UUID and evaluation marker through every scalar Celery payload, then fail closed again in both the task and build service before honoring it. Document-build completion schedules the correlated collection refresh only after every request member has reached the required terminal document state.

- [ ] **Step 5: Run tests and command smoke checks**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_management_commands.py aquillm/apps/knowledge_graph/tests/test_pruning.py aquillm/apps/knowledge_graph/tests/test_tasks.py aquillm/apps/knowledge_graph/tests/test_build_idempotency.py -q`

Run from `aquillm/`: `python manage.py rebuild_knowledge_graph --help`

Run from `aquillm/`: `python manage.py inspect_knowledge_graph --help`

Run from `aquillm/`: `python manage.py prune_knowledge_graph --help`

Expected: PASS and all commands load without optional ML imports.

- [ ] **Step 6: Commit**

```bash
git add aquillm/apps/knowledge_graph/management aquillm/apps/knowledge_graph/migrations/0003_graph_rebuild_request.py aquillm/apps/knowledge_graph/models/artifacts.py aquillm/apps/knowledge_graph/services/builds.py aquillm/apps/knowledge_graph/services/inspection.py aquillm/apps/knowledge_graph/services/pruning.py aquillm/apps/knowledge_graph/tasks.py aquillm/apps/knowledge_graph/tests/test_management_commands.py aquillm/apps/knowledge_graph/tests/test_pruning.py aquillm/apps/knowledge_graph/tests/test_tasks.py aquillm/apps/knowledge_graph/tests/test_build_idempotency.py aquillm/lib/knowledge_graph/config.py aquillm/aquillm/settings.py
git commit -m "feat(kg): add graph operations and retention"
```

### Task 19: Document configuration, rollout, rollback, retention, and ownership

**Subagent:** `kg-runbook`

**Files:**
- Modify: `.env.example`
- Modify: `docs/documents/operations/knowledge-graph-overlay-runbook.md`
- Modify: `docs/documents/README.md`
- Modify: `docs/documents/architecture/2026-04-09-knowledge-graph-index-overlay-design.md`
- Create: `aquillm/apps/knowledge_graph/tests/test_config.py`

- [ ] **Step 1: Write failing default/config tests**

Assert independent off-by-default controls:

```text
KG_BUILD_ENABLED=0
KG_OVERLAY_ENABLED=0
KG_EXTRACTOR_PROVIDER=gliner2_local
KG_EXTRACTOR_FAIL_OPEN=1
KG_EXTRACTION_QUEUE=knowledge-graph-extraction
KG_GLINER2_MODEL=fastino/gliner2-base-v1
KG_GLINER2_REVISION=8437ba583a733d87f56ae902f3b197934eedd58e
KG_GLINER2_DEVICE=cpu
KG_GLINER2_BATCH_SIZE=8
KG_GLINER2_CACHE_DIR=/root/.cache/huggingface
KG_GLINER2_LOCAL_FILES_ONLY=0
KG_OVERLAY_MAX_HOPS=1
KG_OVERLAY_MAX_FANOUT=10
KG_OVERLAY_MAX_CANDIDATES=20
KG_OVERLAY_TIMEOUT_MS=150
KG_ARTIFACT_RETENTION_DAYS=30
KG_ARTIFACT_KEEP_SUPERSEDED=2
KG_EVAL_BYPASS_ALLOWED=0
```

Invalid or missing required values must disable/fail graph work according to fail-open policy, never break Django startup. `KG_EVAL_BYPASS_ALLOWED=1` is valid only with debug/test settings and must never be documented as a deployed production option.

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_config.py -q`

Expected: FAIL.

- [ ] **Step 3: Document staged rollout and exact operator commands**

The runbook order is mandatory:

1. build the optional worker image;
2. prefetch and verify the immutable checkpoint;
3. start the dedicated worker;
4. enable builds only;
5. backfill one explicitly named representative collection with `rebuild_knowledge_graph --collection <uuid>`;
6. inspect extraction/resolution/filter quality and failed builds;
7. run the exact baseline and comparison commands from Task 20 and record numeric gates;
8. enable graph retrieval in development/staging;
9. soak and monitor;
10. enable production retrieval selectively.

Rollback order: disable `KG_OVERLAY_ENABLED`, disable `KG_BUILD_ENABLED`, then stop the optional worker. Keep persisted artifacts for diagnosis/recovery. Document the scheduled pruning task, dry-run review, retention exceptions, and emergency disk-pressure procedure without ever pruning active/building artifacts.

- [ ] **Step 4: Document ownership and non-goals**

State who may activate ontology versions, review candidate identity links, approve numeric retrieval gates, run operator-wide backfills, and execute pruning. Document that Mem0 remains separate, canonical identities are not a user-enumerable global graph, and no graph visualization or automatic ontology generation ships in v1.

- [ ] **Step 5: Run config/docs checks**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_config.py -q`

Run: `python scripts/check_logging_conventions.py`

Run: `python scripts/check_file_lengths.py`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add .env.example docs/documents aquillm/apps/knowledge_graph/tests/test_config.py
git commit -m "docs(kg): add controlled graph rollout runbook"
```

### Task 20: Implement reproducible extraction and retrieval quality gates

**Subagent:** `kg-quality-eval`

**Files:**
- Modify: `aquillm/apps/knowledge_graph/evals/run_kg_eval.py`
- Modify: `aquillm/apps/knowledge_graph/evals/extraction_cases.yaml`
- Modify: `aquillm/apps/knowledge_graph/evals/retrieval_cases.yaml`
- Modify: `aquillm/apps/knowledge_graph/tests/test_eval_runner.py`
- Create: `aquillm/apps/knowledge_graph/tests/test_retrieval_eval.py`
- Modify: `docs/documents/operations/knowledge-graph-overlay-runbook.md`

- [ ] **Step 1: Add failing end-to-end eval tests with deterministic fakes**

The runner must compare vector-only and graph-expanded retrieval on the same concrete collection/database fixture and report Recall@K, MRR, nDCG, graph hit rate, inaccessible-result count, added latency, and citation-evidence coverage. Extraction metrics must score mention spans, relation direction/endpoints, automatic identity links, and suppression decisions separately. Tests must reject different/missing collection scopes between reports and reject `--eval-only` outside explicit test/debug settings.

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_eval_runner.py aquillm/apps/knowledge_graph/tests/test_retrieval_eval.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement reproducible reports and rollout comparison**

Emit JSON plus a human-readable table containing model/checkpoint, ontology, resolver/filter versions, authorized graph version signature, fixture checksum, and vector-only versus graph metrics. The runner must create a missing output parent and atomically replace the final report only after successful completion. Never silently drop skipped cases and never include private entity labels in the report.

Add `--eval-only` to graph-overlay mode. It may bypass `KG_OVERLAY_ENABLED` only when debug/test settings and `KG_EVAL_BYPASS_ALLOWED=1` are both present, only for the one `--collection` supplied, and must use the same permission-scoped expansion service. Reject it in production settings. This is the sole path allowed to measure overlay gates while both shipping feature flags remain `0`.

- [ ] **Step 4: Implement the measured-gate workflow**

Create the runbook's explicit `PENDING_MEASUREMENT` gate table and implement `--write-measured-gates` plus `--verify-gates`. The writer consumes same-collection baseline/overlay reports and atomically records the measured values; verification fails while any value is pending or failing. At minimum gates require zero inaccessible chunks, exact baseline behavior on miss/error, stricter automatic-link precision than candidate-link precision, positive Recall@10/nDCG movement on relationship/alias/cross-document cases, graph expansion p95 within the configured local-DB budget, and 100% curated citation-evidence coverage. `KG_OVERLAY_ENABLED=1` is forbidden while any gate is pending or failing. Actual DB-backed measurement and gate writing occur only in Task 21's Compose-backed sequence.

- [ ] **Step 5: Run deterministic runner tests without host database access**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_eval_runner.py aquillm/apps/knowledge_graph/tests/test_retrieval_eval.py -q`

Expected: PASS; tests use deterministic fakes/pytest DB fixtures, prove report-parent atomic creation, reject cross-collection report comparison, prove measured-gate write/verify behavior, and leave the checked-in runbook gates explicitly pending for Task 21.

- [ ] **Step 6: Commit**

```bash
git add aquillm/apps/knowledge_graph/evals aquillm/apps/knowledge_graph/tests/test_eval_runner.py aquillm/apps/knowledge_graph/tests/test_retrieval_eval.py docs/documents/operations/knowledge-graph-overlay-runbook.md
git commit -m "test(kg): gate graph quality and retrieval uplift"
```

---

## Chunk 8: Final Verification and Handoff

### Task 21: Verify the complete feature with both flags disabled

**Subagent:** `kg-final-verification`

**Files:**
- Verify all files above; modify only to correct discovered plan-scope defects.
- Modify: `docs/documents/operations/knowledge-graph-overlay-runbook.md` only to record the measured/approved gates produced in Step 6.

- [ ] **Step 1: Run knowledge-graph and import-isolation tests**

Run: `python -m pytest aquillm/lib/knowledge_graph/tests aquillm/apps/knowledge_graph/tests aquillm/tests/integration/test_knowledge_graph_import_isolation.py aquillm/tests/integration/test_knowledge_graph_compose.py -q`

Expected: PASS without installing/loading the optional model in the web environment.

- [ ] **Step 2: Run retrieval/RAG compatibility tests**

Run: `python -m pytest aquillm/apps/documents/tests/test_chunk_search_graph_overlay.py aquillm/apps/documents/tests/test_chunk_search_candidate_tuning.py aquillm/apps/documents/tests/test_chunk_search_diagnostics.py aquillm/apps/documents/tests/test_chunk_search_query_cache.py aquillm/lib/tools/search/tests/test_vector_search_pack.py aquillm/apps/chat/tests/test_rag_evidence.py aquillm/apps/chat/tests/test_direct_rag_pipeline.py aquillm/apps/chat/tests/test_single_document_graph_overlay.py aquillm/lib/llm/tests/test_rag_citations.py -q`

Expected: PASS.

- [ ] **Step 3: Run ingestion/lifecycle regressions**

Run: `python -m pytest aquillm/apps/documents/tests aquillm/apps/ingestion/tests aquillm/apps/collections/tests -q`

Expected: PASS.

- [ ] **Step 4: Run architecture, migration, and Django checks**

Run: `python scripts/check_import_boundaries.py`

Run: `python scripts/check_file_lengths.py`

Run: `python aquillm/manage.py makemigrations --check --dry-run`

Run: `python aquillm/manage.py check`

Expected: PASS/no pending migrations.

- [ ] **Step 5: Build and verify the optional worker**

Run:

```bash
docker compose --env-file .env -f deploy/compose/development.yml --profile knowledge-graph build worker_knowledge_graph
docker compose --env-file .env -f deploy/compose/development.yml --profile knowledge-graph run --rm --no-deps --entrypoint uv worker_knowledge_graph pip check --python /opt/venv/bin/python
docker compose --env-file .env -f deploy/compose/development.yml --profile knowledge-graph run --rm --no-deps --entrypoint /opt/venv/bin/python worker_knowledge_graph manage.py check_knowledge_graph_extractor
```

Expected: image builds, dependency check passes, and the pinned local checkpoint passes the smoke fixture.

- [ ] **Step 6: Start isolated dependencies and run the eval-only sample comparison**

Keep both shipping flags disabled. Before starting, the operator must set `KG_EVAL_COLLECTION_ID` in the PowerShell session to an approved existing fixture-collection UUID. From the repository root, run this block; it fails before Compose startup when the variable is absent or invalid, starts the database, Redis, and the explicitly eval-authorized dedicated worker, runs every management/eval command inside the same Compose network, and tears those services down in `finally`:

```powershell
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
[Guid]$kgEvalGuid = [Guid]::Empty
if (-not [Guid]::TryParse($env:KG_EVAL_COLLECTION_ID, [ref]$kgEvalGuid)) {
    throw 'Set KG_EVAL_COLLECTION_ID to an approved collection UUID before running verification.'
}
$kgEvalCollectionId = $kgEvalGuid.ToString()
$kgRebuildRequestId = [Guid]::NewGuid().ToString()
$kgTemporaryEnvNames = @('DJANGO_DEBUG', 'KG_BUILD_ENABLED', 'KG_OVERLAY_ENABLED', 'KG_EVAL_BYPASS_ALLOWED')
$kgPriorEnv = @{}
foreach ($kgEnvName in $kgTemporaryEnvNames) {
    $kgPriorEnv[$kgEnvName] = [Environment]::GetEnvironmentVariable($kgEnvName, 'Process')
}
$env:DJANGO_DEBUG = '1'
$env:KG_BUILD_ENABLED = '0'
$env:KG_OVERLAY_ENABLED = '0'
$env:KG_EVAL_BYPASS_ALLOWED = '1'
$kgComposeArgs = @('--env-file', '.env', '-f', 'deploy/compose/development.yml', '--profile', 'knowledge-graph')

try {
    docker compose @kgComposeArgs up -d db redis worker_knowledge_graph
    docker compose @kgComposeArgs run --rm --no-deps --entrypoint /opt/venv/bin/python worker_knowledge_graph manage.py rebuild_knowledge_graph --collection $kgEvalCollectionId --request-id $kgRebuildRequestId --eval-only
    docker compose @kgComposeArgs run --rm --no-deps --entrypoint /opt/venv/bin/python worker_knowledge_graph manage.py inspect_knowledge_graph --request-id $kgRebuildRequestId --wait --timeout-seconds 1800
    docker compose @kgComposeArgs run --rm --no-deps --entrypoint /opt/venv/bin/python worker_knowledge_graph -m apps.knowledge_graph.evals.run_kg_eval --mode vector-only --collection $kgEvalCollectionId --output /app/artifacts/kg-eval-vector.json
    docker compose @kgComposeArgs run --rm --no-deps --entrypoint /opt/venv/bin/python worker_knowledge_graph -m apps.knowledge_graph.evals.run_kg_eval --mode graph-overlay --eval-only --collection $kgEvalCollectionId --output /app/artifacts/kg-eval-overlay.json
    docker compose @kgComposeArgs run --rm --no-deps --entrypoint /opt/venv/bin/python worker_knowledge_graph -m apps.knowledge_graph.evals.run_kg_eval --write-measured-gates --baseline-report /app/artifacts/kg-eval-vector.json --overlay-report /app/artifacts/kg-eval-overlay.json --runbook /app/docs/documents/operations/knowledge-graph-overlay-runbook.md
    docker compose @kgComposeArgs run --rm --no-deps --entrypoint /opt/venv/bin/python worker_knowledge_graph -m apps.knowledge_graph.evals.run_kg_eval --verify-gates --baseline-report /app/artifacts/kg-eval-vector.json --overlay-report /app/artifacts/kg-eval-overlay.json --runbook /app/docs/documents/operations/knowledge-graph-overlay-runbook.md
    docker compose @kgComposeArgs run --rm --no-deps --entrypoint /opt/venv/bin/python worker_knowledge_graph manage.py prune_knowledge_graph --dry-run
}
finally {
    try {
        docker compose @kgComposeArgs stop worker_knowledge_graph db redis
    }
    finally {
        foreach ($kgEnvName in $kgTemporaryEnvNames) {
            if ($null -eq $kgPriorEnv[$kgEnvName]) {
                Remove-Item "Env:$kgEnvName" -ErrorAction SilentlyContinue
            }
            else {
                [Environment]::SetEnvironmentVariable($kgEnvName, $kgPriorEnv[$kgEnvName], 'Process')
            }
        }
    }
}
```

Expected: both `KG_BUILD_ENABLED` and `KG_OVERLAY_ENABLED` remain `0`; the worker and service each confirm the debug-only bypass authorization; inspection observes the exact rebuild request through all member-document builds and correlated collection activation; both reports use the same collection and exist under repository `artifacts/`; measured gates are written, reviewed/approved per the runbook ownership rule, and verification exits zero with no `PENDING_MEASUREMENT` values; pruning dry-run proves no active/building artifact is eligible; worker, database, and Redis containers are stopped; all four temporary environment overrides are restored to their prior values or removed when previously unset.

- [ ] **Step 7: Commit measured gates, then review the final diff**

After the designated owner reviews the two reports and approves the written numeric gates:

```bash
git add docs/documents/operations/knowledge-graph-overlay-runbook.md
git commit -m "docs(kg): record measured rollout gates"
```

Run: `git status --short` and `git diff --check`.

Expected: only plan-scoped files changed and no whitespace errors.

Do not create any additional verification commit when no fixes were required. If verification finds a defect, return it to the owning task/subagent, rerun that task's review gates and exact commit step, then repeat this final verification task.

---

## Execution Handoff

Implementation must use `superpowers:subagent-driven-development` with one fresh subagent per task and two-stage review. Start at Chunk 0; do not parallelize tasks that share files even if the dependency table permits conceptual concurrency. Keep `KG_BUILD_ENABLED=0` and `KG_OVERLAY_ENABLED=0` through final verification. Enabling either flag in a real environment is an operator rollout action, not an implementation-plan step.
