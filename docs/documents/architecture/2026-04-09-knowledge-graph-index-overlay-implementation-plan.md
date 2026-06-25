# Knowledge Graph Index Overlay Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a graph-as-index overlay over AquiLLM's existing chunk/vector retrieval so document, figure, collection, and meta knowledge graphs can be built asynchronously, associated with source evidence, deduplicated, pruned, rebuilt, and used to improve RAG retrieval quality.

**Architecture:** Add a dedicated `apps.knowledge_graph` Django app that persists graph artifacts, nodes, edges, evidence, associations, and build runs in Postgres beside the current `TextChunk` substrate. Documents and figures enqueue async local graph builds after chunking, collection and meta graphs are refreshed from those local artifacts, and `apps.documents.services.chunk_search` gains a fail-open graph overlay stage that expands and fuses vector-seeded evidence without replacing the current search path.

**Tech Stack:** Django 5, PostgreSQL + pgvector, Celery, structlog, pytest.

---

## Chunk 1: Create the App and Persistence Layer

### Task 1: Add failing tests for app registration and model imports

**Files:**
- Create: `aquillm/apps/knowledge_graph/tests/__init__.py`
- Create: `aquillm/apps/knowledge_graph/tests/test_graph_models.py`
- Create: `tests/integration/test_knowledge_graph_app_registration.py`
- Reference: `aquillm/aquillm/settings.py`

- [ ] **Step 1: Write a failing app-registration test**
  Assert that `apps.knowledge_graph` is present in `INSTALLED_APPS`.

- [ ] **Step 2: Write failing graph-model import tests**
  Expect `GraphArtifact`, `GraphNode`, `GraphEdge`, `GraphEvidence`, `GraphAssociation`, and `GraphBuildRun` to be importable from `apps.knowledge_graph.models`.

- [ ] **Step 3: Run focused tests to confirm failure**
  Run: `python -m pytest tests/integration/test_knowledge_graph_app_registration.py aquillm/apps/knowledge_graph/tests/test_graph_models.py -q`
  Expected: FAIL because the app and models do not exist yet.

### Task 2: Implement the app skeleton, models, and migration

**Files:**
- Create: `aquillm/apps/knowledge_graph/__init__.py`
- Create: `aquillm/apps/knowledge_graph/apps.py`
- Create: `aquillm/apps/knowledge_graph/migrations/__init__.py`
- Create: `aquillm/apps/knowledge_graph/models/__init__.py`
- Create: `aquillm/apps/knowledge_graph/models/artifact.py`
- Create: `aquillm/apps/knowledge_graph/models/build_run.py`
- Create: `aquillm/apps/knowledge_graph/models/node.py`
- Create: `aquillm/apps/knowledge_graph/models/edge.py`
- Create: `aquillm/apps/knowledge_graph/models/evidence.py`
- Create: `aquillm/apps/knowledge_graph/models/association.py`
- Modify: `aquillm/aquillm/settings.py`
- Create: `aquillm/apps/knowledge_graph/migrations/0001_initial.py`

- [ ] **Step 1: Register the new app**
  Add `"apps.knowledge_graph"` to `INSTALLED_APPS`.

- [ ] **Step 2: Implement the core models**
  Create concrete models for artifacts, nodes, edges, evidence, associations, and build runs with the minimum indexes for scope, artifact, and evidence lookup.

- [ ] **Step 3: Add the initial migration**
  Create `0001_initial.py` for the new graph tables.

- [ ] **Step 4: Run focused tests**
  Run: `python -m pytest tests/integration/test_knowledge_graph_app_registration.py aquillm/apps/knowledge_graph/tests/test_graph_models.py -q`
  Expected: PASS.

---

## Chunk 2: Build Local Document and Figure Graphs

### Task 3: Add failing tests for local graph builders

**Files:**
- Create: `aquillm/apps/knowledge_graph/tests/test_local_graph_build.py`
- Reference: `aquillm/apps/documents/tasks/chunking.py`
- Reference: `aquillm/apps/documents/models/document.py`
- Reference: `aquillm/apps/documents/models/document_types/figure.py`

- [ ] **Step 1: Write a failing document-graph test**
  Expect a document with chunks to produce one active document-scoped `GraphArtifact` with nodes, edges, and evidence tied to the document.

- [ ] **Step 2: Write a failing figure-graph test**
  Expect a `DocumentFigure` with caption and parent-document context to produce a figure-scoped graph artifact with evidence linked to the figure and relevant source chunks.

- [ ] **Step 3: Run focused tests to confirm failure**
  Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_local_graph_build.py -q`
  Expected: FAIL because local graph builder services do not exist yet.

### Task 4: Implement local graph builder services

**Files:**
- Create: `aquillm/apps/knowledge_graph/services/__init__.py`
- Create: `aquillm/apps/knowledge_graph/services/config.py`
- Create: `aquillm/apps/knowledge_graph/services/normalization.py`
- Create: `aquillm/apps/knowledge_graph/services/evidence.py`
- Create: `aquillm/apps/knowledge_graph/services/local_graphs.py`
- Test: `aquillm/apps/knowledge_graph/tests/test_local_graph_build.py`

- [ ] **Step 1: Add config helpers**
  Implement helpers such as `kg_overlay_enabled()` and `kg_overlay_fail_open()`.

- [ ] **Step 2: Implement document graph extraction**
  Build a service that reads the document and its chunks, normalizes candidate entities and relations, and persists one active document-scoped graph artifact.

- [ ] **Step 3: Implement figure graph extraction**
  Build a service that uses `DocumentFigure.extracted_caption`, `full_text`, and parent-document context to persist a figure-scoped graph artifact.

- [ ] **Step 4: Persist provenance**
  Write `GraphEvidence` rows keyed to `document_id`, optional `figure_id`, and optional `chunk_id`.

- [ ] **Step 5: Run local graph tests**
  Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_local_graph_build.py -q`
  Expected: PASS.

---

## Chunk 3: Queue Local Graph Builds from the Ingest Path

### Task 5: Add failing async-enqueue tests after chunking

**Files:**
- Create: `aquillm/apps/knowledge_graph/tests/test_graph_enqueue.py`
- Modify: `aquillm/apps/documents/tests/test_multimodal_chunk_position_uniqueness.py`
- Reference: `aquillm/apps/documents/tasks/chunking.py`

- [ ] **Step 1: Write a failing chunking-enqueue test**
  Patch the graph task entry point and expect `create_chunks()` to enqueue a document graph build after successful chunk persistence.

- [ ] **Step 2: Write a failing figure-enqueue test**
  Verify figure graph builds are queued for `DocumentFigure` sources after chunking completes.

- [ ] **Step 3: Run focused tests**
  Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_graph_enqueue.py aquillm/apps/documents/tests/test_multimodal_chunk_position_uniqueness.py -q`
  Expected: FAIL because chunking does not yet enqueue graph jobs.

### Task 6: Implement Celery tasks and chunking hooks

**Files:**
- Create: `aquillm/apps/knowledge_graph/tasks/__init__.py`
- Create: `aquillm/apps/knowledge_graph/tasks/builds.py`
- Modify: `aquillm/apps/documents/tasks/chunking.py`
- Test: `aquillm/apps/knowledge_graph/tests/test_graph_enqueue.py`

- [ ] **Step 1: Add task entry points**
  Implement `build_document_graph(doc_id: str)` and `build_figure_graph(figure_id: str)`.

- [ ] **Step 2: Hook chunking completion to graph queueing**
  After successful `TextChunk.objects.bulk_create(chunks)`, enqueue local graph work behind `KG_OVERLAY_ENABLED`.

- [ ] **Step 3: Preserve fail-open ingest behavior**
  If graph queueing fails, log a warning and keep document ingestion successful.

- [ ] **Step 4: Run enqueue tests**
  Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_graph_enqueue.py aquillm/apps/documents/tests/test_multimodal_chunk_position_uniqueness.py -q`
  Expected: PASS.

---

## Chunk 4: Build Collection Graphs, Meta Graph Promotion, and Associations

### Task 7: Add failing tests for collection aggregation and meta promotion

**Files:**
- Create: `aquillm/apps/knowledge_graph/tests/test_collection_meta_graph.py`
- Reference: `aquillm/apps/collections/models/collection.py`

- [ ] **Step 1: Write a failing collection-graph test**
  Expect two local graphs in one collection to produce a collection-scoped artifact with merged concepts and `GraphAssociation` rows.

- [ ] **Step 2: Write a failing meta-promotion test**
  Expect collection-backed concepts above threshold to be promoted into a meta-scoped artifact with provenance back to collection or local nodes.

- [ ] **Step 3: Run focused tests**
  Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_collection_meta_graph.py -q`
  Expected: FAIL because aggregation and promotion services do not exist yet.

### Task 8: Implement collection aggregation and meta promotion

**Files:**
- Create: `aquillm/apps/knowledge_graph/services/association.py`
- Create: `aquillm/apps/knowledge_graph/services/collection_graph.py`
- Create: `aquillm/apps/knowledge_graph/services/meta_graph.py`
- Create: `aquillm/apps/knowledge_graph/services/dedupe.py`
- Modify: `aquillm/apps/knowledge_graph/tasks/builds.py`
- Test: `aquillm/apps/knowledge_graph/tests/test_collection_meta_graph.py`

- [ ] **Step 1: Implement collection graph refresh**
  Gather active local artifacts in one collection, apply exact and bounded semantic dedupe, and write one active collection-scoped artifact.

- [ ] **Step 2: Persist associations**
  Link local nodes to collection nodes and collection nodes to promoted canonical nodes through `GraphAssociation`.

- [ ] **Step 3: Implement meta graph promotion**
  Promote only collection-backed concepts above a bounded threshold into a single active meta artifact.

- [ ] **Step 4: Extend Celery tasks**
  Add `refresh_collection_graph(collection_id: int)` and `refresh_meta_graph()`.

- [ ] **Step 5: Run aggregation tests**
  Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_collection_meta_graph.py -q`
  Expected: PASS.

---

## Chunk 5: Implement Deduplication, Pruning, Invalidation, and Rebuild Controls

### Task 9: Add failing lifecycle-maintenance tests

**Files:**
- Create: `aquillm/apps/knowledge_graph/tests/test_pruning.py`
- Modify: `aquillm/apps/documents/models/document.py`
- Modify: `aquillm/apps/collections/models/collection.py`

- [ ] **Step 1: Write a failing prune test**
  Assert that unsupported or superseded nodes and edges are pruned from active retrieval consideration while keeping rebuildable history or build-run records.

- [ ] **Step 2: Write a failing move/delete invalidation test**
  Assert document delete and document move mark the correct collection artifacts stale and queue refresh for the affected scopes.

- [ ] **Step 3: Run focused tests**
  Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_pruning.py -q`
  Expected: FAIL because pruning and invalidation hooks do not exist yet.

### Task 10: Implement pruning and invalidation paths

**Files:**
- Create: `aquillm/apps/knowledge_graph/services/pruning.py`
- Create: `aquillm/apps/knowledge_graph/services/invalidation.py`
- Create: `aquillm/apps/knowledge_graph/management/__init__.py`
- Create: `aquillm/apps/knowledge_graph/management/commands/__init__.py`
- Create: `aquillm/apps/knowledge_graph/management/commands/rebuild_graph_overlay.py`
- Create: `aquillm/apps/knowledge_graph/management/commands/prune_graph_overlay.py`
- Modify: `aquillm/apps/documents/models/document.py`
- Modify: `aquillm/apps/collections/models/collection.py`
- Modify: `aquillm/apps/knowledge_graph/tasks/builds.py`
- Test: `aquillm/apps/knowledge_graph/tests/test_pruning.py`

- [ ] **Step 1: Implement artifact invalidation helpers**
  Mark impacted graph artifacts `stale` or `superseded` and queue collection/meta refresh.

- [ ] **Step 2: Hook document lifecycle changes**
  Update `move_to()` and `delete()` paths so graph invalidation follows collection changes and document removal.

- [ ] **Step 3: Implement prune services**
  Prune unsupported edges, orphan nodes, and superseded artifacts beyond retention policy.

- [ ] **Step 4: Add management commands**
  Implement commands to rebuild a document, collection, or all graph artifacts and to run a prune sweep manually.

- [ ] **Step 5: Run lifecycle tests**
  Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_pruning.py -q`
  Expected: PASS.

---

## Chunk 6: Integrate Graph Overlay Retrieval into Chunk Search

### Task 11: Add failing retrieval-overlay tests

**Files:**
- Create: `aquillm/apps/documents/tests/test_chunk_search_graph_overlay.py`
- Reference: `aquillm/apps/documents/services/chunk_search.py`
- Reference: `aquillm/apps/chat/services/tool_wiring/documents.py`

- [ ] **Step 1: Write a failing graph-expansion test**
  Assert that vector-seeded chunk results can pull in additional graph-supported chunks from related document or figure evidence when the overlay is enabled.

- [ ] **Step 2: Write a failing fail-open test**
  Assert `text_chunk_search()` still returns the current vector/trigram path when the graph overlay raises an error.

- [ ] **Step 3: Run focused tests**
  Run: `python -m pytest aquillm/apps/documents/tests/test_chunk_search_graph_overlay.py -q`
  Expected: FAIL because no retrieval overlay stage exists yet.

### Task 12: Implement retrieval overlay scoring and bounded expansion

**Files:**
- Create: `aquillm/apps/knowledge_graph/services/retrieval_overlay.py`
- Modify: `aquillm/apps/documents/services/chunk_search.py`
- Optional Modify: `aquillm/apps/chat/services/tool_wiring/documents.py`
- Test: `aquillm/apps/documents/tests/test_chunk_search_graph_overlay.py`

- [ ] **Step 1: Implement overlay input/output types**
  Accept the query, seed chunks, and visible documents, then return an expanded ordered chunk list plus optional debug metadata.

- [ ] **Step 2: Implement bounded graph expansion**
  Traverse `GraphEvidence`, `GraphAssociation`, and active graph edges with hard caps on hops and fan-out.

- [ ] **Step 3: Fuse vector and graph scores**
  Blend existing seed rank with graph-support weights while preserving deterministic ordering and fail-open fallback.

- [ ] **Step 4: Integrate into `text_chunk_search()`**
  Run the overlay after the current dedupe stage and before final result return, guarded by `KG_OVERLAY_ENABLED`.

- [ ] **Step 5: Keep tool outputs backward compatible**
  Only add optional metadata if it does not break `pack_chunk_search_results()` consumers.

- [ ] **Step 6: Run retrieval-overlay tests**
  Run: `python -m pytest aquillm/apps/documents/tests/test_chunk_search_graph_overlay.py aquillm/apps/documents/tests/test_chunk_search_query_cache.py -q`
  Expected: PASS.

---

## Chunk 7: Add Operator Visibility, Config, and Documentation

### Task 13: Add failing config and build-run tests

**Files:**
- Create: `aquillm/apps/knowledge_graph/tests/test_graph_config.py`
- Reference: `aquillm/aquillm/settings.py`

- [ ] **Step 1: Write a failing config test**
  Add tests for new overlay env defaults and fail-open behavior.

- [ ] **Step 2: Write a failing build-run visibility test**
  Assert graph tasks record `GraphBuildRun` rows with status and stats.

- [ ] **Step 3: Run focused tests**
  Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_graph_config.py -q`
  Expected: FAIL because config helpers and build-run tracking are incomplete.

### Task 14: Document the system and wire config

**Files:**
- Modify: `.env.example`
- Modify: `deploy/compose/development.yml`
- Modify: `deploy/compose/production.yml`
- Modify: `docs/documents/README.md`
- Create: `docs/documents/operations/knowledge-graph-overlay-runbook.md`
- Modify: `docs/documents/architecture/2026-04-09-knowledge-graph-index-overlay-design.md`
- Test: `aquillm/apps/knowledge_graph/tests/test_graph_config.py`

- [ ] **Step 1: Add environment examples**
  Document `KG_OVERLAY_*` settings in `.env.example`.

- [ ] **Step 2: Add compose defaults**
  Wire conservative off-by-default overlay flags into development and production compose files.

- [ ] **Step 3: Add the runbook**
  Document rebuild, prune, stale-graph inspection, and fail-open operating procedures.

- [ ] **Step 4: Update the documents index**
  Add links to the new design, plan, and runbook files.

- [ ] **Step 5: Run config tests**
  Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_graph_config.py -q`
  Expected: PASS.

### Task 14A: Document Mem0 compatibility and future convergence points

**Files:**
- Modify: `docs/documents/architecture/2026-04-09-knowledge-graph-index-overlay-design.md`
- Reference: `docs/specs/2026-03-30-self-hosted-mem0-graph-memory-design.md`
- Reference: `docs/specs/2026-03-31-mem0-graph-memory-quality-tuning-design.md`

- [ ] **Step 1: Add an explicit Mem0 integration note**
  Document how the graph overlay should coexist with Mem0 in the near term, where shared normalization, relation filtering, and graph-quality heuristics may be reused, and how stable overlay facts might later promote into Mem0 to grow durable model knowledge over time.

- [ ] **Step 2: Keep boundaries explicit**
  State that Mem0 remains memory-oriented while the graph overlay remains corpus- and retrieval-oriented until a later design intentionally converges them.

---

## Chunk 8: Final Verification

### Task 15: Verify the touched surface

**Files:**
- Modify/Create all files above

- [ ] **Step 1: Run the focused graph and chunk-search suites**
  Run: `python -m pytest aquillm/apps/knowledge_graph/tests aquillm/apps/documents/tests/test_chunk_search_graph_overlay.py aquillm/apps/documents/tests/test_chunk_search_query_cache.py -q`
  Expected: PASS.

- [ ] **Step 2: Run the broader document and integration suites**
  Run: `python -m pytest aquillm/apps/documents/tests tests/integration/test_knowledge_graph_app_registration.py -q`
  Expected: PASS.

- [ ] **Step 3: Run Django checks**
  Run: `python manage.py check`
  Expected: no app registration or model errors.

- [ ] **Step 4: Smoke-test management commands**
  Run: `python manage.py rebuild_graph_overlay --help`
  Run: `python manage.py prune_graph_overlay --help`
  Expected: both commands load successfully.
