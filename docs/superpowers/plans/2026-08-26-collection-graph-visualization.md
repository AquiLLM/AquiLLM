# Collection Graph Reliability and Visualization Implementation Plan

> **For Codex:** Execute this plan in order with test-driven development. Keep
> remote environment files and `.codex-ssh/` out of every commit.

**Goal:** Repair collection graph builds for unsafe PDF control characters and
add an evidence-oriented Schema/Instance visualization tab to collection pages.

**Architecture:** A shared, length-preserving graph source sanitizer feeds both
extraction span mapping and resolver chunk context. A permission-checked Django
API exposes a bounded active PostgreSQL graph and its lifecycle state. A React
feature maps schema or instance data into Cytoscape.js and provides accessible
selection/evidence details.

**Tech stack:** Django, PostgreSQL/pgvector graph models, Celery, React 19,
TypeScript, Vitest/Testing Library, Cytoscape.js, pytest.

---

## Task 1: Add length-preserving graph source sanitization

**Files:**

- Modify: `aquillm/apps/knowledge_graph/extraction/windows.py`
- Modify: `aquillm/apps/knowledge_graph/extraction/pipeline.py`
- Modify: `aquillm/apps/knowledge_graph/resolution/coreference.py`
- Test: `aquillm/apps/knowledge_graph/tests/test_extraction_windows.py`
- Test: `aquillm/apps/knowledge_graph/tests/test_coreference.py`

1. Add failing tests proving disallowed controls are replaced one-for-one, text
   whitespace remains intact, mapped spans stay unchanged, chunk-derived source
   context is sanitized, and explicit unsafe resolver source text is rejected.
2. Run only those tests and confirm the expected failures.
3. Implement a shared sanitizer and apply it to derived extraction windows,
   full-text span validation, and chunk-derived resolver context.
4. Re-run the focused tests and confirm they pass.

## Task 2: Harden raw observation caps without truncating unique evidence

**Files:**

- Modify: `aquillm/apps/knowledge_graph/extraction/pipeline.py`
- Test: `aquillm/apps/knowledge_graph/tests/test_extraction_pipeline.py`

1. Add failing tests for overlapping raw observations exceeding 512 while the
   deterministic deduplicated entity set remains within 512, plus a genuine
   deduplicated overflow case.
2. Confirm the first test fails under the pre-materialization cap.
3. Introduce a separate bounded raw observation cap, enforce the existing 512
   cap after mapping/deduplication, and retain explicit failure for true unique
   overflow. Do not truncate.
4. Re-run focused extraction tests.

## Task 3: Add the bounded collection graph visualization API

**Files:**

- Create: `aquillm/apps/collections/services/graph_visualization.py`
- Create: `aquillm/apps/collections/views/graph_api.py`
- Modify: `aquillm/apps/collections/views/__init__.py`
- Modify: `aquillm/aquillm/api_views.py`
- Modify: `aquillm/aquillm/context_processors.py`
- Create: `aquillm/apps/collections/tests/test_graph_visualization_api.py`
- Modify: `aquillm/tests/integration/test_context_processors_urls.py`

1. Add failing API tests for authentication, collection view permissions,
   ready/empty/partial/failed/building states, deterministic node/edge ordering,
   150/300 caps, focus search, and evidence scoping.
2. Add a failing context-route test.
3. Implement a query-bounded service over current `CollectionEntity`,
   `CollectionRelation`, entity-document links, and relation evidence. Avoid
   per-node queries and expose only safe error codes and bounded source excerpts.
4. Add GET visualization and POST rebuild routes. Rebuild requires edit access
   and uses the existing durable rebuild service.
5. Re-run API and URL tests.

## Task 4: Extend collection navigation for Visualization

**Files:**

- Modify: `react/src/features/collections/components/collectionViewTypes.ts`
- Modify: `react/src/features/collections/components/CollectionModeNav.tsx`
- Modify: `react/src/features/collections/components/CollectionViewShell.tsx`
- Modify: `react/src/features/collections/components/CollectionModeNav.test.tsx`
- Modify: `react/src/features/collections/components/CollectionViewShell.test.tsx`

1. Add failing tests for parsing, rendering, selecting, and deep-linking the
   `visualization` mode without triggering the schema draft navigation guard.
2. Implement the third mode and shell slot.
3. Re-run the focused component tests.

## Task 5: Add typed graph data and schema mapping

**Files:**

- Create: `react/src/features/collections/visualization/collectionGraphTypes.ts`
- Create: `react/src/features/collections/visualization/collectionGraphApi.ts`
- Create: `react/src/features/collections/visualization/schemaGraphMapper.ts`
- Create: `react/src/features/collections/visualization/collectionGraphApi.test.ts`
- Create: `react/src/features/collections/visualization/schemaGraphMapper.test.ts`

1. Add failing tests for strict response parsing, URL placeholder substitution,
   abort behavior, and deterministic schema node/edge mapping.
2. Implement the smallest typed API and mapper that satisfy the contract.
3. Re-run focused Vitest tests.

## Task 6: Build the interactive Schema/Instance visualization workspace

**Files:**

- Create: `react/src/features/collections/visualization/CollectionGraphCanvas.tsx`
- Create: `react/src/features/collections/visualization/CollectionGraphVisualization.tsx`
- Create: `react/src/features/collections/visualization/CollectionGraphVisualization.test.tsx`
- Modify: `react/package.json`
- Modify: `react/package-lock.json`
- Modify: `react/src/features/collections/components/CollectionViewShell.tsx`

1. Add failing UI tests for Schema/Instance switching, loading, ready, partial,
   failed and empty states, filters, selection details, evidence expansion,
   rebuild permission, and graph-library cleanup.
2. Install Cytoscape.js and its TypeScript types if required by the installed
   release.
3. Implement the workspace, an accessible list fallback, and a canvas wrapper
   that destroys the Cytoscape instance on unmount.
4. Re-run visualization and collection shell tests.

## Task 7: Verify locally, commit, push, deploy, and rebuild

1. Run focused backend knowledge-graph and visualization API tests.
2. Run focused frontend tests, TypeScript checking, and the production build.
3. Run adjacent regression suites for collection schema editing and graph
   extraction/resolution.
4. Inspect the final diff and verify `.env` and `.codex-ssh/` are excluded.
5. Commit implementation on `development` and push `origin/development`.
6. On `aquillm-dev2`, verify the expected branch/commit, pull, and rebuild only
   web and applicable knowledge-graph workers using the server's existing env
   and compose override. Do not deploy transcription.
7. Reload nginx after healthy services, queue collection 217 rebuild, monitor
   both documents and collection projection, and verify nonzero nodes, edges,
   evidence links, and the public visualization endpoint.
