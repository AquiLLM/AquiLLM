# Knowledge Graph Audit Remediation Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development with regression tests before implementation and independent review before publication.

**Goal:** Correct all twelve audit findings and verify the development deployment before end-user testing.

**Architecture:** Preserve the existing PostgreSQL authority, fixed Memgraph projection, and hybrid retrieval design. Repair contracts at authorization, identifier, schema, extraction, and worker boundaries; use bounded recovery and queries rather than broad rewrites.

**Tech Stack:** Python/Django, Celery, PostgreSQL/pgvector, Memgraph, React/TypeScript.

**Spec:** `docs/audits/2026-09-21-knowledge-graph/report.md`; the user's approval of its repair priorities and request to prepare the development box.

## Global constraints

- Work on the explicitly requested `development` branch. Preserve existing work.
- Commit and push reviewed local changes to `development` before pulling on the remote development box.
- Never commit environment secrets or SSH private keys. Stage explicit source paths and inspect the staged changes and outgoing commit range before push.
- Prefix shell commands with `rtk`. Use `rtk proxy` for unsupported programs.
- Reproduce the intended fixed behavior with a failing regression before modifying application code.
- Do not use real credentials for local tests. Do not modify production or discard remote work.
- Each implementation lane owns separate files. Root alone handles Git commits/push and deployment.
- Live permission, worker-loss, database, projection, and retrieval checks are required before claiming end-user readiness.

## Task 1: Authorization and query identifier contracts (findings 1, 2, 8)

**Files:** `aquillm/apps/documents/services/chunk_search.py`, `hybrid_graph_authorization.py`, `hybrid_graph_dependencies.py`; `aquillm/apps/knowledge_graph/retrieval/direct_seed_repository.py`, `production_direct.py`, `production_extended.py`; relevant tests; a focused seed-query module if necessary.

**Interfaces:** Keep public retrieval result formats and graph fusion unchanged. Raw UUIDs and opaque generation keys must be separately represented. Direct canonical lookup must encode the same integer canonical entity ID as projection construction. Extended seed lookup must bind selected chunk IDs, current membership authority, artifact/version provenance, and document authorization.

- [ ] Convert audit query probes into regressions expecting an empty pool after revocation and equal query/projection keys for both identity modes. Include a dependency-construction exception case.
- [ ] Run the regressions and observe failure, then reauthorize every fallback before reranking/output.
- [ ] Share matching encoding inputs and test actual projection-to-query identity compatibility.
- [ ] Replace whole-projection loading for extended seeds with a bounded lookup restricted to the selected seed chunks and ready generation. Test that unrelated rows are not loaded and stale or unauthorized mappings are rejected.
- [ ] Run affected query/authorization/projection parity tests and report changes for independent review.

## Task 2: Schema generation and mutation correctness (findings 3, 5, 7)

**Files:** `aquillm/apps/collections/tasks/schema_generation.py`, `services/schema.py`, `views/schema_api.py`; `aquillm/apps/knowledge_graph/services/ontology.py`; collection schema React API/editor call sites and related tests.

**Interfaces:** Definition mutations must include draft UUID plus revision. The schema naming contract must enforce canonical snake_case, maximum 64 characters, and provider-reserved names consistently. Published version/history remains immutable.

- [ ] Add failing regressions for a worker-loss redelivery during a live lease, a generation request after lease expiry, overlong/provider-reserved type names, and a stale editor saving against a replacement draft.
- [ ] Retain/retry a live-lease delivery until it can be safely reclaimed; provide safe expired-run/API recovery without duplicate ownership.
- [ ] Validate names before activation and before provider use, returning structured draft validation errors.
- [ ] Bind PUT/DELETE to draft UUID and revision; update all frontend and fixture callers and API documentation together.
- [ ] Run affected schema task/API/ontology and frontend tests, then report for review.

## Task 3: Extraction semantics and resolution work (findings 6, 9)

**Files:** `aquillm/lib/knowledge_graph/extractors/gliner2_local.py`, `aquillm/apps/knowledge_graph/extraction/pipeline.py`, `resolution/collection.py`; extraction and resolution tests.

**Interfaces:** Directed relation behavior is unchanged. Undirected relations accept either jointly valid endpoint orientation, with grounding and ambiguity checks preserved.

- [ ] Add failing tests for forward/reversed undirected relations with asymmetric endpoint type sets at provider and pipeline boundaries; include invalid and directed controls.
- [ ] Resolve both allowed endpoint orientations consistently.
- [ ] Add a behavior/operation-count regression for representative selection across repeated-label groups, then replace the Cartesian scan with the equivalent deterministic minimum pair.
- [ ] Run extraction, span, graph assembly, and resolution tests and report for review.

## Task 4: Durable scheduling and projection maintenance (findings 4, 10, 11, 12)

**Files:** KG graph/build scheduling modules and tasks as needed; `projection/tasks.py`, `generation_audit.py`, `reconciler.py`, management commands, and regression tests. Migrations only if durable state cannot use existing authority safely.

**Interfaces:** Ingestion succeeds independently of broker availability while retaining recoverable graph intent. Projection worker roles retain their function-only write restriction. Maintenance remains paged, retryable, and retains audit history.

- [ ] Add failed-publication recovery tests and implement durable intent or bounded reconciliation of ingested documents lacking current graph artifacts.
- [ ] Test initial and newly created projection outbox work; publish both and continue bounded backlogs. Management recovery must trigger dispatch too.
- [ ] Test schema/format/key version rollover; detect incompatible authority before loading its bundle and replay it without aborting unrelated collections.
- [ ] Test pruning more than one page and repeated calls; add stable cursor/completion semantics so already deleted rows cannot starve later generations or orphans.
- [ ] Verify role constraints, dry runs, retention, retry, and management command behavior.

## Task 5: Review, development publication, and end-user readiness

- [ ] Independently review implementation lanes and the combined changes; resolve important findings.
- [ ] Run targeted tests, integration tests available locally, lint/type checks appropriate to edited files, migration checks, and frontend validation.
- [ ] Inspect ignored secret paths, staged files, and every outgoing commit; commit reviewed work and push `development` without force.
- [ ] Connect using the temporary key after receiving hostname/user/port. Inspect remote branch/status and service health before changing anything.
- [ ] Pull the pushed development revision, apply needed migrations/builds, and restart affected development services without printing environment values.
- [ ] Run live schema publication, build recovery, graph identity/traversal, permission revocation, and representative question/answer tests on development-only fixtures.
- [ ] Record actual pass/fail evidence and any remaining gates. End-user readiness is conditional on these live checks passing.
