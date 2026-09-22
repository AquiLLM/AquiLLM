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
- Live schema, permission, database, projection, recovery, retrieval, and cited-answer checks are required for controlled development testing. Worker-loss and destructive race cases use isolated tests; full-stack fault and load qualification remains a separate production gate.

## Task 1: Authorization and query identifier contracts (findings 1, 2, 8)

**Files:** `aquillm/apps/documents/services/chunk_search.py`, `hybrid_graph_authorization.py`, `hybrid_graph_dependencies.py`; `aquillm/apps/knowledge_graph/retrieval/direct_seed_repository.py`, `production_direct.py`, `production_extended.py`; relevant tests; a focused seed-query module if necessary.

**Interfaces:** Keep public retrieval result formats and graph fusion unchanged. Raw UUIDs and opaque generation keys must be separately represented. Direct canonical lookup must encode the same integer canonical entity ID as projection construction. Extended seed lookup must bind selected chunk IDs, current membership authority, artifact/version provenance, and document authorization.

- [x] Convert audit query probes into regressions expecting an empty pool after revocation and equal query/projection keys for both identity modes. Include a dependency-construction exception case.
- [x] Run the regressions and observe failure, then reauthorize every fallback before reranking/output.
- [x] Share matching encoding inputs and test actual projection-to-query identity compatibility.
- [x] Replace whole-projection loading for extended seeds with a bounded lookup restricted to the selected seed chunks and ready generation. Test that unrelated rows are not loaded and stale or unauthorized mappings are rejected.
- [x] Run affected query/authorization/projection parity tests and report changes for independent review.

## Task 2: Schema generation and mutation correctness (findings 3, 5, 7)

**Files:** `aquillm/apps/collections/tasks/schema_generation.py`, `services/schema.py`, `views/schema_api.py`; `aquillm/apps/knowledge_graph/services/ontology.py`; collection schema React API/editor call sites and related tests.

**Interfaces:** Definition mutations must include draft UUID plus revision. The schema naming contract must enforce canonical snake_case, maximum 64 characters, and provider-reserved names consistently. Published version/history remains immutable.

- [x] Add failing regressions for a worker-loss redelivery during a live lease, a generation request after lease expiry, overlong/provider-reserved type names, and a stale editor saving against a replacement draft.
- [x] Retain/retry a live-lease delivery until it can be safely reclaimed; provide safe expired-run/API recovery without duplicate ownership.
- [x] Validate names before activation and before provider use, returning structured draft validation errors.
- [x] Bind PUT/DELETE to draft UUID and revision; update all frontend and fixture callers and API documentation together.
- [x] Run affected schema task/API/ontology and frontend tests, then report for review.

## Task 3: Extraction semantics and resolution work (findings 6, 9)

**Files:** `aquillm/lib/knowledge_graph/extractors/gliner2_local.py`, `aquillm/apps/knowledge_graph/extraction/pipeline.py`, `resolution/collection.py`; extraction and resolution tests.

**Interfaces:** Directed relation behavior is unchanged. Undirected relations accept either jointly valid endpoint orientation, with grounding and ambiguity checks preserved.

- [x] Add failing tests for forward/reversed undirected relations with asymmetric endpoint type sets at provider and pipeline boundaries; include invalid and directed controls.
- [x] Resolve both allowed endpoint orientations consistently.
- [x] Add a behavior/operation-count regression for representative selection across repeated-label groups, then replace the Cartesian scan with the equivalent deterministic minimum pair.
- [x] Run extraction, span, graph assembly, and resolution tests and report for review.

## Task 4: Durable scheduling and projection maintenance (findings 4, 10, 11, 12)

**Files:** KG graph/build scheduling modules and tasks as needed; `projection/tasks.py`, `generation_audit.py`, `reconciler.py`, management commands, and regression tests. Migrations only if durable state cannot use existing authority safely.

**Interfaces:** Ingestion succeeds independently of broker availability while retaining recoverable graph intent. Projection worker roles retain their function-only write restriction. Maintenance remains paged, retryable, and retains audit history.

- [x] Add failed-publication recovery tests and implement durable intent or bounded reconciliation of ingested documents lacking current graph artifacts.
- [x] Test initial and newly created projection outbox work; publish both and continue bounded backlogs. Management recovery must trigger dispatch too.
- [x] Test schema/format/key version rollover; detect incompatible authority before loading its bundle and replay it without aborting unrelated collections.
- [x] Test pruning more than one page and repeated calls; add stable cursor/completion semantics so already deleted rows cannot starve later generations or orphans.
- [x] Verify role constraints, dry runs, retention, retry, and management command behavior.

## Task 5: Review, development publication, and end-user readiness

- [x] Independently review implementation lanes and the combined changes; resolve important findings.
- [x] Run targeted tests, integration tests available locally, lint/type checks appropriate to edited files, migration checks, and frontend validation.
- [x] Inspect ignored secret paths, staged files, and every outgoing commit; commit reviewed work and push `development` without force.
- [x] Connect using the temporary key after receiving hostname/user/port. Inspect remote branch/status and service health before changing anything.
- [x] Pull the pushed development revision, apply needed migrations/builds, and restart affected development services without printing environment values.
- [x] Run live schema publication, build recovery, graph identity/traversal, permission revocation, and representative question/answer tests on development-only fixtures.
- [x] Record actual pass/fail evidence and any remaining gates. End-user readiness is conditional on these live checks passing.

## Live integration follow-up: custom schema query extraction

The development run exposed additional connections absent from isolated tests:
web's required read-only projection source connection was blanked; direct ontology
selection counted unrelated active collection schemas; the client posted to the
extractor's service origin; and the extractor protocol carried only the deployment
ontology checksum, preventing direct queries against published custom schemas.
The connection and exact ontology-selection fixes have already been committed,
pulled, and verified on development.

The protocol repair adds an optional canonical ontology definition to the existing
authenticated request. Legacy checksum-only requests remain supported. Validate
the full definition and recomputed checksum before loading the backend, with a
64 KiB definition cap, 64 entity types, 128 relations, bounded aliases/descriptions,
and a 128 KiB total request cap. Inference receives immutable request-local schema
state; model/build/schema/span provenance checks remain. The client accepts the
documented service-origin URL by selecting `/v1/extract` when no path is provided.
Deploy the compatible extractor service before the updated client.

- [x] Verify tampering, malformed names/endpoints, oversized input, legacy requests,
  schema isolation between requests, and checksum parity with persisted ontologies.
- [x] Independently review the protocol and resolve findings before committing.
- [x] Commit/push, pull/rebuild, and query both the default-schema fixture and the
  actually generated/published collection schema through direct and extended paths.

## Completion evidence

The application revision `525b71a3823ea729f7ddeb7dbb3f96e3b52204ab` is deployed on
development with persistent graph retrieval and maintenance flags enabled. Both
schema fixtures passed real direct/extended retrieval and hybrid search. The
configured LLM returned the fixture fact with valid citations; revoked permission
blocked access; real restricted database roles passed their guards. The scheduler
dispatched both jobs, with 25 successful recovery pages and one successful
projection reconciliation observed after the final deployment and no task failures
in those maintenance paths.

Validation scope was narrowed to safe live fixture checks on the shared development
box. Worker-loss and destructive pruning/race cases ran in isolated regressions;
full-stack chaos/load qualification remains a production gate. The cited-answer
probe covered the in-memory service path, not live browser WebSocket transport.
Older backlog documents that exceed the entity cap remain failed for review.
See [the remediation record](../../audits/2026-09-21-knowledge-graph/remediation.md)
for test evidence, operational settings, and remaining limitations.
