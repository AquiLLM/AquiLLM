# Knowledge graph bulk upload repair implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development. Root coordinates shared contracts, verification, and deployment; workers own disjoint implementation areas.

**Goal:** Complete graph processing for the uploaded research collection and remove the identified bulk ingestion failure modes.

**Architecture:** Preserve immutable document, ontology, draft, lease, and projection authority checks. Scale complete document evidence and collection projection within explicit aggregate limits, isolate schema work from extraction backfill, and expose automatic build progress in the existing collection graph UI.

**Tech stack:** Django/PostgreSQL, Celery/Redis, GLiNER, Memgraph, React, Docker Compose.

**Spec:** The observed failure and required behavior below are the repair specification approved by the user's request to fix the diagnosed gaps.

## Evidence and required behavior

Collection 226 contains 30 PDFs and one report, with 2,168 embedded chunks. At the start of this repair 24 document graphs failed and seven were active. A 512-mention limit rejects normal papers, and a 4,096 raw-observation limit rejects longer documents. Resolution and projection have additional incompatible ceilings. No source documents need re-uploading.

A schema request made during uploads failed its source fence. A second request succeeded after waiting about 22 minutes behind extraction backfill. Generation must settle on a stable collection snapshot without overwriting intervening draft edits. Work must remain bounded and fail clearly if the source never settles.

Automatic document failures currently look like an empty collection graph because the status API only considers explicit rebuild requests. Users need counts and safe failure categories. Status polling must continue across multiple unchanged building responses.

## Constraints

- Commit and push reviewed source to `development` before pulling the development host.
- Never commit environment secrets, SSH private keys, source documents, or raw private logs.
- Never truncate evidence silently or mark incomplete graphs complete.
- Keep collection assembly dependent on every eligible document graph.
- Retain separate restricted projection source/state roles and current-source/lease fences.
- Test regressions before implementation and verify the actual uploaded collection after deployment.

## Task 1: Stable schema scheduling

Files: `aquillm/apps/collections/tasks/schema_generation.py`, associated schema service/API tests, and five `deploy/compose/*.yml` variants.

- [x] Reproduce request-time source changes and starvation routing with failing tests.
- [x] Route schema tasks to a dedicated queue and worker with minimal credentials.
- [x] Refresh the source snapshot only before inference under ownership and unchanged draft identity; defer incomplete ingestion and source changes within a durable bounded settling budget.
- [x] Keep the final immutable snapshot check, lease fencing, and draft UUID/revision check.
- [x] Run schema lifecycle, API, lease, and Compose tests.

## Task 2: Complete bounded paper extraction

Files: `aquillm/apps/knowledge_graph/extraction/pipeline.py`, document coreference and persistence, `services/builds.py`, and regression tests.

- [x] Reproduce failures above 512 retained mentions and 4,096 observations.
- [x] Remove quadratic resolution work before increasing the compatible aggregate document budget; retain every validated mention and relation below that budget.
- [x] Bind changed extraction/resolution behavior to build identity and commit validation.
- [x] Persist safe, specific capacity failure codes.
- [x] Verify overlap deduplication, relation evidence, deterministic resolution, overflow rejection, and large synthetic paper performance.

## Task 3: Collection projection compatibility

Files: Django projection readers, restricted state repository, additive migration 0011, and projection regressions.

- [x] Reproduce projection truncation/rejection above 4,999 family rows and 5,000 chunk references.
- [x] Use bounded fetch/write pages with explicit total ceilings matching the existing projection protocol.
- [x] Preserve full-family checksum validation and reject aggregate overflow.
- [x] Update the narrow SQL chunk-reference fence without expanding role authority.
- [x] Verify real PostgreSQL restricted roles, multi-page persistence, lease loss, and source mutation behavior.

## Task 4: Visible automatic build progress

Files: `aquillm/apps/collections/services/graph_visualization.py`, a focused progress service, visualization API/React types/component/tests.

- [x] Reproduce automatic failures being reported as empty and polling stopping after one unchanged response.
- [x] Count collection-scoped ingestion and document build states without loading document content.
- [x] Show safe failure categories and progress while retaining existing permission checks and bounded visualization output.
- [x] Verify failure, pending, empty, authorization, and repeated polling cases.

## Task 5: Integration, deployment, and collection repair

- [x] Review all changes, run targeted PostgreSQL and frontend checks, and scan outgoing files for secrets.
- [ ] Commit/push development, then pull/rebuild/migrate/restart the development host.
- [ ] Replay exact current collection document builds through normal service entry points; preserve the user's generated draft and any later edits.
- [ ] Verify all uploaded documents, collection assembly, projection readiness, real retrieval, and cited answers. Report any remaining limitation explicitly.
- [ ] Record evidence and final revisions; revoke only the temporary key created for this repair after work completes.
