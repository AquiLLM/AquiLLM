# Graph RAG Development Reliability Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development task-by-task. The user approved fixing these gaps on development; production promotion is deferred.

**Goal:** Resolve the reported extractor timeouts, topology failures, and duplicate reconcile backlog on development, and verify an isolated graph-off baseline.
**Architecture:** Preserve bounded authorized retrieval and existing vector fallback. Extend the supported extractor budget, coalesce scheduled reconciliation before broker publication, and fix topology defects reproduced with the internal Q1/Q2 cases. Keep graph-off selection process-local for experiments.
**Tech Stack:** Python, Django, Celery, Redis, Memgraph, GLiNER2, pytest, Docker Compose.
**Spec:** Approved scope in the October 8 user request and the preceding diagnosis; exact incident evidence is AquiLLM-internal note-for-jack-2026-10-08.md section 4.

## Global Constraints
- Development server is 149.165.150.254; production 149.165.169.204 must remain unchanged.
- Preserve authorization, graph readiness checks, finite candidate caps, and the existing 4500 ms branch / 5000 ms overall budgets.
- Support a 3000 ms extractor timeout without changing production configuration or unconditionally extending the parent deadline.
- No deletion of live tasks, projections, documents, or embeddings. Test broker faults using isolated Redis keys/containers.
- Keep private study questions and corpus data out of tracked public repository files.
- Every behavioral change needs a failing regression test before implementation.
- Do not change embedding precision, evidence/rerank modes, model versions, or generation for this task.
- Replays must use exact study wording; graph-off on the current release is not the historical b70f58a Study 2 baseline.

### Task 1: CPU extractor deadline support
**Files:** lib/knowledge_graph/retrieval_config.py and query_extractor/config.py under aquillm; associated test_retrieval_config.py and test_query_extractor_client.py/service tests.
**Interfaces:** Both loaders must accept 3000 ms and reject invalid or excessive input. QueryExtractorClient.extract must still cap its HTTP timeout to the caller's remaining deadline.
- [ ] Add tests that load 3000 ms in both configurations and exercise success past 1000 ms but before 3000 ms using a controlled clock/transport, plus expiry at the parent deadline.
- [ ] Run focused tests and retain the expected validation failure.
- [ ] Use a supported range of 10..5000 ms consistently in both loaders; preserve existing defaults. Update any other exact timeout validation that demonstrably blocks the supported configuration.
- [ ] Verify allowed/boundary/rejected values and existing redaction, provenance, service deadline behavior.
- [ ] Commit only this task's changes and provide a test report.

### Task 2: Coalesce scheduled reconcile publication
**Files:** aquillm/aquillm/celery_schedules.py, aquillm/apps/knowledge_graph/projection/maintenance.py and tasks.py; add a focused publication module if necessary; associated maintenance/scheduling tests.
**Interfaces:** Celery scheduled reconcile remains global and scoped manual requests remain usable. Keep the existing consumer-side admission gate. Producers must not enqueue another scheduled global reconcile while its predecessor is queued/running.
- [ ] Reproduce duplicate publication when the projection worker is stopped; test concurrent producers, publish failure, completion and abandoned-worker recovery.
- [ ] Implement bounded scheduled-task coalescing with ownership tokens and recovery. A stale owner must not delete a successor's token; failure to coordinate must not trigger a publish flood.
- [ ] Ensure hung maintenance cannot indefinitely starve normal project/prune work; finite task lifetimes and priority/isolation must match the actual worker pool.
- [ ] Verify real Redis behavior in an isolated broker, scheduling compatibility, and normal projection task eligibility.
- [ ] Commit and report exactly what guarantee holds for long outages and crashes.

### Task 3: Topology reproductions and graph-off validation
**Files:** aquillm/apps/knowledge_graph/retrieval/topology/* as established by evidence; corresponding topology tests; scripts/replay_retrieval.py and docs/operations/retrieval-replay.md only if needed.
**Interfaces:** Preserve bounded authorized topology snapshots. Do not simply remove caps, bypass readiness, or silently mark invalid results successful.
- [ ] Fetch the internal incident note and exact Q1/Q2 wording; match the corresponding parent/figure collection scopes on development.
- [ ] Run retrieval-only baseline and focused topology diagnostics to identify the exact query family/cap/validation or timeout source. Record safe counts, reasons and timing.
- [ ] Add failing tests representing each proven defect, implement narrow fixes, and rerun coverage of graph authorization/deadline/fallback.
- [ ] Verify graph-off contributes zero graph candidates and restores the graph-on configuration after each run; do not describe current-release graph-off as a historical Study 2 replica.
- [ ] Commit fixes and document any genuinely unresolved issues with measured evidence.

### Task 4: Integrated development validation and rollout
**Files:** Development deployment environment and a committed audit under docs/audits/2026-10-08-graph-rag-development-reliability.md.
- [ ] Run focused unit/integration suites and independent review of the combined diff.
- [ ] Merge/push the reviewed changes to development only; preserve unrelated local drafts.
- [ ] Back up development runtime config and retain rollback images. Deploy verified application/KG images with matching revision labels; set the extractor timeout to 3000 in the app and extractor service.
- [ ] Check web readiness, worker/scheduler behavior, queue depth, and exact-case repeated retrieval with graph on/off.
- [ ] Publish a concise outcome with before/after timings and failure counts, the development revision, and remaining production promotion prerequisites.


### Task 5: Close the extractor cold-start readiness gap found during rollout
Live verification of the first development rollout showed a 3040 ms first-call timeout followed by warm extraction at approximately 230–250 ms. The service health endpoint reported healthy before lazy model initialization. Move local pinned-model initialization and one fixed synthetic warmup into a bounded ASGI lifespan startup, and admit extraction/report healthy only after success. Preserve authentication, provenance, request deadlines and inference-slot lifetime. Add lifecycle, cancellation, timeout and late-worker regressions, then review and repeat a fresh development restart/replay. Production stays unchanged.
