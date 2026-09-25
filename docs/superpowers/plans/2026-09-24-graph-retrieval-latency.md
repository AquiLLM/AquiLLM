# Graph retrieval latency implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development, with one implementer at a time and a task review after each task.

**Goal:** Reduce graph retrieval latency without changing successful retrieval scope, candidates, PageRank scores, or ranking.

**Architecture:** Batch equivalent database reads, overlap the independent direct branch with baseline passage retrieval, and publish aggregate branch outcomes. Preserve query, graph, row, and timeout bounds and current authorization validation. Keep the gateway wire contract unchanged when database-side batching is sufficient.

**Tech stack:** Django/PostgreSQL, Memgraph, bounded Python futures, pytest, Docker Compose.

**Spec:** User-approved four changes in this conversation: batch chunk-to-entity lookups; start direct graph retrieval alongside ordinary search; reduce graph database round trips; record branch outcomes. Successful non-timeout results must be equivalent; newly completed branches are evaluated separately.

## Global constraints

- Preserve authorization and current projection/generation validation before returning evidence.
- Preserve deterministic ordering, full-source overflow detection, seed weights, topology caps, PPR policy, and final reranking.
- Do not enlarge existing graph deadlines, result caps, or inference budgets to manufacture a speedup.
- Development retains adaptive PPR and adaptive selection, 12 final passages, approximately 7,000 evidence tokens, and a document ceiling of 15 (therefore no additional per-document restriction).
- Preservation/source/iterative/windowed modes remain disabled. No schema migration or graph rebuild is planned.
- Logs contain only bounded counts, timings, fixed statuses/reasons, and request correlation; never source text, identifiers, credentials, or prompts.
- Prefix all shell commands with `rtk`. Worktree: `C:/Users/jackj/.codex/worktrees/evidence-preservation/AquiLLM`; branch `codex/graph-retrieval-latency`.
- User authorized merge/push and development deployment. Do not deploy to production or remove temporary SSH access.

## Task 1: Batch extended seed rows

**Files:** `aquillm/apps/knowledge_graph/retrieval/extended_seed_repository.py`, a focused helper if needed for SQL construction, `aquillm/apps/knowledge_graph/tests/test_extended_seed_lookup.py`, and a focused new batched-query test file.

**Interface:** Keep `ExtendedSeedRepository.load_seed_identities(authority, chunks, authorization, codec, max_rows)` unchanged. Load one bounded collection of `(chunk_id, entity_id, canonical_id)` rows for all requested chunks in a projection, including JSON observation matches. Preserve each chunk's distinct identities, the aggregate `max_rows` boundary and overflow sentinel, and revalidation before/after the read. Keep chunk-to-document/artifact mapping exact.

- [x] Add failing tests for two chunks sharing an entity, non-representative JSON observations, independent per-chunk associations, stale/revoked authorization, aggregate overflow, and fewer database round trips.
- [x] Implement a bounded batched query rather than one query for each chunk; use a deterministic ordered union if that preserves the existing join semantics most simply.
- [x] Compare literal expected identities and normalized seed masses with the prior path, and test real SQL execution on an isolated test database when available.
- [x] Run focused tests, self-review, commit, and obtain task review.

## Task 2: Batch graph manifest reads

**Files:** `aquillm/apps/knowledge_graph/projection/topology_adapter.py`, focused topology helper if needed, and `aquillm/apps/knowledge_graph/tests/test_projected_topology_adapter.py` plus focused batch tests.

**Interface:** Keep the four existing topology gateway query families and response schemas. Replace one manifest query per generation with one bounded query for all selected generation keys. Validate exact membership, duplicates, ready state, requested ordering, and deadline; retain every existing provenance check. Inspect remaining graph reads for an equivalent bounded batching opportunity; do not trade off coverage to skip reads.

- [x] Add failing behavior tests showing one manifest database round trip for multiple generations and identical ordered manifest output.
- [x] Test missing, duplicate, unexpected, non-ready, and over-cap rows and deadlines.
- [x] Implement batched Cypher with a bounded result count; verify against the actual Memgraph service with read-only requests before deployment.
- [x] Run topology and gateway contract suites, self-review, commit, and obtain task review.

## Task 3: Overlap retrieval and expose branch outcomes

**Files:** retrieval `scheduler.py` / focused scheduling helper; document `chunk_search.py`, `hybrid_graph_orchestration.py` / focused lifecycle helper; existing branch scheduler/lifecycle/orchestration tests and new overlap/observability tests.

**Interfaces:** Add a request-bound start/finish lifecycle that starts readiness and the direct branch before embedding/baseline retrieval; start extended work only once the baseline exists. Reuse the bounded worker pool, cancellation/cleanup, deadline and branch-failure contracts. Existing synchronous callers remain supported. Authorize/revalidate before fusion and final materialization as today.

- [x] Add event/barrier tests proving direct retrieval starts while baseline retrieval is blocked, extended sees the finished baseline, and results match the sequential scheduler when no deadline expires.
- [x] Test baseline failure, early branch completion, deadline expiry, revocation, worker saturation, and concurrent requests without leaked permits or cross-request state.
- [x] Emit direct/extended success/failure/reason/count/time plus fusion duplicate/new counts in a fixed aggregate event, including shared failures and no-result paths.
- [x] Assert log redaction and distinguish a duplicate-only successful branch from a timeout.
- [ ] Run focused and existing retrieval/PPR/authorization/selection tests, structure and import checks; obtain task and whole-branch review.

## Task 4: Integrate and deploy to development

Review, CI, and deployment outcomes are recorded in the implementation PR; the
remaining checklist is the release gate rather than a claim of deployment.

- [ ] Include new regression files in CI, document runtime behavior and rollback, open a PR against development, attach it to this task, and wait for CI.
- [ ] Merge the verified head and fast-forward development locally and on the authorized server.
- [ ] Rebuild/recreate the topology gateway only if baked graph code changed; recreate web/main worker with the established Compose configuration. Do not restart GPU models or rebuild graph data.
- [ ] Verify effective flags, service health, graph readiness, paired read-only result equivalence and stage timings, then a complete answer request. Report quality/latency limits honestly.
