# Development Retrieval Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Checkboxes track execution.

**Goal:** Restore working retrieval on254 and deliver reproducible old/new development comparisons.
**Architecture:** Repair existing lifetime boundaries, replay current interfaces, and independently verified dense sidecars. Keep deployment and data mutations under root control.
**Tech Stack:** Django, PostgreSQL/pgvector, Celery/Redis, Python asyncio, vLLM, Docker Compose.
**Spec:** docs/superpowers/specs/2026-10-13-development-retrieval-recovery.md

## Global Constraints

- Base b76f2ef7afa257436e1310d55e38e585f878f4e4; isolated existing worktree and codex/development-retrieval-recovery.
- Every shell command starts rtk. No credentials, source bodies or full environment dumps in outputs.
- Do not widen the15000ms retrieval cap or reset a spent ledger. Preserve authorization and publication fences.
- No fabricated embedding provenance; no legacy-vector relabeling or blanket regeneration/deletion.
- No GitHub SSH keys; GitHub transport remains HTTPS.
- Agents edit disjoint assigned files only and do not commit, push, SSH, deploy or change shared services. Root owns integration and server operations.
- Regression tests fail before code changes; existing assertions remain unless an intentional corrected behavior is documented.
- Preserve independent acceptance gates; development experiments are explicitly labeled experiments.

### Task 1: Correct expired retrieval and user-tool handoff

**Files:** aquillm/lib/llm/turn_context.py; aquillm/apps/chat/services/rag_pipeline.py; focused existing/new chat and llm test modules.
**Interfaces:** consumes TurnBudget, bind_execution and existing limited-result helpers; produces unchanged public signatures with corrected terminal behavior.
- [x] Reproduce with real call_tool_async and fake-clock expired TurnBudget under bound execution: dispatch_count0 must produce assistant context_limited rather than uncertain side effects.
- [x] Test direct retrieval TimeoutError at exhausted budget finishes handled with limited answer, avoiding fallback tool selection.
- [x] Test user-facing uncertain vector_search passes through real provider completion without evidence synthesis; actually started uncertain side-effect tools remain nonreplayable.
- [x] Run tests RED, implement smallest coherent guards, run focused GREEN plus adjacent lifetime/execution/publication modules.
- [x] Record files, commands, results and limitations in task1 report.

### Task 2: Repair replay and make comparisons explicit

**Files:** aquillm/apps/chat/evals/retrieval_replay.py; scripts/replay_retrieval.py; tests/test_replay_retrieval.py and relevant root CLI tests; .github/workflows/test-backend-frontend.yml; docs/operations/retrieval-replay.md.
**Interfaces:** consumes run_direct_rag_turn and current retrieval_query_runtime; produces generation-blocked replay with explicit effective settings and comparable reports.
- [x] Reproduce current AttributeError for removed rag_query._llm_rewrite_query through the existing real run_turn test.
- [x] Guard current generation/query-proposal entry points, not a fabricated obsolete attribute. Baseline explicitly disables optional proposal flags and retains generic GenerationGuard.
- [x] Preserve legacy baseline reproducibility and add only explicit modes/options needed to exercise current preservation and graph treatments without falsely naming them legacy.
- [x] Record selected evidence/graph/text/capacity/proposal modes, observed budgets and configured settings; never expose endpoint credentials or source bodies.
- [x] Include root replay tests in selected release CI. Run RED then GREEN for actual helper/CLI behavior and adjacent tests.
- [x] Document baseline semantics, supported treatment matrix and disabled generation/cache behavior; report remaining limits.

### Task 3: Remove measured retrieval bottlenecks safely

**Files:** aquillm/apps/documents/services/lexical_collection.py, chunk_search_authorization.py and narrowly related source/scope/collection authorization helpers as evidence warrants; focused authorization and lexical regressions.
**Interfaces:** consumes existing selected-scope snapshots and querysets; preserves result authorization, ordering and deadlines.
- [x] Investigate existing query shape, per-candidate authorization and repeated selected-document work; root supplies live profile (lexical9603ms, selected metadata1749ms, authorized rows1632ms for31collections).
- [x] Identify concrete avoidable work and reproduce via query-count/semantic regression before edits. Do not optimize by removing validation boundaries, increasing budgets, lowering recall or broadening scope.
- [x] Implement bounded batch reads/index-compatible queries only where semantics remain unchanged.
- [x] Verify revocation, inactive users, ancestry and document lifecycle changes remain rejected; preserve stable candidate order and exact term behavior.
- [x] Report evidence and measured local improvement; root performs final live profiling.

### Task 4: Verify and build real dense coverage

**Owner:** root.
**Files:** docs/runbooks/development-embedding-policy.md; docs/runbooks/rag-source-index-generations.md; existing embedding contracts/provider/manifest/reindex interfaces; add narrowly scoped verified registration/deployment support and tests only where required.
- [x] Inspect live embedder image, model snapshot, tokenizer, precision/quantization, preprocessing, dimensions and actual transport. Record hashes privately.
- [x] Check that serving responses and provenance contracts can bind honestly to the verified artifact; repair any missing integration with RED/GREEN tests.
- [x] Inventory selected development documents/chunks and storage; prepare reversible bounded rebuild and rollback manifest.
- [x] Register verified immutable space, schedule bounded sidecars using real worker path, wait for terminal outcomes and inspect exact active coverage.
- [ ] Configure verified space consistently in web/workers, check actual dense candidate contribution and usable citations.
- [x] Preserve all exclusions and failed owners; do not infer coverage from queue success.

### Task 5: Review, merge and deploy the correction

**Owner:** root after Tasks1-4 preparation.
- [ ] Review each independent implementation, integrate and run focused combined regression plus complete required CI.
- [x] Open a PR with current change summary, validation and links to internal#4/#6 plus relevant internal graph notes; attach it.
- [ ] Stage exact candidate images, verify source/config identity, preserve rollback and take appropriate fresh backup before data/cutover changes.
- [ ] Merge into development under current user authorization, update254 via HTTPS fetch and fast-forward, deploy only intended services.
- [ ] Verify real chat path, truthful timeout behavior, actual dense retrieval, migrations/worker routing/schedulers/HTTPS/assets and stability. Never substitute health-only checks for retrieval acceptance.

### Task 6: Run matched old/new development comparisons

**Owner:** root.
- [ ] Freeze authorized corpus selection and the10known study questions plus chat431 question, input hashes and budgets.
- [ ] Obtain safely isolated old-code baseline from pre-PR244 development; do not run an old mutable binary against new representation history.
- [ ] Run fixed new-code legacy baseline and explicit independent graph/source-window/vocabulary comparisons supported by the repaired harness.
- [ ] Record gold passage/evidence support, candidate contributions, errors, latency and effective flags; retain failed runs.
- [ ] Present measured improvements/regressions and comparison artifacts; keep independent holdout and production acceptance visibly separate.
