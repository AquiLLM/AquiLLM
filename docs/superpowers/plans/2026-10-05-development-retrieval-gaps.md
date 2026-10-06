# Development Retrieval Gaps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Correct verified development retrieval/maintenance defects and ship repeatable tests for the outstanding internal findings.
**Architecture:** Reuse production search, authorization, observation, and pruning interfaces. Keep evaluation overrides process-local; deploy only tested fixes and conservative operational settings.
**Tech Stack:** Python 3.12+, Django, PostgreSQL/pgvector, Celery, Memgraph, local embedding/reranker services, Docker Compose.
**Spec:** docs/superpowers/specs/2026-10-05-development-retrieval-gaps-design.md

## Global Constraints
- All shell commands start with rtk.
- Worktree: C:/Users/jackj/.codex/worktrees/chat-collections-skills/AquiLLM; branch codex/development-retrieval-gaps.
- Preserve unrelated drafts in C:/Users/jackj/Github/AquiLLM.
- Development only: 149.165.150.254, checkout /home/exouser/AquiLLM.
- Do not log prompts, source passages, credentials, or entity/document identifiers in ordinary telemetry.
- No study-specific query expansions, fabricated gold labels, or live authorization bypass.
- Existing experimental evidence modes stay off absent their activation evidence.
- Retain active graph artifacts and the existing 30-day/keep-2 policy.
- No production deployment, re-ingestion, model replacement, or external messages.

### Task 1: Truthful graph failures and substage timing
**Files:** retrieval/branch_contracts.py, scheduler_support.py, production_direct.py and production runtime helpers under aquillm/apps/knowledge_graph; related tests and docs/operations/graph-retrieval-latency.md.
**Interfaces:** Existing BranchEnvelopeV1/BranchSafeDiagnosticsV1 remain the retrieval contract; fixed-label telemetry may add stage and elapsed_ms. Add explicit direct_branch_timeout / extended_branch_timeout labels for the scheduler's whole-branch expiry while keeping extractor_timeout for actual extractor calls.
- [x] Write regression tests using the real scheduler with blocked work after extractor completion: whole-direct timeout must not report extractor_timeout; real extractor timeout remains distinguishable.
- [x] Run the new tests and record their expected failure.
- [x] Implement narrow deadline attribution and safe stage timing around ontology, extraction, entity resolution, and topology/materialization where available; no ranking or deadline-budget changes.
- [x] Test branch scheduling, overlap, cancellation lifecycle, graph diagnostics, and redaction.
- [x] Commit and obtain task review.

### Task 2: Reproducible development retrieval replay
**Files:** create scripts/replay_retrieval.py, focused helper module(s) under aquillm/apps/chat/evals, tests/test_replay_retrieval.py, operations documentation.
**Interfaces:** Run rag_pipeline.run_direct_rag_turn with a real authorized principal and selected collections; consume lib.evidence_observation / lib.replay_observation events. Replace synthesis with a packet recorder, deny generation, and never save a conversation.
- [x] Tests first: missing/invalid manifest rejected; generation blocked; failed run remains failure; gold identity mapping fails on ambiguity; summarization reports pool/rerank/packet membership independently.
- [x] Implement bounded explicit CLI parameters for questions, principal ID, collection IDs, local report output, repetitions, and named process-local experiments. Keep source text out of default output; use stable source/text fingerprints for mapping.
- [x] Capture revision, effective limits/modes, stage timings, branch status, unique contributions, supporting-passage ranks, and final packet text coverage without emitting source text.
- [x] Run baseline across ten unmodified internal questions for parent and parent-plus-figures; compare graph off/direct/extended where useful.
- [x] Commit and obtain task review.

### Task 3: Reproduce and correct measured retrieval gaps
**Files:** narrowly scoped production modules identified by replay; add regression tests beside the affected modules; docs/audits/2026-10-05-development-retrieval-gaps.md.
**Interfaces:** Reuse collect_hybrid_candidate_snapshot, materialize_and_rerank_candidates and the evidence packet observer. No answer labels enter production search.
- [x] Compare current candidate depth with 30 and 60, and current reranker text with full-chunk text, retaining all ten questions as diagnostics.
- [x] Inspect actual embedding and reranker input contracts against the deployed model, cache identity, and token limits; a query instruction is an experiment, not a presumed fix.
- [x] Attribute graph timeouts with Task 1 instrumentation; fix demonstrated wasted work or incorrect input handling using a failing reproduction before production edits.
- [x] Check source/windowed/iterative activation prerequisites and run applicable existing quality/operational suites; report unavailable attestation or human review accurately.
- [x] Add focused regression tests for each identified correction, compare baseline/result packets and latency, and reject changes that merely trade one study answer for another.
- [x] Commit verified fixes/results and obtain task review. Record remaining research work with evidence rather than claiming every question is solved.

### Task 4: Conservative retention scheduling and development rollout
**Files:** aquillm/aquillm/celery_schedules.py, settings.py, graph-maintenance tests, deployment/operations documentation.
**Interfaces:** Existing prune_graph_artifacts_task and prune_graph_artifacts(execute=False/True) retain current retention and bounded batch behavior.
- [x] Write tests for a default-off pruning schedule, an explicit opt-in daily interval, valid bounds, and unchanged existing maintenance entries.
- [x] Add scheduling configuration without changing retention thresholds or deleting active records.
- [x] Run graph retention/projection tests and a development dry run; execute only eligible bounded cleanup and verify active counts/readiness.
- [x] Run authorization, graph, replay, and evidence regression suites and independent whole-branch review.
- [ ] Commit, merge/push development while preserving drafts, deploy affected services using the existing Compose configuration; preserve rollback images.
- [ ] Verify running revision, health, retention scheduling, and repeated retrieval smoke; publish an audit with measurements, limitations, and exact rollback procedure.

## Task 3 measured correction: direct alias SQL (2026-10-05)
Execute this correction after Task 1 and before the repeated Task 2 live runs.
Evidence: baseline direct alias SQL outlived the4500ms branch deadline by tens of seconds, exhausting the process worker pool. A process-local2500ms DB timeout reproduces the alias stall while name lookup takes94ms. Applying join_collapse_limit=1 only to alias lookup preserves identical predicates and finishes in85-118ms (two Q4 runs); changing enable_nestloop slowed name queries and is rejected.

**Files:** retrieval/direct_seed_queries.py, direct_seed_repository.py, direct_seed_resolution.py, production_direct.py, a focused bounded SQL helper, direct-seed and PostgreSQL tests.
- [x] Add a PostgreSQL regression validating scoped alias rows, ambiguity/cap behavior, and identical results with the current predicates.
- [x] Add deadline tests: expired budget executes no lookup; SQL cancellation reports the fixed direct_branch_timeout; after cancellation the connection is usable and the worker lease becomes available; transaction-local settings restore on success and failure.
- [x] Apply join_collapse_limit=1 only to fully consumed ALIAS tier queries; explicitly restore successful nested-transaction settings. Preserve all permission, artifact, document, ontology, membership, and canonical-link predicates.
- [x] Propagate the existing absolute branch deadline into direct seed repository calls. Use remaining monotonic budget for PostgreSQL statement_timeout; check between tiers/spans, and stop immediately after expiry. Do not change pool size or branch budget.
- [x] Run current direct-seed repository/resolution tests plus isolated PostgreSQL tests and the repeated twenty-turn development replay before declaring the worker starvation resolved.
