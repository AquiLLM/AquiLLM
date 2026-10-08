# Cold graph scope latency implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Remove measured serialization and validation costs so cold combined-scope graph retrieval meets existing deadlines on development.

**Architecture:** Keep topology queries and wire contracts unchanged. Use equivalent compiled validators and Neo4j's matching native PackStream extension in the KG runtime.

**Tech Stack:** Python 3.12+, Neo4j Python driver 5.28.4, Memgraph, pytest, uv, Docker Compose.

**Spec:** docs/superpowers/specs/2026-10-08-graph-rag-cold-scope.md

## Global Constraints
- Development deployment only; do not contact or change production.
- Direct/extended budgets stay 4500 ms, overall 5000 ms, extractor 3000 ms.
- Preserve caps, authorization/readiness/provenance, canonical encoding, exact key/text accepted languages, exception precedence, protocol schema and checksum, retry0, graph-off behavior and bounded workers.
- Do not change Cypher, cache behavior, models, evidence/ranking, or add a new wire endpoint.
- No private questions, collection/user IDs, credentials, raw corpus data or private replay inputs in tracked files.
- All shell commands begin with rtk. Work only in the isolated worktree; preserve primary drafts.
- Tests verify behavior and real boundaries; timing comparisons belong in a manual benchmark, not a flaky CI threshold.

### Task 1: Equivalent validation and accelerated KG decoding

**Files:**
- Modify aquillm/apps/knowledge_graph/projection/serialization.py (_key)
- Modify aquillm/apps/knowledge_graph/retrieval/topology/gateway_contracts.py (_safe_text)
- Modify pyproject.toml, uv.lock, deploy/docker/knowledge-graph/Dockerfile
- Add focused tests under aquillm/apps/knowledge_graph/tests for validator equivalence and packaging/runtime requirements as appropriate
- Add a small repeatable, no-network diagnostic benchmark under scripts only if it improves maintainability; otherwise keep benchmark in the ignored task workspace

**Interfaces:** Existing signatures _key(value, name) -> None and _safe_text(value, name, limit, *, size_error=ValueError) -> None remain unchanged. Match neo4j==5.28.4 with neo4j-rust-ext==5.28.4.0 in knowledge-graph-local only. Existing runtime driver factories and query APIs remain unchanged.

- [ ] Read the spec and the two validators; record current contract/schema checksum and baseline manual performance.
- [ ] Add behavior tests comparing known-valid and invalid text/key inputs, exact errors and size precedence. Include all surrogate code points and C0/DEL, near-boundary Unicode and str subclasses. Exercise canonical DTO round trips and checksum preservation. Run the tests before changes; correctness must already hold, while the recorded benchmark demonstrates the performance defect (do not manufacture a failing semantic test for an equivalent optimization).
- [ ] Replace the loops with precompiled expressions. For keys use fullmatch of [0-9a-f]{64} after the original exact type guard; for forbidden text use search of [\\x00-\\x1f\\x7f\\ud800-\\udfff] after the original type and size guards. Retain original errors and signatures. Do not add caches or normalize strings.
- [ ] Add the exact optional dependency and narrowly regenerate the lockfile with uv. Verify unrelated package versions do not drift.
- [ ] Add a KG Docker build assertion immediately after uv sync verifying the matching neo4j and neo4j-rust-ext versions and that neo4j._codec.packstream.v1._rust_pack and _rust_unpack are non-None. Use a clear failure message. No assertion or dependency is required for web's non-KG environment.
- [ ] Run focused tests, relevant projection/gateway/contract suites, lint and the manual benchmark. The coordinator handles clean-image tests, disposable backend checks, live equivalence and cold replay. Record exact commands and results, distinguishing local from remote checks.
- [ ] Self-review and commit the narrow implementation. Write report to the task workspace; do not push or deploy.

### Task 2: Review, development verification and rollout (coordinator)
- [ ] Generate review package for Task 1 and obtain independent specification/code review; route findings to the implementer.
- [ ] Build clean source-labelled web and KG images. Verify the native codec in the KG image, run appropriate suites using isolated dependencies and real disposable Memgraph tests, and retain evidence.
- [ ] Compare canonical snapshots for private authorized development cases. Run the complete retrieval pipeline using fresh service caches and unchanged deadlines. Require all sampled direct and extended branches to succeed; if failures remain, investigate and revise the design before declaring fixed.
- [ ] Merge/push development within prior user authorization, preserve primary drafts, deploy reviewed images to149.165.150.254 with rollback retained and extractor-ready gate before web.
- [ ] Repeat cold combined-scope, parent-only, graph-off and restoration checks; verify revision labels, readiness/HTTPS and maintenance queues. Write a public-safe audit with scope, timings, counts and limitations.
- [ ] Obtain whole-branch review and finish with development evidence and explicit production status.

