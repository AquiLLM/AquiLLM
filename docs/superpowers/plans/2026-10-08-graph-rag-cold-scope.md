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
- Do not change Cypher, cache behavior, models, evidence/ranking, or the V1 wire contract. Task 3 adds an explicitly enabled V2 snapshot exchange.
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

### Task 3: Bounded single-exchange snapshot transport

**Context:** Execute before completing Task 2 rollout. The initial reviewed implementation passed5/6 clean-image cold retrievals; the failure exhausted its remaining budget during the final V1 family exchange. The follow-up design in the spec is authoritative.

**Files and interfaces:**
- Add closed V2 contract/client/service modules under aquillm/apps/knowledge_graph/retrieval/topology, following existing naming/style. Keep modules focused; factor transport helpers where they remove duplication without changing V1 behavior.
- Extend TopologyGatewayClient with an exact bool snapshot opt-in default False and optional execute_snapshot capability; retain existing constructor compatibility and execute_read.
- Extend projection/topology_adapter.py with a snapshot operation that calls existing _decode exactly once, _manifests before _snapshot, and produces fresh manifests plus the existing full canonical snapshot.
- Extend retrieval/topology/memgraph.py loader to select snapshot transport only when explicitly supported/enabled, and otherwise retain V1. Share final scope/caps validation rather than duplicating it.
- Integrate strict KG_TOPOLOGY_GATEWAY_SNAPSHOT_ENABLED default disabled through lib/knowledge_graph/topology_gateway_config.py, Django settings mapping, apps/documents/services/hybrid_graph_dependencies.py, and .env.example. No production .env changes; the coordinator enables the setting only for development. Compose already passes the web .env; change manifests only if an actual propagation requirement exists.
- Add focused contract/ASGI/client/loader/adapter/config integration tests and run existing adjacent suites.

**Global constraints:** Preserve all original plan constraints except the deliberate addition of a separate V2 route. V1 schema/checksum/enum, per-family/source caps, request4MiB/response1MiB hard ceilings, configured ceilings, decoded snapshot2,000,000-byte cap, authorization/readiness, retry0, immutable exact cache key, branch/overall/extractor budgets, worker capacity ownership, canonical output, graph-off, redaction, models and ranking are unchanged. No raw query strings/general batch API. No subagents, push, deploy or production access from implementer.

- [ ] Read existing V1 transport/codec/worker code and the spec follow-up; write failing tests for optional single-exchange behavior and malformed V2 contracts before implementation. At minimum a fake driver test must fail because one snapshot operation is not yet supported; exact V1 output is the golden reference.
- [ ] Define POST /v2/topology/snapshot and a separate pinned descriptor/checksum. Request fields are parameters and deadline; family limits derive from existing decoded ready/caps. Responses are exact complete snapshot plus manifests, exact successful use_family_transport plus manifests, or the same fixed failure classes under the V2 schema headers. Reject unknown/missing/duplicate fields and noncanonical encoding.
- [ ] Add a snapshot adapter operation. Fresh _manifests always precedes _snapshot, including cache hits. Preserve generation sentinel and existing source-family maxima/provenance/error mappings. Serialize the complete immutable snapshot once and enforce its independent byte bound before success.
- [ ] Execute decode/manifest/hydrate/serialization under one existing admitted worker slot and absolute deadline; reject late success after final encoding. Keep auth/header/body validation before runtime access. Retain configured/hard response bounds.
- [ ] Preserve result-domain compatibility: if only the complete encoded envelope exceeds the unchanged wire ceiling, return use_family_transport with fresh manifests. Do not truncate, cache manifests, extend a deadline, or select this path for malformed/source-cap/auth/deadline errors. If even the directive exceeds the wire cap, return the existing result-cap failure.
- [ ] The client/loader validates V2 header/framing/closed response and fresh manifests, then either decodes the full canonical snapshot or executes precisely the original three family calls with the same parameters and absolute deadline. No second manifest call. Both paths share existing final authorized scope/node/edge/depth checks and error/readiness diagnostics. Reject response completion after deadline, including CPU decode time. A network/protocol failure must not cause another request.
- [ ] Add strict default-disabled config and factory wiring. Disabled clients use existing V1; explicit enabled clients require V2. Preserve graph-off no-network behavior. Show exact deployment configuration in docs/spec.
- [ ] Tests: one HTTP call/one parameter decode/fresh manifest on every request; golden V1/V2 canonical equality including evidence/audits/provenance; warm stale manifest rejection; maximum supported scope; auth-before-runtime; wrong schema/route/framing/duplicates/fields; below/at/above combined wire threshold, configured tighter bound, escaping/UTF8 growth, final2MB bound, and V1-valid split result above1MiB using exactly three family reads; source cap and sibling isolation; deadlines before/during/after I/O and serialization/decode; admitted worker retains slot until actual completion; strict enable/disable parsing/factory and graph-off. Use deterministic fake clocks for deadlines and behavioral assertions rather than timing thresholds.
- [ ] Run focused red/green tests, adjacent V1/client/service/contract/deadline/config suites, lint and diff checks. Coordinator runs clean-image, actual Memgraph/snapshot comparison and cold replay gates.
- [ ] Self-review, commit, and write task-3-report.md with exact commands/results and concerns. Return a concise report; no push/deploy.
