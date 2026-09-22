# Production reliability backport implementation plan

> Execute with isolated parallel implementation tasks, regression-first testing, scoped review, and a final whole-branch review.

**Goal:** Bring the diagnosed tool-call crash, terminal UI state, retrieval latency, and deployment compatibility fixes onto main.

**Design:** Start from main and include the two Genesis commits already deployed on production. Backport relevant changes from development with their tests, excluding graph/schema infrastructure. Preserve existing main-only tool-result identity fixes. Add explicit terminal UI error state so an aborted tool sequence cannot appear active indefinitely.

**Constraints:** Python >=3.12; existing React/Vitest stack; all commands prefixed with rtk; no live production writes during implementation; no new database migrations or knowledge-graph dependencies; preserve environment overrides. Static deployment defaults must be compatible across base, development and production Compose files. Existing production .env values are not changed by editing defaults and require a documented rollout.

## Task 1: Tool-result validation and terminal UI state

- [x] Add regressions validating real successful/empty packed search results as ToolMessage with null diagnostics; demonstrate pre-fix failure.
- [x] Add `None` to ToolResultValue and `retrieval_diagnostics` to permitted keys (1c747e86).
- [x] Add a regression reproducing an assistant tool request followed by a WebSocket error: spinner stops, error remains visible, input usable, a later retry can start normally.
- [x] Implement explicit terminal-error behavior consistent with existing websocket state; do not mutate historical model content to hide an error.
- [x] Run tool-message, cached-result identity, hook/grouping and component tests; report changed files and red/green evidence.

## Task 2: Reranker and memory latency backports

- [x] Port focused tests/code from 3fae9d2d and omitted reranker portions of a4ee6ca1, then latest tokenizer protection from a222dec0/46abf677. Preserve non-graph result-cache behavior.
- [x] Verify concurrent scoring, capability caching, token-fit handling, fallback ordering, failed requests, and no cache poisoning.
- [x] Port omitted memory changes from 8c3d520c: collection chats skip episodic search and blocking embedding/ORM work uses the appropriate nonshared async adapter.
- [x] Run relevant rerank and memory/chat receive tests; report evidence and any deliberately excluded unrelated changes.

## Task 3: Deployment compatibility

- [x] Include 2153adb and 0f7f58f already deployed on production.
- [x] Add regression for pooling embedding startup preserving bitsandbytes; port narrow 3741bc40 startup fix.
- [x] Reconcile context defaults to 131072 from 085a8ea2 and correct service DNS where applicable, preserving explicit overrides.
- [x] Align relevant documented rerank/embedding limits with the working development profile only where cross-service GPU capacity is valid; document aggregate allocation and rollout overrides rather than blindly increasing allocations.
- [x] Run vLLM argument, Genesis image, readiness, and Compose contracts.

## Task 4: Bounded evidence handoff and integration

- [x] Backport 6689a20a's direct-RAG evidence budgeting, document balancing and request-only evidence handoff without graph dependencies; retain compatibility with main's recovery behavior.
- [x] Run evidence/retrieval/synthesis regressions and frontend build/typecheck, plus appropriate backend integration checks.
- [ ] Review task diffs and full branch; resolve actionable findings.
- [x] Add rollout instructions covering source revision, application reload, image rebuild/recreate where required, existing .env reconciliation, health checks and rollback.
- [ ] Commit and push the complete backport to existing PR #227 (`codex/backport-genesis-runtime`), then follow required CI/review checks to integration permitted by repository policy and user authorization.

## Scope decisions

Grounding/citation stylistic follow-ups ab0e4491/b694615e are not necessary for this incident and are excluded unless required by evidence-handoff compatibility. The shared executor/event-loop architecture is not silently refactored in this backport; diagnose/test separately if the selected latency backports reveal a necessary dependency. Production deployment is distinct from getting the fixes into main: do not restart the live service as a side effect of local testing.
