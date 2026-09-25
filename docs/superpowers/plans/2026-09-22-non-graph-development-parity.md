# Non-graph development parity implementation plan

> Use subagent-driven development for independent ownership groups, followed by integration and an independent whole-branch review.

**Goal:** Port all implemented development changes independent of the knowledge graph onto main, preserving production-only reliability fixes, and prepare a tested PR and deployment configuration.

**Source:** development `c086ddc0`; **base:** deployed main `3c71dd1d`.

**Design:** Audit actual file differences rather than commit ancestry because prior selective backports use different commits. Copy independent components with their development tests; adapt mixed files so no graph runtime, schema editor, projection, extractor, graph-specific data migration, or dependency is introduced. Retain production's hardened tool-result identity, null handling, rerank parsing, and tested Genesis/GPU profile. Enable bounded automatic collection retrieval through configuration and exercise its complete application route.

**User authorization:** The user requested all missing non-knowledge-graph work, including tool calling, optimization, and deployment. The preceding comparison supplies the design context; continue implementation without another approval round. Merge remains a user action, as in the preceding PR workflow.

## Constraints

- Work only in this isolated checkout; leave the development and prior backport checkouts unchanged.
- Prefix shell commands with `rtk`.
- No live production mutations during implementation/testing; no migrations against production for tests.
- Preserve stronger main-only fixes; a source-file difference is not automatically a missing feature.
- Port implemented features and meaningful tests, not unimplemented design proposals as functioning code.
- No blanket dependency upgrades, model replacement, or GPU-allocation changes to the healthy production runtime.
- Maintain a path-level audit accounting for every source delta as ported, equivalent, mixed/adapted, or graph-only excluded.
- Use imported development regressions to establish missing behavior, then implement, test, and review.

## Task 1: Complete inventory and baseline

- [x] Record every changed path and classify ownership/dependency scope.
- [x] Establish backend/frontend and test infrastructure baselines.
- [x] Review mixed-file decisions and source dependency changes.

## Task 2: Chat, retrieval routing, and LLM parity

Owner: chat worker. Files: `aquillm/apps/chat/**`, `aquillm/lib/llm/**`, `aquillm/lib/conversations/**`, associated tests. Shared settings/celery wiring are owned by integration.

- [x] Port single-query limit, selective prompt-skill reads, concise grounding/numeric citation repair, and independent conversation-history search/indexing.
- [x] Preserve main tool-result UUID/null fixes and bounded evidence behavior.
- [x] Run regressions for query limits, direct synthesis, tool loops, and conversation search; report dependency wiring required.

## Task 3: Document and citation parity

Owner: document worker. Files: `aquillm/apps/documents/**`, `aquillm/aquillm/ingestion/**`, React chat citation components/utilities and tests. Shared dependencies/settings/routes are integration owned.

- [x] Port streaming/progressive/virtualized PDF citations and independent citation/figure improvements.
- [x] Extract independent document/ingestion behavior from graph-related edits; exclude graph-only fields/migrations/hooks.
- [x] Preserve hardened production reranker behavior; identify rather than overwrite equivalent implementations.
- [x] Run document/citation regressions and frontend tests; report integration hooks.

## Task 4: Deployment/runtime parity

Owner: deployment worker. Files: `deploy/**`, `tests/asr/**`, ASR verification scripts and their tests, relevant integration contracts. Shared `.env.example`, dependency manifests, and CI belong to integration.

- [x] Port graceful workers, independent runtime/startup fixes, supported transcription implementation and tests.
- [x] Preserve healthy Genesis version/pins and coordinated production GPU allocations.
- [x] Make optional service model changes explicit; document existing-env reconciliation.
- [x] Run startup/Compose/ASR contracts without launching GPU services.

## Task 5: Shared integration and remaining independent deltas

Owner: root. Files: application startup/settings/logging/tasks/routes, embeddings/memory, collections/integrations/core/ingestion APIs, dependencies/CI/docs, non-graph React collection changes.

- [x] Port background vector-index warmup, privacy-safe useful INFO logging, independent general/integration changes, and shared wiring requested by workers.
- [x] Enable automatic bounded collection retrieval and retain an explicit opt-out.
- [x] Reconcile dependency and build files without introducing KG libraries.
- [x] Import applicable documentation and user docs; classify historical planning artifacts separately from runtime gaps.

## Task 6: Completeness and release verification

- [x] Re-audit source differences after integration; no unclassified implemented non-graph delta remains.
- [x] Run backend suites with an isolated test database, frontend tests/build, migration checks, import boundaries, formatting/hygiene and Compose checks. Results and inherited limitations are in the parity audit.
- [x] Independent scope and code-quality reviews; fix actionable findings and recheck.
- [ ] Commit/push new branch, open and attach PR, wait for CI and report merge readiness.
- [x] Publish precise deployment instructions including configuration activation, migrations, rebuild scope, smoke verification and rollback; do not claim deployed before actual deployment.
