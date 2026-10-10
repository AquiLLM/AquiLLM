# Development retrieval gap parallel implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development with independent reviews. The user explicitly requested parallel agents: independent tasks use separate native worktrees, with serial integration by the coordinator.

**Goal:** Address the remaining embedding, evidence, graph and validation gaps on development with verified changes.

**Architecture:** Three independent first-wave worktrees isolate embedding contracts, evidence evaluation/activation, and graph capacity/validation. Runtime experiments and deployment have one coordinator owner; subsequent quality experiments depend on the first-wave evidence.

**Tech Stack:** Python/Django, pytest, PostgreSQL/pgvector, Memgraph, Redis/Celery, vLLM, Docker Compose, TypeScript/React.

**Spec:** docs/superpowers/specs/2026-10-09-retrieval-gap-program-design.md

## Global Constraints

- Development host 149.165.150.254 only; do not contact or change production 149.165.169.204.
- Preserve graph budgets 3000/4500/4500/5000 ms, authorization, provenance, canonical encoding, resource caps, bounded workers and retry-zero behavior.
- Keep V2 transport and graph-off controls working. Models, precision, vectors and gated evidence modes change only after the matching controlled validation.
- Preserve frozen evaluation labels; model output is never independent human review. Keep private inputs/credentials outside tracked files.
- All shell commands begin with rtk. Each worker edits only its worktree and assigned subsystem; no worker deploys, pushes or spawns another agent.
- Preserve primary drafts. Coordinator owns shared runtime experiments and serial integration.

## Execution map

| Lane | Worktree/branch suffix | Code ownership | First acceptance gate |
|---|---|---|---|
| A | retrieval-embedding-contract | lib/embeddings, chunk_embeddings, embedding audit/tests | Measured contract and behavioral regressions; no unsupported claim about historical precision |
| B | retrieval-evidence-gates | chat/evals, document rerank capability/scoring, evidence tooling/tests | Accurate stage/support measurement and fail-closed activation prerequisites |
| C | retrieval-graph-capacity | KG resolution/extraction/topology diagnostics and their tests | Bounded reproduction, equivalent supported results, unchanged safety caps |
| Coordinator | retrieval-gaps-integration | Shared runtime config, integration/audit docs, merge/deploy | Reviewed combined commit, isolated backend checks, development replay |

### Task 1: Embedding contract and compatibility

**Files:** Inspect aquillm/lib/embeddings/{config,local,utils,cohere}.py, aquillm/aquillm/utils.py, aquillm/apps/documents/services/chunk_embeddings.py and existing embedding tests. Any shared Django model/migration or settings edit requires coordinator agreement before implementation.

**Interface:** Deliver a redacted report distinguishing declared runtime identity, observed vector behavior and unknown historical identity; provide a bounded read-only command for coordinator execution. Do not synthesize provenance for old vectors.

- [x] Reproduce input-type propagation, dimension fitting, zero/nonfinite vectors and provider-fallback behavior using existing call paths.
- [x] Identify concrete defects and write an exact scoped subplan with the production API and behavioral tests before changing it.
- [x] Implement demonstrated contract/audit fixes with red/green tests; keep compatibility-changing request formatting behind an explicit experiment control.
- [x] Run adjacent embedding/chunk tests and lint; commit only owned files and report concerns.

```powershell
rtk rg -n "input_type|fit_embedding|fallback|get_embedding" aquillm/lib/embeddings aquillm/apps/documents/services/chunk_embeddings.py aquillm/aquillm/utils.py
rtk python -m pytest aquillm/apps/documents/tests/test_chunk_embeddings.py -q
```

### Task 2: Evidence recall and activation prerequisites

**Files:** Existing aquillm/apps/chat/evals/retrieval_replay*.py and evidence_quality*.py, aquillm/apps/documents/services/chunk_rerank_pair_capability.py and chunk_rerank_window*.py, related tests, docs/runbooks/evidence-preservation.md. Do not edit embedding providers or graph resolution.

**Interface:** Use existing replay/activation schemas. Reports distinguish pool, rerank, packet, exact support, complete observation and human-review eligibility. Human labels and frozen cases remain unchanged.

- [x] Audit the runbook against actual runtime/CLI integration; reproduce any mechanical blocker with real existing paths.
- [x] Write the exact fix/test subplan; implement bounded evaluator or capability integration fixes with red/green tests.
- [x] Prepare a concrete isolated live-experiment recipe and reviewable evidence bundle, preserving budgets and one-change comparisons.
- [x] Run relevant quality/capability/operational tests; commit owned code and report what still requires human review or verified runtime identities.

```powershell
rtk rg -n "attestation|eligible|activation-v2|human|runtime_digest" docs/runbooks/evidence-preservation.md aquillm/apps/chat/evals aquillm/apps/documents/services/chunk_rerank_pair_capability.py
rtk python -m pytest aquillm/apps/chat/tests/test_evidence_quality_gates.py aquillm/apps/documents/tests/test_chunk_rerank_pair_capability.py -q
```

### Task 3: Graph capacity and topology validation

**Files:** aquillm/apps/knowledge_graph/resolution/coreference.py, coreference_types.py, coreference_validation.py, extraction/pipeline.py, related test_coreference*.py and topology diagnostics/tests. Coordinate any resolver-version or DTO changes before implementation.

**Interface:** Preserve complete supported partitions, cannot-link/provenance rules, canonical results and hard caps. A production-only unresolved incident remains explicitly unresolved unless its actual cause is reproduced.

- [x] Reproduce candidate-audit exhaustion using deterministic synthetic patterns and trace its actual algorithmic cause.
- [x] Determine whether equivalent bounded enumeration can remove redundant candidates; compare complete supported outputs and fail-closed limits before implementing.
- [x] Investigate topology-invalid classification/diagnostics and close any demonstrated development gap without weakening validation.
- [x] Implement bounded fixes with behavioral regressions, run coreference/capacity/topology tests and commit owned files.

```powershell
rtk python -m pytest -p no:django aquillm/apps/knowledge_graph/tests/test_coreference.py aquillm/apps/knowledge_graph/tests/test_coreference_capacity.py aquillm/apps/knowledge_graph/tests/test_coreference_sparse_partition.py -q
rtk rg -n "candidate audit cap|MAX_DOCUMENT_DECISIONS|topology_invalid" aquillm/apps/knowledge_graph
```

### Task 4: Coordinator development measurements

- [x] Capture only allowlisted model/runtime settings, service revisions, GPU capacity and queue/readiness state; never print secrets.
- [x] Execute lane A/B/C bounded diagnostic commands one at a time where shared model/DB load affects measurements.
- [x] Replay the ten exact private study questions in both authorized scopes using the maintained harness; retain raw results outside git. No verified target/span mapping was available, so these 20 turns establish operational behavior only.
- [x] Record confirmed fixes, failed hypotheses and remaining human/corpus-dependent gaps separately.

### Task 5: Follow-on retrieval/ingestion experiments

- [x] From Tasks 1–4, select the earliest demonstrated loss stage and write its exact testable subplan before implementation.
- [ ] Evaluate bounded rewrite/alias/search/reranker or structural parsing/reference handling against a dedicated development copy. Preserve original questions and identifiers; never tune against heldout answers.
- [x] Keep unsuccessful changes out of deployment; retain counterexamples and measured limits. Shared indexes and gated modes remain unchanged until their acceptance requirements pass.

### Task 6: Independent validation debt

- [x] Re-run the recorded frontend typecheck and migration/model consistency checks on the current baseline.
- [x] Assign a separate free agent to bounded confirmed fixes; preserve historical migration integrity and unrelated drafts.
- [x] Review and test those fixes independently before integration.

### Task 7: Review, integration and development rollout

- [x] Independent task reviews for each implemented lane; route findings back to the implementer.
- [x] Cherry-pick reviewed commits into the integration branch sequentially; resolve shared interfaces explicitly.
- [ ] Run appropriate combined regression tests in isolated backends and exact release images; repeat cold combined, parent and graph-off controls.
- [ ] Perform whole-branch review, then merge/push development and deploy only .254 with rollback material and health gates.
- [ ] Publish an honest rollout audit, preserve private evidence, clean owned test resources and archive worktrees after integration.

## Preflight decisions

The preceding gap inventory supplies scope, but not evidence that every proposed research technique helps. First-wave agents therefore reproduce before choosing fixes and write exact subsystem subplans before behavioral implementation. Expert labels, historical precision and production incident reproduction cannot be fabricated. The explicit parallel request supersedes the single-implementer default; separate worktrees and exclusive runtime ownership prevent shared-state races.


## Measured follow-on decision

Task 5A selected the demonstrated cross-provider embedding-space risk first; its exact subplan is 2026-10-09-embedding-policy.md. Development application services now explicitly require the local provider. No precision, request-format, model, stored-vector, or index conversion is included. The outage tradeoff is explicit vector unavailability, with existing scoped lexical/error behavior; transient ingestion retries can still backlog.

The rewrite/alias/stronger-reranker/structural-parser experiment remains deferred. The synthetic vector audit and bounded 20-document sample are not historical provenance or whole-corpus compatibility proof. Exact study-question replays lack verified gold target mapping and independent human labels. Evidence activation still requires the frozen four-arm evaluation, operational bundle, runtime attestation, and independent review. Host GPU driver/library mismatch additionally prevents a verified fresh GPU runtime experiment; serving model containers were left running.

The graph capacity fix reproduces an initialism-collision pattern and preserves meaningful partitions, audits and hard caps. Historical production topology_invalid and the specific failed production document remain unreproduced. Resolver identity is versioned for the changed audit; no bulk rebuild is scheduled.

Task 6 includes an evidence-driven extension restoring inherited document metadata and correcting transfer-migration state. Both new migrations emit no schema/DML SQL; fresh migration and reverse/reapply preserve all physical index/constraint identities. The global model consistency check now detects no changes.
