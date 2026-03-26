# AquiLLM Program Roadmap (Implemented vs Planned)

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this roadmap in execution mode. Steps use checkbox (`- [ ]`) syntax for tracking.

**Snapshot Date:** 2026-03-26
**Scope:** All roadmap/spec/plan artifacts currently in `docs/roadmap/*`, `docs/specs/*`, and `docs/roadmap/plans/*`  
**Method:** Status determined from repository evidence (current files, tests, env/contracts, and recent commits), not checkbox completion state in plan markdown.

---

## Status Legend

- **Implemented:** Core intent is in repo and test-backed.
- **Partial:** Some outcomes landed, but planned architecture/cleanup is incomplete.
- **Not started:** No meaningful implementation artifacts in repo yet.
- **Superseded:** Plan replaced by a newer plan/design; only carry forward unresolved items.

---

## Portfolio Status Matrix

| Workstream | Source Plan(s) | Current Status | Evidence in Repo | Main Gap |
|---|---|---|---|---|
| Feedback rating tracking + CSV export | `2026-03-21-feedback-rating-csv-export.md` | **Implemented** | `apps/chat/models/message.py`, `apps/chat/services/feedback.py`, `apps/platform_admin/services/feedback_export.py`, `apps/platform_admin/views/api.py`, `templates/aquillm/email_whitelist.html`, and platform-admin/chat tests | Optional enhancement: richer filtering/reporting UX |
| Architecture boundary + structural remediation | `2026-03-21-architecture-boundary-and-structural-remediation.md` | **Implemented (baseline landed)** | Commit-backed execution sequence (`f112631`, `f259b5f`, `03aa4a6`, `770ccb6`, `5792895`); split modules now active (`apps/ingestion/views/api/*`, `apps/documents/services/chunk_*`, decomposed chat consumer files) | Keep enforcing module boundaries as new features land |
| Architecture remediation commit execution track | `2026-03-21-architecture-remediation-commit-plan.md` | **Implemented** | Plan sequence has landed in repository history and follow-through notes under `docs/roadmap/plans/completed/*` | No blocking gap; continue as maintenance guardrail |
| Multimodal RAG caching + latency optimization | `2026-03-22-multimodal-rag-caching-latency-optimization.md` | **Implemented** | Cache/settings/helpers + retrieval/rerank/image caching landed (`4420fc8`, `37c4ffb`) with tests (`tests/integration/test_cache_settings_flags.py`, `apps/documents/tests/test_rag_cache.py`) | Continue workload-based TTL/cost tuning |
| RAG token efficiency enhancements (LM-Lingua2 + LMCache) | `2026-03-22-rag-token-efficiency-enhancements.md` | **Implemented (feature-flagged)** | Prompt budget + LM-Lingua2 + salience context packer + metrics landed (`f2c7b30`, `da94b9b`, `23e4a7d`, `8fc3ed2`, `a40ff6b`); LMCache plumbing covered by `tests/integration/test_vllm_lmcache_plumbing.py` | Optional KVTC-style codec overlay remains future work |
| Codebase refactor (`apps/` + `lib/`) | `2026-03-18-codebase-refactor.md`, `2026-03-18-codebase-refactor-handoff.md` | **Implemented** | `aquillm/apps/*`, `aquillm/lib/*`, wrapper compatibility modules, refactor commits (`297af88`) | Remaining cleanup/perf passes, but core migration is done |
| Unified multi-format ingestion | `2026-03-17-unified-multiformat-ingestion.md`, `docs/roadmap/plans/completed/2026-03-16-auto-ingest-implementation-plan.md` | **Implemented (with evolution)** | `aquillm/aquillm/ingestion/parsers.py`, `apps/ingestion/tests/test_unified_ingestion_*`, ingestion batch tests and APIs | Align legacy docs with post-refactor file paths and current endpoint names |
| Full multimodal storage + ingestion | `2026-03-17-full-multimodal-storage-and-ingestion.md` | **Implemented** | `vllm_ocr` + `vllm_transcribe` env/tests, media/transcribe modules, `test_multimodal_ingestion_media_storage.py`, compose launch checks | Add explicit operational SLOs and capacity guidance |
| Unified document figure extraction | `2026-03-18-unified-document-figure-extraction.md` (supersedes PDF-only plan) | **Implemented** | `apps/documents/models/document_types/figure.py`, `aquillm/ingestion/figure_extraction/*`, `apps/ingestion/tests/test_figure_extraction.py` | Improve performance tuning and observability around figure extraction cost |
| PDF-only figure extraction | `2026-03-18-pdf-figure-extraction.md` | **Superseded** | Functionality exists under unified figure extraction architecture | Archive as historical, keep only unresolved test/perf tasks |
| Structure + code-quality remediation | `2026-03-19-aquillm-structure-code-quality-remediation.md`, `2026-03-19-structure-and-code-quality-remediation-execution-notes.md` | **Implemented (baseline complete)** | Remediation sequence and under-300-line budget closure are commit-backed, including `5792895` and related split/refactor commits | Ongoing hygiene only (no open blocker from this track) |
| Large-file remediation and tool extraction | `2026-03-19-large-file-remediation-lib-tools-and-splits.md` | **Implemented (core targets complete)** | Large-file split/remediation commits landed across backend/frontend/provider modules; file-length allowlist cleared in `5792895` | Continue opportunistic refactors as new hotspots emerge |
| Bigger vLLM model configuration | `2026-03-16-bigger-vllm-models.md` | **Superseded** | Later vLLM tuning/config commits adjusted model args and runtime defaults beyond this original model-sizing plan | Keep for history only; do not use as current execution source |
| GCP Secret Manager migration | `2026-03-25-gcp-secret-management-implementation.md`, design spec | **Partial** | Hardcoded-secret cleanup landed (`a253cc1`); active implementation plan exists in `docs/roadmap/plans/active/2026-03-25-gcp-secret-management-implementation.md` | Complete staged Secret Manager rollout, inventory/runbook, and CI policy checks |
| Jenkins CI/CD pipeline | design spec `2026-03-25-jenkins-pipeline-design.md` | **Not started** | Draft design exists; no Jenkinsfile(s) or Jenkins execution parity evidence in repo | Create implementation plan and land Jenkins parity jobs and deploy gates |
| Kubernetes support services (phase-based scaling model) | `2026-03-19-kubernetes-support-services-deployment-scaling.md`, design spec | **Not started** | `deploy/k8s/` does not exist yet | Implement full kustomize tree and overlay contracts |
| K8s background pod services | `2026-03-18-k8s-services-background-pods.md` | **Not started** | No `deploy/k8s/services/*` manifests in repo | Fold into newer k8s scaling roadmap as MVP subset |
| MCP + skills + agents architecture | `2026-03-20-mcp-skills-agents-structure.md` | **Not started** | No `lib/mcp`, `lib/skills`, `lib/agents`, no runtime registry service | Build unified runtime/tool registration first |
| Agentic support services (vendor-agnostic controlled access) | `2026-03-20-agentic-support-services-vendor-agnostic.md` | **Not started** | No `lib/agent_services` package, no external compute service adapters | Implement provider-agnostic service layer and controlled provider onboarding |
| Observability stack | `docs/roadmap/plans/active/2026-03-16-observability-implementation-plan.md` | **Partial (app metrics landed, infra stack pending)** | App-level observability improvements exist (e.g., context-packing metrics/logging commits `a40ff6b`, `6e1e205`), but no Prometheus/Grafana/Pyroscope deployment wiring | Implement stack manifests/compose wiring and operational runbooks |
| Sandboxed math + extensibility prototype | `docs/roadmap/plans/pending/2025-03-16-sandboxed-math-integration.md` | **Not started** | No math sandbox runtime/tool integration artifacts | Decide whether to defer behind MCP/skills foundation |

---

## Master Roadmap (Sequenced)

## Phases 0-2B: Completed Foundations (2026-03-22 to 2026-03-24)

### Completed Workstreams
- [x] Phase 0 feedback analytics export shipped end-to-end (capture validation + timestamping + superuser CSV endpoint + admin UI download + gzip streaming support).
- [x] Phase 1 architecture/stability remediation baseline landed through the planned commit sequence and split-module boundaries.
- [x] Phase 2 multimodal RAG caching landed (cache flags/settings, query/doc/image/rerank caching, and regression coverage).
- [x] Phase 2B token-efficiency rollout landed (shared prompt budget, LM-Lingua2 fail-open integration, salience-aware packing, observability hooks).

### Residual Follow-ups (non-blocking)
- [ ] Optional KVTC-style codec overlay on top of LMCache.
- [ ] Additional tuning for cache TTLs, compression thresholds, and rollout defaults by workload profile.

---

## Phase 3: Kubernetes Deployment Foundation

### Objective
Ship the full k8s support baseline described in the March 19 spec.

### Deliverables
- [ ] Implement `deploy/k8s/base` with namespace, config contracts, web/worker/vllm workloads.
- [ ] Implement `deploy/k8s/overlays/portable` and `deploy/k8s/overlays/production`.
- [ ] Implement optional `portable-keda` overlay and scaling profiles.
- [ ] Add validation script and CI-friendly checks (`kubectl kustomize`, client dry-run).

### Exit Gate
- [ ] `kubectl kustomize` and `kubectl apply --dry-run=client` pass for base + overlays.
- [ ] Env contract mapping for service hostnames is validated in both overlays.

---

## Phase 4: Runtime Extensibility Foundation (MCP + Skills)

### Objective
Create one runtime registration path for all non-core tools.

### Deliverables
- [ ] Implement `apps/chat/services/runtime_context.py` + `tool_registry.py`.
- [ ] Refactor `ChatConsumer` to build tools via registry service.
- [ ] Add `lib/mcp/*` config/client/adapter and wire MCP tool discovery into runtime.
- [ ] Add `lib/skills/*` contract/loader/registry and wire skill tools + prompt extras.

### Exit Gate
- [ ] `ChatConsumer` no longer hardcodes full runtime tool list inline.
- [ ] MCP and skills can be toggled independently via env/config.

---

## Phase 4B: Agentic Support Services (Vendor-Agnostic)

### Objective
Add a governed external-access service layer so agents can use tools, platforms, and compute services through auditable, policy-controlled capabilities.

### Deliverables
- [ ] Implement `lib/agent_services/*` provider contracts, policy, registry, and tests.
- [ ] Implement provider adapter framework and runtime tool factories.
- [ ] Add external job lifecycle persistence and admin inspection/cancel controls.
- [ ] Integrate support-service tools into runtime tool registry behind feature flags.

### Exit Gate
- [ ] External support-service jobs are auditable, cancellable, and policy-bounded.
- [ ] Runtime degrades gracefully when provider credentials/services are unavailable.

---

## Phase 5: Agent Orchestration (feature-flagged)

### Objective
Introduce bounded agent loop behavior without regressing non-agent chat.

### Deliverables
- [ ] Implement `lib/agents/*` policy + orchestrator.
- [ ] Add `apps/chat/services/agent_runtime.py`.
- [ ] Gate by explicit safety/env settings (`AGENT_ENABLED`, max steps, max tool calls).

### Exit Gate
- [ ] Agent mode and non-agent mode both pass regression tests.
- [ ] Termination and safety policies are test-enforced.

---

## Phase 6: Observability and Runtime Operations

### Objective
Make new and existing runtime paths operable in production.

### Deliverables
- [ ] Implement observability plan artifacts (compose and docs for metrics/logging/profiling).
- [ ] Add structured metrics/logging for MCP/skills/agents and ingestion hotspots.
- [ ] Define and document operational runbooks and alert thresholds.

### Exit Gate
- [ ] Minimal observability stack deploys and exposes core health signals.
- [ ] Runtime failures are diagnosable without ad-hoc debugging.

---

## Phase 7: Optional Advanced Track (Math Sandbox)

### Objective
Reassess math sandbox plan after extensibility primitives are live.

### Deliverables
- [ ] Decision record: implement now, defer, or re-scope as an MCP/skill provider.
- [ ] If approved, implement as provider on the same runtime registry path.

### Exit Gate
- [ ] Decision documented with scope, risk, and ownership.

---

## Immediate Next 4 Execution Batches

### Batch A (Secrets hardening + ops safety)
- [ ] Execute `docs/roadmap/plans/active/2026-03-25-gcp-secret-management-implementation.md` in staged rollout order.
- [ ] Produce `secrets-inventory`, migration runbook, and CI policy checks for secret-source enforcement.

### Batch B (Observability stack enablement)
- [ ] Execute `docs/roadmap/plans/active/2026-03-16-observability-implementation-plan.md` for Prometheus/Grafana/Pyroscope wiring.
- [ ] Add production-ready alert/runbook docs and verify compose/k8s compatibility.

### Batch C (CI/CD execution parity)
- [ ] Convert `2026-03-25-jenkins-pipeline-design.md` into an implementation plan and land Jenkinsfile parity gates.
- [ ] Validate deploy gate behavior and rollback checks match current release expectations.

### Batch D (Deployment + orchestration foundation)
- [ ] Start Kubernetes baseline (`deploy/k8s/base` + overlays) from pending k8s plans.
- [ ] Sequence new pending orchestration/ingestion plans (`2026-03-25-langgraph-research-orchestration.md`, `2026-03-26-ingestion-work-queue-batching-implementation.md`) behind Batch A/B prerequisites.

---

## Governance Notes

- Keep this roadmap as the single prioritization view; individual plan docs remain implementation detail.
- When a phase completes, update this file first, then link to execution notes and verification outputs.
- Do not treat unchecked boxes in old plans as status truth; use repository evidence and tests.


