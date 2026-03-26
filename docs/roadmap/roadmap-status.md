# AquiLLM Program Roadmap (Implemented vs Planned)

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this roadmap in execution mode. Steps use checkbox (`- [ ]`) syntax for tracking.

**Snapshot Date:** 2026-03-25  
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
| Feedback rating tracking + CSV export | `2026-03-21-feedback-rating-csv-export.md` | **Partial (capture exists, export missing)** | `apps.chat.models.Message` already has `rating` and `feedback_text`; websocket actions for rate/feedback exist in `apps/chat/consumers/chat.py` | Missing feedback timestamp, validated capture service, question-number export query, superuser CSV endpoint, and admin download UX |
| Architecture boundary + structural remediation | `2026-03-21-architecture-boundary-and-structural-remediation.md` | **Not started** | Plan and commit roadmap exist; target modules (`apps.documents.services.chunk_*`, ingestion API split, chat tool wiring extraction) are not yet the active execution baseline everywhere | Execute remediation commits to establish stable ownership boundaries before optimization-heavy work |
| Architecture remediation commit execution track | `2026-03-21-architecture-remediation-commit-plan.md` | **Not started** | Commit-by-commit execution plan is documented | Execute and verify commits in sequence (especially 3, 6, 11, 12) |
| Multimodal RAG caching + latency optimization | `2026-03-22-multimodal-rag-caching-latency-optimization.md` | **Not started** | Plan exists with chunked TDD tasks and Phase 2 extensions | Execute after remediation baseline to avoid rework and keep cache logic in final module boundaries |
| RAG token efficiency enhancements (LM-Lingua2 + LMCache) | `2026-03-22-rag-token-efficiency-enhancements.md` | **Not started** | Plan exists; no cross-provider LM-Lingua2 wiring, chat/embed/rerank LMCache deployment hooks, or KVTC-style codec overlay are landed yet | Execute after remediation baseline and core Phase 2 caching, with feature-flagged rollout |
| Codebase refactor (`apps/` + `lib/`) | `2026-03-18-codebase-refactor.md`, `2026-03-18-codebase-refactor-handoff.md` | **Implemented** | `aquillm/apps/*`, `aquillm/lib/*`, wrapper compatibility modules, refactor commits (`297af88`) | Remaining cleanup/perf passes, but core migration is done |
| Unified multi-format ingestion | `2026-03-17-unified-multiformat-ingestion.md`, `docs/roadmap/plans/completed/2026-03-16-auto-ingest-implementation-plan.md` | **Implemented (with evolution)** | `aquillm/aquillm/ingestion/parsers.py`, `apps/ingestion/tests/test_unified_ingestion_*`, ingestion batch tests and APIs | Align legacy docs with post-refactor file paths and current endpoint names |
| Full multimodal storage + ingestion | `2026-03-17-full-multimodal-storage-and-ingestion.md` | **Implemented** | `vllm_ocr` + `vllm_transcribe` env/tests, media/transcribe modules, `test_multimodal_ingestion_media_storage.py`, compose launch checks | Add explicit operational SLOs and capacity guidance |
| Unified document figure extraction | `2026-03-18-unified-document-figure-extraction.md` (supersedes PDF-only plan) | **Implemented** | `apps/documents/models/document_types/figure.py`, `aquillm/ingestion/figure_extraction/*`, `apps/ingestion/tests/test_figure_extraction.py` | Improve performance tuning and observability around figure extraction cost |
| PDF-only figure extraction | `2026-03-18-pdf-figure-extraction.md` | **Superseded** | Functionality exists under unified figure extraction architecture | Archive as historical, keep only unresolved test/perf tasks |
| Structure + code-quality remediation | `2026-03-19-aquillm-structure-code-quality-remediation.md`, `2026-03-19-structure-and-code-quality-remediation-execution-notes.md` | **Partial (high-priority fixes landed)** | Runtime/security fixes committed (`d9655ad`, `c049ad9`, `48696d5`, `772eb7f`, `c0b0fa9`, `44d1bcc`, `3c816aa`) | Remaining maintainability tasks and full verification closure |
| Large-file remediation and tool extraction | `2026-03-19-large-file-remediation-lib-tools-and-splits.md` | **Partial** | Some frontend extraction done; major hotspots still large (`apps/chat/consumers/chat.py`, `lib/llm/providers/openai.py`, `lib/llm/providers/base.py`, large React components) | Complete planned splits and chat tool-registry extraction |
| Bigger vLLM model configuration | `2026-03-16-bigger-vllm-models.md` | **Superseded** | Later vLLM tuning/config commits adjusted model args and runtime defaults beyond this original model-sizing plan | Keep for history only; do not use as current execution source |
| GCP Secret Manager migration | `2026-03-25-gcp-secret-management-implementation.md`, design spec | **Not started** | Design + implementation plan exist, but no completed runbook/inventory rollout evidence yet | Implement staged migration and complete ops artifacts (`secrets-inventory`, runbook, CI checks) |
| Jenkins CI/CD pipeline | design spec `2026-03-25-jenkins-pipeline-design.md` | **Not started** | Draft design exists; no Jenkinsfile(s) or Jenkins execution parity evidence in repo | Create implementation plan and land Jenkins parity jobs and deploy gates |
| Kubernetes support services (phase-based scaling model) | `2026-03-19-kubernetes-support-services-deployment-scaling.md`, design spec | **Not started** | `deploy/k8s/` does not exist yet | Implement full kustomize tree and overlay contracts |
| K8s background pod services | `2026-03-18-k8s-services-background-pods.md` | **Not started** | No `deploy/k8s/services/*` manifests in repo | Fold into newer k8s scaling roadmap as MVP subset |
| MCP + skills + agents architecture | `2026-03-20-mcp-skills-agents-structure.md` | **Not started** | No `lib/mcp`, `lib/skills`, `lib/agents`, no runtime registry service | Build unified runtime/tool registration first |
| Agentic support services (vendor-agnostic controlled access) | `2026-03-20-agentic-support-services-vendor-agnostic.md` | **Not started** | No `lib/agent_services` package, no external compute service adapters | Implement provider-agnostic service layer and controlled provider onboarding |
| Observability stack | `docs/roadmap/plans/pending/2026-03-16-observability-implementation-plan.md` | **Not started (or not present in this branch)** | No Prometheus/Grafana/Pyroscope manifests or compose wiring in current repo | Implement env + stack wiring and runtime metrics/log standards |
| Sandboxed math + extensibility prototype | `docs/roadmap/plans/pending/2025-03-16-sandboxed-math-integration.md` | **Not started** | No math sandbox runtime/tool integration artifacts | Decide whether to defer behind MCP/skills foundation |

---

## Master Roadmap (Sequenced)

## Phase 0: Feedback Analytics Export (highest priority now)

### Objective
Ship feedback ratings tracking and CSV export for operational reporting immediately.

### Deliverables
- [ ] Execute `2026-03-21-feedback-rating-csv-export.md` end-to-end.
- [ ] Add `feedback_submitted_at` on messages and validated feedback capture service.
- [ ] Add superuser-only CSV endpoint with required columns: `date,user_number,rating,question_number,comments`.
- [ ] Add admin-facing one-click CSV download control.
- [ ] Add execution notes and README operator usage docs.

### Exit Gate
- [ ] Superuser can download CSV with correct schema and escaping.
- [ ] Tests pass for capture validation, timestamping, permissions, and export query logic.

---

## Phase 1: Close Current Stability Debt and Remediation Baseline

### Objective
Finish architecture/stability remediation loop and establish final module boundaries.

### Deliverables
- [ ] Execute `2026-03-21-architecture-remediation-commit-plan.md` through at least commits 1-12.
- [ ] Prioritize commits 3, 6, 11, and 12 as baseline for downstream caching work.
- [ ] Complete unresolved tasks from `2026-03-19-aquillm-structure-code-quality-remediation.md`.
- [ ] Execute remaining critical slices from `2026-03-19-large-file-remediation-lib-tools-and-splits.md`.
- [ ] Add/update regression tests around newly split boundaries and import guardrails.

### Exit Gate
- [ ] No critical runtime/security findings open from remediation notes.
- [ ] Top backend/frontend hotspot files are reduced to manageable ownership units.
- [ ] Remediation baseline commits (3, 6, 11, 12) are merged and verified.

---

## Phase 2: Multimodal RAG Caching and Latency Optimization

### Objective
Reduce chat-time RAG latency with low-risk caching once remediation boundaries are stable.

### Deliverables
- [ ] Execute Chunks 1-3 from `2026-03-22-multimodal-rag-caching-latency-optimization.md`.
- [ ] Add shared Redis-backed Django cache wiring and feature flags.
- [ ] Implement query embedding cache, doc access cache, image payload cache, rerank capability/result caches.
- [ ] Add cache observability and rollout/rollback notes.
- [ ] Optionally execute Chunk 4 (Phase 2 extended optimizations) after core cache rollout is stable.

### Exit Gate
- [ ] Caching feature flags allow safe disable without code rollback.
- [ ] Targeted cache tests + regression suites pass.
- [ ] Latency instrumentation logs clearly show cache hit/miss behavior and no correctness regression.

---

## Phase 2B: RAG Token Efficiency Enhancements (LM-Lingua2 + LMCache)

### Objective
Improve prompt and runtime token efficiency across the full RAG pipeline after Phase 2 cache stabilization, without regressing answer quality or tool-call correctness.

### Deliverables
- [ ] Execute `2026-03-22-rag-token-efficiency-enhancements.md`.
- [ ] Add feature-flagged LM-Lingua2 prompt compression across RAG answer-provider paths (OpenAI-compatible, Claude, Gemini) with fail-open behavior.
- [ ] Add optional LMCache env + compose + `vllm_start.sh` wiring for chat/embed/rerank KV reuse/offloading.
- [ ] Implement KVTC-style transform-coding overlay on top of LMCache (codec + calibration + fail-open fallback).
- [ ] Add token-efficiency observability (compression ratio, skip/error reasons, safe rollout/rollback notes).
- [ ] Run targeted and regression tests proving correctness under enabled and disabled states.

### Exit Gate
- [ ] Compression can be disabled instantly via env flags with no code rollback.
- [ ] LMCache integration remains optional and does not block non-LMCache deployments.
- [ ] KVTC-on-LMCache codec path is quality-bounded, fail-open, and operationally reversible by flags.
- [ ] Token savings are measurable in logs/metrics and quality regressions are not observed in core chat flows.

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

### Batch A (Business-critical: feedback export)
- [ ] Execute `2026-03-21-feedback-rating-csv-export.md` completely.
- [ ] Verify CSV schema/permissions with platform-admin and chat tests.

### Batch B (Remediation baseline)
- [ ] Execute architecture remediation commit plan through baseline commits (3, 6, 11, 12) plus verification.
- [ ] Update execution notes with objective pass/fail evidence.

### Batch C (Latency optimization)
- [ ] Execute caching plan Chunks 1-3 from `2026-03-22-multimodal-rag-caching-latency-optimization.md`.
- [ ] Promote Chunk 4 tasks only after core cache rollout is stable.

### Batch D (RAG token efficiency enhancements)
- [ ] Execute `2026-03-22-rag-token-efficiency-enhancements.md` after Batch C exits.
- [ ] Roll out with feature flags and canary validation before broader enablement.

---

## Governance Notes

- Keep this roadmap as the single prioritization view; individual plan docs remain implementation detail.
- When a phase completes, update this file first, then link to execution notes and verification outputs.
- Do not treat unchecked boxes in old plans as status truth; use repository evidence and tests.


