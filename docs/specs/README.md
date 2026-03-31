# Specs Index

Last updated: 2026-03-30 (Mem0 async pipeline)

## Status legend

- `Implemented`: core spec intent is in repository.
- `Partial`: some implementation exists, but drift or gaps remain.
- `Planned`: approved or drafted, implementation not started.
- `Superseded`: historical spec/plan replaced by newer operational direction.
- `Canceled`: explicitly retired and should not be executed.

## Spec folders

- Active/in-use specs remain in `docs/specs/`.
- Superseded specs live in `docs/specs/superseded/`.
- Canceled specs live in `docs/specs/canceled/`.

## Specification status

| Spec | Status | Execution Artifact |
|---|---|---|
| `2026-03-16-auto-ingest-design.md` | Implemented | `docs/roadmap/plans/completed/2026-03-16-auto-ingest-implementation-plan.md` |
| `superseded/2026-03-16-bigger-vllm-models-design.md` | Superseded | `docs/roadmap/plans/superseded/2026-03-16-bigger-vllm-models.md` |
| `2026-03-16-observability-design.md` | Planned | `docs/roadmap/plans/pending/2026-03-16-observability-implementation-plan.md` |
| `2026-03-17-unified-multiformat-ingestion-design.md` | Implemented | `docs/roadmap/plans/completed/2026-03-17-unified-multiformat-ingestion.md` |
| `2026-03-18-codebase-refactor-design.md` | Implemented (core) | `docs/roadmap/plans/completed/2026-03-18-codebase-refactor.md` |
| `2026-03-18-unified-document-figure-extraction-design.md` | Implemented | `docs/roadmap/plans/completed/2026-03-18-unified-document-figure-extraction.md` |
| `2026-03-19-kubernetes-support-services-deployment-scaling-design.md` | Planned | `docs/roadmap/plans/pending/2026-03-19-kubernetes-support-services-deployment-scaling.md` |
| `2026-03-25-gcp-secret-management-design.md` | Planned | `docs/roadmap/plans/pending/2026-03-25-gcp-secret-management-implementation.md` |
| `2026-03-25-jenkins-pipeline-design.md` | Planned | (implementation plan not yet created) |
| `2026-03-26-multi-backend-inference-deploy-strategy-design.md` | Planned | (implementation plan not yet created) |
| `2026-03-26-ingestion-work-queue-batching-design.md` | Planned | `docs/roadmap/plans/pending/2026-03-26-ingestion-work-queue-batching-implementation.md` |
| `2026-03-26-langgraph-mcp-tools-orchestration-design.md` | Planned | `docs/roadmap/plans/pending/2026-03-25-langgraph-research-orchestration.md` |
| `2026-03-30-adaptive-tool-call-budget-design.md` | Planned | `docs/roadmap/plans/pending/2026-03-30-adaptive-tool-call-budget-implementation.md` |
| `2026-03-31-rag-citation-enforcement-design.md` | Implemented | `docs/roadmap/plans/completed/2026-03-31-rag-citation-enforcement-implementation.md` |
| `2026-03-29-semantic-versioning-design.md` | Planned | `docs/roadmap/plans/pending/2026-03-29-semantic-versioning-implementation.md` |
| `2026-03-30-whisperx-transcription-design.md` | Planned | `docs/roadmap/plans/pending/2026-03-30-whisperx-transcription-implementation.md` |
| `2026-03-30-self-hosted-mem0-graph-memory-design.md` | Planned | `docs/roadmap/plans/pending/2026-03-30-self-hosted-mem0-graph-memory-implementation.md` |
| `2026-03-30-mem0-async-memory-pipeline-design.md` | Implemented | (see spec; async Mem0 in WebSocket chat path) |

## Notes

- Canonical progress tracking is maintained in `docs/roadmap/roadmap-status.md`.
- If spec intent changes after implementation, update this index and link the updated execution notes.
- `docs/specs/canceled/` currently has no files.
