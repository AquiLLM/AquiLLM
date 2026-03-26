# Roadmap Guide

Last updated: 2026-03-25

## Single source of truth

- Program status board: `docs/roadmap/roadmap-status.md`

## Status at a glance

### Done

- Codebase refactor (`apps/` + `lib/`) core migration
- Unified multi-format ingestion
- Full multimodal storage + ingestion
- Unified document figure extraction

### In progress / partial

- Feedback rating tracking + CSV export
- Structure and code-quality remediation
- Large-file remediation and tool extraction
- Bigger vLLM model configuration (drift from plan)

### Not started

- Kubernetes support services
- MCP + skills + agents architecture
- Agentic support services (vendor-agnostic)
- Observability stack
- Sandboxed math prototype
- GCP secret management rollout
- Jenkins pipeline implementation

### Superseded

- PDF-only figure extraction plan (replaced by unified figure extraction)
- Early RAG caching/token-efficiency plans replaced by refresh plans

## What to work next

1. Complete feedback CSV export end-to-end.
2. Execute remediation baseline commits (architecture + large-file splits).
3. Run caching optimization rollout from refreshed plan.
4. Run token-efficiency rollout after caching baseline is stable.

## Plan archive

- Detailed plans and execution notes: `docs/roadmap/plans/README.md`
- Active plans folder: `docs/roadmap/plans/active/`
- Pending plans folder: `docs/roadmap/plans/pending/`
- Completed plans folder: `docs/roadmap/plans/completed/`
- Superseded plans folder: `docs/roadmap/plans/superseded/`
