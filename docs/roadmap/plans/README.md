# Plan Archive

Last audit: 2026-03-25

This folder is organized by execution status, with status decisions checked against commit history.

## Buckets

- `active/`: in-flight or follow-up work with unresolved scope.
- `pending/`: planned work with no clear implementation evidence yet.
- `completed/`: workstreams with implementation evidence in commit history.
- `superseded/`: replaced planning artifacts that should no longer drive execution.

## Commit-history audit notes

### Addressed (commit-backed)

- Architecture remediation sequence: `f112631`, `f259b5f`, `03aa4a6`, `770ccb6`, `5792895`.
- RAG cache + token/context efficiency rollout: `4420fc8`, `37c4ffb`, `f2c7b30`, `da94b9b`, `d1d5d5d`, `23e4a7d`, `8fc3ed2`, `a40ff6b`.
- Feedback CSV export flow (backend + UI + compression): `3a12c25`, `224af34`, `7c05ca7`, `4105331`, `f246b16`, `4aa9f69`, `ef9bca7`.
- Unified/document figure extraction follow-through: `d091587`, `ecc24fb`, `47de58f`, `2a1a9a3`.

### Still unaddressed or only partially addressed

- Observability stack (Prometheus/Grafana/Pyroscope): no implementation commits found after planning.
- Kubernetes deployment plans: no `k8s`/`kubernetes` implementation commits found.
- MCP/agentic support-services foundation: no implementation commits found.
- Sandboxed math plan: no implementation commits found.
- GCP Secret Manager migration: only partial hardcoded-secret cleanup (`a253cc1`), not full GSM rollout.

## Current working sets

- Active: `docs/roadmap/plans/active/`
- Pending: `docs/roadmap/plans/pending/`
- Completed: `docs/roadmap/plans/completed/`
- Superseded: `docs/roadmap/plans/superseded/` (historical plans, including legacy vLLM sizing plan)

## Priority active plans

- `active/2026-03-16-observability-implementation-plan.md`
- `active/2026-03-25-gcp-secret-management-implementation.md`
