# AquiLLM Code Style and Quality Guide

## Purpose

This guide captures the standards established during the 2026 code-quality remediation work. It is designed to keep structure, formatting, and reliability consistent as the codebase grows.

## Non-Negotiable Quality Gates

These are enforced by CI and should pass before every PR:

```bash
python scripts/check_file_lengths.py
python scripts/check_import_boundaries.py
python scripts/check_logging_conventions.py
pwsh -ExecutionPolicy Bypass -File scripts/check_hygiene.ps1
```

Expected outcomes:

- `aquillm/**/*.py`, `react/src/**/*.{ts,tsx}` stay at `<= 300` lines.
- `aquillm/lib/**` does not import `apps.*`.
- Runtime code in `aquillm/apps/**` does not import `aquillm.models` compatibility barrels.
- Every `logger.<level>(...)` call uses an `obs.<domain>.<event>` event id (see [Structured Logging](#structured-logging)).
- Tracked paths do not include generated artifacts like `node_modules/` or `aquillm/tmp/`.

## Architecture and Ownership Rules

### Backend

- `aquillm/apps/*` is the runtime domain layer (models, views, consumers, tasks, services by domain).
- `aquillm/lib/*` is shared provider/helper code and must remain runtime-framework-light.
- `aquillm/aquillm/models.py` is compatibility surface only. New runtime code should import from domain modules directly.

### Frontend

- Domain UI belongs in `react/src/features/<domain>/`.
- `react/src/components/` should mainly host legacy mount shims and compatibility exports.
- Avoid duplicate source-of-truth components. Keep one implementation path and re-export if needed.

## File Design Standards

- One file should have one clear responsibility.
- Views and consumers should be thin: auth, request parsing, response shaping, delegation.
- Business logic belongs in `services/` (backend) and hooks/util modules (frontend).
- If a file approaches the 300-line budget, split by cohesive behavior, not arbitrary chunks.
- Do not create `misc.py`, `helpers2.py`, or similar ambiguous buckets.

## Python Standards

- Use type hints on public functions and service boundaries.
- Prefer module docstrings where a module has non-trivial behavior.
- Use structured logging (see [Structured Logging](#structured-logging)); do not use `print()` or `breakpoint()` in runtime paths.
- Validate external input early and return precise, safe error messages.
- Keep import groups clean (stdlib, third-party, local) and avoid circular imports.
- Re-raise exceptions with context only when it improves diagnosis.

## Django Standards

- Enforce authorization checks before data mutation.
- Use domain services for orchestration instead of embedding heavy logic in views/models.
- Use `reverse()` and named URL routes for URL map generation to avoid resolver drift.
- Keep Celery payloads JSON-safe and deterministic.
- Do not perform threaded ORM writes.

## Structured Logging

Backend logs are structured: they are routed through `structlog` (configured in `aquillm/aquillm/settings_logging.py`) and rendered as JSON in production and Loki. To stay queryable, every log line carries a stable, machine-grep-able **event id** plus its variable data as discrete fields rather than baked into a sentence.

### The convention

- Every `logger.<level>(...)` call's **first positional argument is a string literal** of the form `obs.<domain>.<event>` — lowercase `snake_case`, exactly two dotted segments after `obs.` (e.g. `obs.zotero.api_error`). This is the stable event id; it must never contain interpolated values.
- Put all dynamic context in **keyword arguments** (`structlog` renders them as fields). Do not build messages with f-strings, `%`, or string concatenation.
- This is enforced by `scripts/check_logging_conventions.py`, which fails CI on any `logger.<level>()` call whose first argument is not a matching literal.

```python
# Don't — interpolated, unqueryable
logger.info(f"[Task {task_id}] Starting crawl for URL: {url}, depth: {d}")
logger.error(f"Zotero API request failed: {url} - {e}")

# Do — stable event id + structured fields
logger.info("obs.crawl.start", task_id=task_id, url=url, max_depth=d)
logger.error("obs.zotero.api_error", url=url, error=str(e), error_type=type(e).__name__)
```

### Naming

- **Domain** = the subsystem. Use an existing domain where one fits: `crawl`, `zotero`, `ingest`, `figures`, `memory`, `chat`, `llm`, `rag`, `embed`, `documents`, `ocr`, `collections`, `auth`, `observability`, `core`, `bug_report`. Add a new domain only for a genuinely new subsystem.
- **Event** = a short `snake_case` description of what happened (`start`, `fetch_error`, `cache_hit`, `chunk_indexed`, `sync_complete`). Reuse the same event id across call sites that represent the same event; distinguish variants with a field, not a new event.

### Field conventions

- Exceptions: `error=str(e), error_type=type(e).__name__`. Preserve any existing `exc_info=...` flag; keep `logger.exception(...)` as `logger.exception("obs.<domain>.<event>", ...)`.
- Reuse consistent keys: `error`/`error_type`, `*_id` (`user_id`, `task_id`, `collection_id`), `*_count`/`count`, `url`, `path`/`filename`, `duration_ms`.
- Never log secrets, tokens, or raw credential material (see [Logging, Security, and Operational Hygiene](#logging-security-and-operational-hygiene)).

### stdlib loggers

A few modules use a stdlib logger (`logging.getLogger(__name__)`) rather than `structlog`. Bare keyword arguments raise `TypeError` there, so pass context via `extra={...}` instead — the first-argument event id rule is unchanged:

```python
logger.info("obs.memory.graph_relation_dropped", extra={"relation": item})
```

Do not use a `# ignore` comment to bypass the linter; make the call conform.

## TypeScript and React Standards

- Keep `tsconfig` strict guarantees intact (`strict`, `noUnusedLocals`, `noUnusedParameters`).
- Use typed props/interfaces for components and hooks.
- Keep components focused on rendering; move protocol/state orchestration into hooks.
- Keep mount registries typed and explicit.
- Prefer feature-local utils/types over cross-tree ad hoc imports.

## Testing Standards

- Every bug fix should include a regression test.
- Add unit tests for pure helpers and service logic.
- Add integration tests for routing, import boundaries, settings gates, and auth behavior.
- Keep tests deterministic and scoped to the behavior being changed.

Recommended local verification:

```bash
# backend
cd aquillm
python -m pytest -q --tb=short

# frontend
cd ../react
npm run typecheck
npm run build
```

## Logging, Security, and Operational Hygiene

- Never log secrets, tokens, or raw credential material.
- Debug-only tooling should be gated behind `DEBUG`.
- Keep serializer formats explicit and safe.
- Remove dead code and debug artifacts in the same PR where found.
- Keep `.env` defaults and deployment scripts aligned with safe behavior.

## PR Checklist

Before opening or merging a PR, verify:

1. Structure rules are respected (`apps` vs `lib`, feature ownership, no compatibility backsliding).
2. File-length and import-boundary scripts pass.
3. Hygiene script passes.
4. Relevant pytest targets pass.
5. Frontend typecheck/build passes when touching React/TS.
6. New behavior has tests.
7. Logs/errors are useful, safe, and follow the `obs.<domain>.<event>` structured-logging convention.

## Commit and Review Practices

- Keep commits focused by responsibility (`fix`, `refactor`, `test`, `docs`, `chore`).
- Preserve stable public interfaces when splitting files; use thin compatibility re-exports.
- In PR descriptions, include:
  - what changed
  - why it changed
  - what was verified
  - any residual risks or follow-ups

## Definition of Done

A change is done when it is readable, bounded, tested, and passes all structural guardrails without introducing compatibility regressions.
