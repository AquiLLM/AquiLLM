# Structure And Code Quality Remediation — Execution Notes

## What changed

- **Chat WebSocket (`append`)**: Initialize `files` to an empty list before optional upload handling so appends without a `files` key do not hit an unbound local.
- **Ingest WebSockets**: After `close()` for unauthenticated clients, return immediately so `group_add` is never called.
- **Settings / URLs**: Register `debug_toolbar` only when `DEBUG` is true; mount debug toolbar URL patterns only in debug; Celery accepts JSON only with JSON serializers.
- **Logging / debug**: Replace OAuth adapter `print` calls with structured logging; remove `breakpoint()` from the debug models view; use `logger.debug` in the LLM tool wrapper instead of `print`.
- **Context processors**: Build `api_urls` / `page_urls` with `reverse()` and `%(kwarg)s` placeholders compatible with `react/src/utils/formatUrl.ts`.
- **Frontend**: Move `IngestRowsContainer` to `react/src/features/ingestion/components/IngestRowsContainer.tsx`; `main.tsx` and legacy `components/IngestRow.tsx` re-export from features.
- **Repository**: Ignore top-level `node_modules/` and `aquillm/tmp/`; add `scripts/check_hygiene.ps1` and `.github/workflows/hygiene-check.yml`.

## What was verified

- Integration tests: `test_context_processors_urls.py`, `test_no_runtime_debug_artifacts.py`, `test_settings_security_flags.py` (subprocess check for `DJANGO_DEBUG=0` without debug_toolbar).
- Ingest WebSocket auth tests (unauthenticated) with `base_send` mocked.
- Full pytest with DB-backed tests requires a running PostgreSQL instance matching `DATABASES` settings (e.g. Docker Compose stack).

## Residual / follow-ups

- `test_chat_consumer_append.py` and authenticated ingest tests need PostgreSQL; run inside the usual dev container or with local Postgres credentials.
- CI should set dummy `OPENAI_API_KEY`, `GEMINI_API_KEY`, and Google OAuth env vars for subprocess settings tests if not already present.
