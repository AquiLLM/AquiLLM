# Architecture boundary remediation â€” execution notes

Date: 2026-03-23  
Plans: `2026-03-21-architecture-boundary-and-structural-remediation.md`, `2026-03-21-architecture-remediation-commit-plan.md`

## Completed in this pass

### Runtime entry points (commits 1â€“2)

- Ingestion WebSocket consumers and routing live under `aquillm/apps/ingestion/`; `aquillm/ingest/*` re-exports for compatibility.
- `aquillm/asgi.py` imports `apps.chat.routing` and `apps.ingestion.routing`.
- Root URLconf uses `apps.chat.urls` and `apps.chat.views.pages.new_ws_convo`; `chat/views.py` is a deprecation-oriented shim.

### Documents / compatibility layer (commits 3â€“4)

- Chunking Celery task implementation: `apps/documents/tasks/chunking.py` (registered via `apps/documents/tasks/__init__.py` for autodiscover).
- Helpers: `apps/documents/services/image_payloads.py`, `apps/documents/services/chunk_progress.py`, `apps/documents/services/document_meta.py`.
- `aquillm/models.py` re-exports `create_chunks`, `_doc_image_data_url`, and document metadata helpers without hosting the task body.
- `apps/documents/models/document.py` queues `create_chunks` from the documents app.
- `TextChunk` image URL resolution uses `doc_image_data_url` from `image_payloads` (not `aquillm.models`).
- `tests/integration/test_architecture_import_boundaries.py` asserts no `from aquillm.models import` in non-test code under `aquillm/apps/` and `aquillm/lib/`.
- Ingestion consumers import `Document` / metadata from `apps.documents` (not the compat barrel).

### CI and local structure enforcement (commit 16)

- `scripts/check_file_lengths.py`: default budget 300 lines; allowlist covers known large files until further splits land.
- `scripts/check_import_boundaries.py`: `lib/` must not import `apps.*`; `apps/` (excluding tests/migrations) must not import `aquillm.models` directly.
- `.github/workflows/hygiene-check.yml`: new `structure` job runs both scripts.
- `.github/workflows/test-backend-frontend.yml`: Postgres (pgvector) service, `manage.py check`, focused pytest integration tests, and `react` production build.

### Documentation (commit 17)

- `README.md`: module ownership and boundary policy.
- `docs/documents/architecture/aquillm-current-architecture-mermaid.md`: ASGI routing note.

## Verification (run locally)

Set `DJANGO_DEBUG=1`, dummy `OPENAI_API_KEY` / `GEMINI_API_KEY`, and working Postgres (or use compose) for full pytest.

```bash
python scripts/check_file_lengths.py
python scripts/check_import_boundaries.py
python -m pytest aquillm/tests/integration/test_architecture_import_boundaries.py aquillm/tests/integration/test_context_processors_urls.py -q
cd react && npm ci && npm run build
```

## Residual work (plan commits not fully executed here)

The following items from the remediation commit plan remain for follow-up PRs:

- **Commits 5â€“10**: Split `apps/chat/consumers/chat.py`, extract `lib/tools` search/document modules with `tool_wiring`, split `lib/llm/providers/base.py` and `openai.py`, decompose `apps/chat/tests/test_messages.py`.
- **Commit 11**: Split `apps/ingestion/views/api.py` into `views/api/*.py` plus services.
- **Commit 12**: Move embedding/search/rerank orchestration out of `apps/documents/models/chunks.py` into services.
- **Commit 13**: Split `aquillm/zotero_tasks.py`, remove threaded ORM writes, route through `apps/integrations/zotero/`.
- **Commits 14â€“15**: Move large React components under `react/src/features/*` with re-exports from `components/`.

## Risks

- Intermediate git commits were rewritten once so that `apps/ingestion/consumers.py` does not reference `document_meta` before that module exists (bisect-safe order).
- File-length allowlist will need trimming as splits land; otherwise new large files could slip through until added to the allowlist.

