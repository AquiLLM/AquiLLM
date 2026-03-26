# Architecture Boundary And Structural Remediation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close remaining structural debt by finishing large-file remediation, removing legacy runtime coupling, and adding enforceable architecture guardrails.

**Architecture:** Treat `apps/*` as domain/runtime ownership and keep `aquillm/*` compatibility modules thin. Move orchestration and business logic out of views/models into service/task modules. Complete unresolved large-file splits (`chat.py`, provider files, large React components), then enforce boundaries via CI checks so drift cannot reoccur.

**Tech Stack:** Django 5.1, Channels, Celery, pytest, React 19 + TypeScript, GitHub Actions.

**Depends on:** `docs/roadmap/plans/active/2026-03-19-large-file-remediation-lib-tools-and-splits.md`, `docs/roadmap/roadmap-status.md`

---

## Chunk 1: Runtime Entry-Point Consolidation (Legacy to Apps)

### Task 1.1: Add first-class ingestion websocket modules under `apps/`

**Files:**
- Create: `aquillm/apps/ingestion/consumers.py`
- Create: `aquillm/apps/ingestion/routing.py`
- Modify: `aquillm/ingest/consumers.py` (back-compat re-export only)
- Modify: `aquillm/ingest/routing.py` (back-compat re-export only)

- [ ] **Step 1: Move ingestion websocket consumers to `apps.ingestion`**

```python
# aquillm/apps/ingestion/consumers.py
from ingest.consumers import IngestMonitorConsumer, IngestionDashboardConsumer
__all__ = ["IngestMonitorConsumer", "IngestionDashboardConsumer"]
```

- [ ] **Step 2: Add `apps.ingestion.routing` and point patterns there**

```python
# aquillm/apps/ingestion/routing.py
from django.urls import re_path
from .consumers import IngestMonitorConsumer, IngestionDashboardConsumer
websocket_urlpatterns = [...]
```

- [ ] **Step 3: Run targeted tests**

Run: `cd aquillm && pytest apps/ingestion/tests/test_ingest_consumers_auth.py -q`  
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add aquillm/apps/ingestion/consumers.py aquillm/apps/ingestion/routing.py aquillm/ingest/consumers.py aquillm/ingest/routing.py
git commit -m "refactor(ingestion): make apps.ingestion the primary websocket runtime path"
```

### Task 1.2: Switch ASGI and URL runtime to `apps/*` entry points

**Files:**
- Modify: `aquillm/aquillm/asgi.py`
- Modify: `aquillm/aquillm/urls.py`
- Modify: `aquillm/chat/views.py` (compat-only module with explicit deprecation note)
- Test: `aquillm/tests/integration/test_context_processors_urls.py`

- [ ] **Step 1: Replace legacy imports**

```python
# asgi.py
from apps.chat.routing import websocket_urlpatterns as chat_patterns
from apps.ingestion.routing import websocket_urlpatterns as ingest_patterns
```

- [ ] **Step 2: Keep legacy URL names but resolve from `apps.chat.views.pages`**
- [ ] **Step 3: Run URL/context regression tests**

Run: `cd aquillm && pytest tests/integration/test_context_processors_urls.py -q`  
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add aquillm/aquillm/asgi.py aquillm/aquillm/urls.py aquillm/chat/views.py aquillm/tests/integration/test_context_processors_urls.py
git commit -m "refactor(runtime): route ASGI and URL wiring through apps modules"
```

---

## Chunk 2: Decouple Domain Logic From `aquillm.models` Compatibility Layer

### Task 2.1: Extract chunking task orchestration from `aquillm/models.py`

**Files:**
- Create: `aquillm/apps/documents/tasks/chunking.py`
- Create: `aquillm/apps/documents/services/chunk_progress.py`
- Create: `aquillm/apps/documents/services/image_payloads.py`
- Modify: `aquillm/apps/documents/models/document.py`
- Modify: `aquillm/apps/documents/models/chunks.py`
- Modify: `aquillm/aquillm/models.py` (re-export wrappers only)

- [ ] **Step 1: Move task implementation and helper functions to `apps.documents`**
- [ ] **Step 2: Update `Document.save()` to queue task from `apps.documents.tasks.chunking`**
- [ ] **Step 3: Replace `from aquillm.models import _doc_image_data_url` with service import**
- [ ] **Step 4: Keep a compatibility shim in `aquillm/models.py`**

```python
# aquillm/aquillm/models.py
from apps.documents.tasks.chunking import create_chunks
from apps.documents.services.image_payloads import doc_image_data_url as _doc_image_data_url
```

- [ ] **Step 5: Run targeted tests**

Run: `cd aquillm && pytest apps/documents/tests/test_multimodal_chunk_position_uniqueness.py apps/ingestion/tests/test_multimodal_ingestion_media_storage.py -q`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add aquillm/apps/documents/tasks/chunking.py aquillm/apps/documents/services/chunk_progress.py aquillm/apps/documents/services/image_payloads.py aquillm/apps/documents/models/document.py aquillm/apps/documents/models/chunks.py aquillm/aquillm/models.py
git commit -m "refactor(documents): move chunking orchestration out of compatibility module"
```

### Task 2.2: Add architecture test to prevent future reverse coupling

**Files:**
- Create: `aquillm/tests/integration/test_architecture_import_boundaries.py`

- [ ] **Step 1: Add a grep-based import boundary assertion**

```python
assert "from aquillm.models import" not in runtime_module_text
```

- [ ] **Step 2: Run architecture test**

Run: `cd aquillm && pytest tests/integration/test_architecture_import_boundaries.py -q`  
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add aquillm/tests/integration/test_architecture_import_boundaries.py
git commit -m "test(architecture): enforce no runtime imports from aquillm.models compatibility module"
```

---

## Chunk 3: Split Additional Backend Hotspots Not Covered in Prior Plan

### Task 3.1: Split `apps/documents/models/chunks.py` by responsibility

**Files:**
- Create: `aquillm/apps/documents/services/chunk_embeddings.py`
- Create: `aquillm/apps/documents/services/chunk_search.py`
- Create: `aquillm/apps/documents/services/chunk_rerank.py`
- Modify: `aquillm/apps/documents/models/chunks.py`

- [ ] **Step 1: Keep model definitions + minimal model methods in `chunks.py`**
- [ ] **Step 2: Move embedding/rerank/search orchestration into service modules**
- [ ] **Step 3: Preserve existing `TextChunk.text_chunk_search(...)` API by delegating to service**
- [ ] **Step 4: Run document/ingestion tests**

Run: `cd aquillm && pytest apps/documents/tests apps/ingestion/tests/test_unified_ingestion_parsers.py -q`  
Expected: PASS

### Task 3.2: Split `apps/ingestion/views/api.py` into endpoint modules + service layer

**Files:**
- Create: `aquillm/apps/ingestion/services/arxiv_ingest.py`
- Create: `aquillm/apps/ingestion/services/upload_batches.py`
- Create: `aquillm/apps/ingestion/services/web_ingest.py`
- Create: `aquillm/apps/ingestion/views/api/arxiv.py`
- Create: `aquillm/apps/ingestion/views/api/uploads.py`
- Create: `aquillm/apps/ingestion/views/api/web.py`
- Modify: `aquillm/apps/ingestion/views/api.py` (compat re-export map)

- [ ] **Step 1: Move heavy helper `insert_one_from_arxiv` into service module**
- [ ] **Step 2: Move upload-batch orchestration into `services/upload_batches.py`**
- [ ] **Step 3: Keep public view names stable via re-exports**
- [ ] **Step 4: Run ingestion API tests**

Run: `cd aquillm && pytest apps/ingestion/tests/test_unified_ingestion_api.py apps/ingestion/tests/test_ingestion_monitor_includes_non_pdf.py -q`  
Expected: PASS

### Task 3.3: Split `aquillm/zotero_tasks.py` and remove threaded ORM writes

**Files:**
- Create: `aquillm/apps/integrations/zotero/services/sync_collections.py`
- Create: `aquillm/apps/integrations/zotero/services/sync_items.py`
- Create: `aquillm/apps/integrations/zotero/tasks.py`
- Modify: `aquillm/aquillm/zotero_tasks.py` (compat task forwarding)
- Modify: `aquillm/aquillm/zotero_views.py` (import new task path)

- [ ] **Step 1: Keep network fan-out parallelism only for fetch/download**
- [ ] **Step 2: Persist Django models on the main task thread (no threaded ORM writes)**
- [ ] **Step 3: Preserve task signature and queue names for compatibility**
- [ ] **Step 4: Add/adjust tests for idempotent sync behavior**

Run: `cd aquillm && pytest tests/integration/test_legacy_models_compat_module.py -q`  
Expected: PASS

---

## Chunk 4: Complete Remaining Large-File Remediation (Carry-Forward)

### Task 4.1: Execute unresolved backend splits from 2026-03-19 large-file plan

**Files:**
- Create: `aquillm/apps/chat/refs.py`
- Create: `aquillm/apps/chat/services/tool_wiring.py`
- Create: `aquillm/lib/tools/search/vector_search.py`
- Create: `aquillm/lib/tools/search/context.py`
- Create: `aquillm/lib/tools/documents/list_ids.py`
- Create: `aquillm/lib/tools/documents/whole_document.py`
- Create: `aquillm/lib/tools/documents/single_document.py`
- Create: `aquillm/lib/llm/providers/fallback_heuristics.py`
- Create: `aquillm/lib/llm/providers/tool_evidence.py`
- Create: `aquillm/lib/llm/providers/image_context.py`
- Create: `aquillm/lib/llm/providers/openai_tokens.py`
- Modify: `aquillm/apps/chat/consumers/chat.py`
- Modify: `aquillm/lib/llm/providers/base.py`
- Modify: `aquillm/lib/llm/providers/openai.py`
- Modify: `aquillm/apps/chat/tests/test_messages.py` (+ split files)

- [ ] **Step 1: Run existing chat/provider tests before each extraction**
- [ ] **Step 2: Extract one concern at a time with no behavior changes**
- [ ] **Step 3: Keep `lib/tools` free of `apps.*` and ORM imports via injected callables**
- [ ] **Step 4: Re-run tests after each sub-split**

Run: `cd aquillm && pytest apps/chat/tests lib/llm/tests -q --tb=short`  
Expected: PASS

---

## Chunk 5: Complete Frontend Domain Splits + Registry Cleanup

### Task 5.1: Finish moving oversized `react/src/components/*` into feature domains

**Files:**
- Create: `react/src/features/chat/components/Chat.tsx`
- Create: `react/src/features/chat/hooks/useChatWebSocket.ts`
- Create: `react/src/features/collections/components/CollectionView.tsx`
- Create: `react/src/features/documents/components/FileSystemViewer.tsx`
- Create: `react/src/features/platform_admin/components/UserManagementModal.tsx`
- Modify: `react/src/components/ChatComponent.tsx` (back-compat re-export)
- Modify: `react/src/components/CollectionView.tsx` (back-compat re-export)
- Modify: `react/src/components/FileSystemViewer.tsx` (back-compat re-export)
- Modify: `react/src/components/UserManagementModal.tsx` (back-compat re-export)
- Modify: `react/src/main.tsx` (mount registry sourced from feature barrels)

- [ ] **Step 1: Move implementation files into `features/*` domains**
- [ ] **Step 2: Leave old paths as tiny compatibility re-exports**
- [ ] **Step 3: Replace ad-hoc `components` map in `main.tsx` with typed registry**
- [ ] **Step 4: Build + run UI tests**

Run: `cd react && npm run build`  
Expected: success (non-zero exit is failure)

---

## Chunk 6: Enforceability (CI + Structural Checks)

### Task 6.1: Add file-length and import-boundary checks to CI

**Files:**
- Create: `scripts/check_file_lengths.py`
- Create: `scripts/check_import_boundaries.py`
- Modify: `.github/workflows/hygiene-check.yml` (or split into `ci-structure-checks.yml`)

- [ ] **Step 1: Add fail-on-threshold checks (`>300` lines) with allowlist**
- [ ] **Step 2: Add boundary checks (`lib` cannot import `apps`/Django runtime modules; `apps` runtime cannot import `aquillm.models`)**
- [ ] **Step 3: Run scripts locally**

Run: `python scripts/check_file_lengths.py && python scripts/check_import_boundaries.py`  
Expected: exit code `0`

### Task 6.2: Add baseline automated test workflow

**Files:**
- Create: `.github/workflows/test-backend-frontend.yml`

- [ ] **Step 1: Add backend smoke target (`pytest` subset that runs without external services)**
- [ ] **Step 2: Add frontend build check (`npm ci`, `npm run build`)**
- [ ] **Step 3: Wire required env placeholders for deterministic CI**

---

## Chunk 7: Verification and Documentation Closure

### Task 7.1: Verification pass and execution notes

**Files:**
- Create: `docs/roadmap/plans/active/2026-03-21-architecture-boundary-and-structural-remediation-execution-notes.md`
- Modify: `README.md` (module ownership + boundary policy section)
- Modify: `docs/documents/architecture/aquillm-current-architecture-mermaid.md` (runtime path updates)

- [ ] **Step 1: Run final targeted suites**

Run:
- `cd aquillm && pytest apps/chat/tests apps/ingestion/tests apps/documents/tests lib/llm/tests tests/integration -q --tb=short`
- `cd react && npm run build`

Expected: PASS

- [ ] **Step 2: Record what changed, what was verified, and residual risks**
- [ ] **Step 3: Commit docs and notes**

```bash
git add README.md docs/documents/architecture/aquillm-current-architecture-mermaid.md docs/roadmap/plans/active/2026-03-21-architecture-boundary-and-structural-remediation-execution-notes.md
git commit -m "docs: capture architecture-boundary remediation outcomes and runbook"
```

---

## Suggested Execution Order

1. Chunk 1 (runtime entry points)
2. Chunk 2 (compatibility decoupling)
3. Chunk 4 (carry-forward large-file splits)
4. Chunk 3 (new backend hotspots)
5. Chunk 5 (frontend splits)
6. Chunk 6 (CI enforcement)
7. Chunk 7 (final verification + docs)

---

**Plan complete and saved to `docs/roadmap/plans/pending/2026-03-21-architecture-boundary-and-structural-remediation.md`. Ready to execute?**



