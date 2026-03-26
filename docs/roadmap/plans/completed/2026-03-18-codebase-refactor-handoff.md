# Codebase Refactor Handoff

**Date:** 2026-03-18
**Status:** In Progress - Chunk 8 Complete, Starting Chunk 9

## Resume Instructions

Start new chat with:
> "Continue the codebase refactor from Chunk 9 (Cleanup). Chunks 1-8 are complete. Docker validation will be done on remote at the end. Read handoff at `docs/roadmap/plans/completed/2026-03-18-codebase-refactor-handoff.md` first."

## Key Documents

- **Spec:** `docs/specs/2026-03-18-codebase-refactor-design.md`
- **Plan:** `docs/roadmap/plans/completed/2026-03-18-codebase-refactor.md`

## Progress Overview

| Chunk | Phase | Status | Commits |
|-------|-------|--------|---------|
| 1 | Create Directory Structure | âœ… COMPLETE | `b4a727e` |
| 2 | Move Models to apps/ | âœ… COMPLETE | `0b04ce7` |
| 3 | Extract lib/llm/ | âœ… COMPLETE | (pending commit) |
| 4 | Extract remaining lib/ | âœ… COMPLETE | (pending commit) |
| 5 | Restructure Views/Consumers | âœ… COMPLETE | `3d00a87` |
| 6 | Deployment Restructure | âœ… COMPLETE | `4142e24` |
| 7 | Test Migration | âœ… COMPLETE | (pending commit) |
| 8 | Frontend Restructure | âœ… COMPLETE | (pending commit) |
| 9 | Cleanup | â³ PENDING | - |

## Chunk 4 Completed Work

### lib/memory/ - Memory Backend Logic

Extracted pure Python memory logic from `aquillm/memory.py`:

| Module | Purpose |
|--------|---------|
| `lib/memory/types.py` | RetrievedEpisodicMemory dataclass |
| `lib/memory/config.py` | Configuration (EPISODIC_TOP_K, MEMORY_BACKEND, etc.) |
| `lib/memory/mem0/client.py` | Mem0 cloud and OSS client management |
| `lib/memory/mem0/operations.py` | Mem0 search and write operations |
| `lib/memory/extraction/stable_facts.py` | Fact extraction (LLM-based and heuristic) |
| `lib/memory/formatting.py` | Memory formatting for system prompts |
| `lib/memory/__init__.py` | Public API exports |

Original `aquillm/memory.py` now imports from lib/memory for pure Python operations and keeps Django model interactions.

### lib/embeddings/ - Embedding Providers

Extracted pure Python embedding logic from `aquillm/utils.py`:

| Module | Purpose |
|--------|---------|
| `lib/embeddings/config.py` | Configuration (base URL, model, context limits) |
| `lib/embeddings/local.py` | Local OpenAI-compatible embedding (vLLM) |
| `lib/embeddings/cohere.py` | Cohere embedding provider |
| `lib/embeddings/multimodal.py` | Multimodal (text + image) embedding |
| `lib/embeddings/utils.py` | Dimension fitting utilities |
| `lib/embeddings/__init__.py` | Public API exports |

Original `aquillm/utils.py` now imports from lib/embeddings and integrates with Django app config for Cohere client.

### lib/ocr/ - OCR Providers

Extracted pure Python OCR logic from `aquillm/ocr_utils.py`:

| Module | Purpose |
|--------|---------|
| `lib/ocr/config.py` | Configuration and usage logging callback |
| `lib/ocr/image_utils.py` | Image reading, MIME detection, resizing |
| `lib/ocr/tesseract.py` | Local Tesseract OCR |
| `lib/ocr/qwen.py` | Qwen vision model OCR |
| `lib/ocr/gemini.py` | Google Gemini OCR with cost tracking |
| `lib/ocr/__init__.py` | Public API with auto-selection |

Original `aquillm/ocr_utils.py` now imports from lib/ocr and registers Django model usage logger.

### lib/parsers/ - File Parsing

Extracted pure Python parsing logic from `aquillm/ingestion/parsers.py`:

| Module | Purpose |
|--------|---------|
| `lib/parsers/config.py` | Extension sets and ingest type detection |
| `lib/parsers/text_utils.py` | Text encoding utilities |
| `lib/parsers/documents/` | PDF, HTML, DOCX, EPUB parsers |
| `lib/parsers/spreadsheets/` | XLSX, XLS, ODS, CSV parsers |
| `lib/parsers/presentations/` | PPTX, ODP parsers |
| `lib/parsers/structured/` | JSON, JSONL, XML, YAML parsers |
| `lib/parsers/media/` | SRT caption parser |
| `lib/parsers/__init__.py` | Public API exports |

Original `aquillm/ingestion/parsers.py` now imports from lib/parsers for core parsing, keeps figure extraction and media transcription integration.

### lib/tools/ - NOT Extracted (Intentional)

Tool definitions in `chat/consumers.py` were NOT extracted because:
1. They're tightly coupled with Django models (User, Collection, TextChunk, Document, ConversationFile)
2. They already use dependency injection patterns (factory functions)
3. Moving them would require complex abstractions with minimal benefit

The tools remain well-organized in consumers.py with clear separation.

## Chunk 4 Verification Results

- âœ… No lib/ â†’ apps/ imports (verified with grep)
- âœ… No Django imports in lib/memory, lib/embeddings, lib/ocr, lib/parsers (pure Python)
- âœ… All lib/ files pass py_compile syntax check
- âœ… All updated aquillm files pass py_compile syntax check
- âš ï¸ Note: lib/llm/types/conversation.py has Django import (from Chunk 3) - minor tech debt

## Chunk 5 Completed Work

### Views Split by Domain

Moved views from `aquillm/api_views.py` and `aquillm/views.py` to domain-specific modules:

| Domain | API Views | Page Views |
|--------|-----------|------------|
| collections | `apps/collections/views/api.py` | `apps/collections/views/pages.py` |
| documents | `apps/documents/views/api.py` | `apps/documents/views/pages.py` |
| ingestion | `apps/ingestion/views/api.py` | `apps/ingestion/views/pages.py` |
| platform_admin | `apps/platform_admin/views/api.py` | `apps/platform_admin/views/pages.py` |
| chat | `apps/chat/views/api.py` | `apps/chat/views/pages.py` |
| core | `apps/core/views/api.py` | `apps/core/views/pages.py` |

### ChatConsumer Moved

Moved ChatConsumer to `apps/chat/consumers/chat.py` with all tool functions:
- `get_vector_search_func`
- `get_document_ids_func`
- `get_whole_document_func`
- `get_search_single_document_func`
- `get_more_context_func`
- `get_sky_subtraction_func`
- `get_flat_fielding_func`
- `get_point_source_detection_func`
- `get_weather_func` (DEBUG only)

### URL Routing

Created app-specific URL files with `api_urlpatterns` and `page_urlpatterns`:
- `apps/collections/urls.py`
- `apps/documents/urls.py`
- `apps/ingestion/urls.py`
- `apps/platform_admin/urls.py`
- `apps/chat/urls.py`
- `apps/core/urls.py`

Created WebSocket routing:
- `apps/chat/routing.py`

### Backward Compatibility

Original files now re-export from apps/ for backward compatibility:
- `aquillm/api_views.py` - thin wrapper importing from apps/.../api.py
- `aquillm/views.py` - thin wrapper importing from apps/.../pages.py
- `chat/consumers.py` - thin wrapper importing from apps/chat/consumers/

## Chunk 6 Completed Work

### Deployment Restructure

Moved all deployment files to `deploy/` directory with organized subdirectories:

#### New Directory Structure
```
deploy/
â”œâ”€â”€ docker/
â”‚   â”œâ”€â”€ web/
â”‚   â”‚   â”œâ”€â”€ Dockerfile           (was Dockerfile)
â”‚   â”‚   â””â”€â”€ Dockerfile.prod      (was Dockerfile.prod)
â”‚   â”œâ”€â”€ vllm/
â”‚   â”‚   â”œâ”€â”€ Dockerfile           (was Dockerfile.vllm)
â”‚   â”‚   â””â”€â”€ chat_templates/
â”‚   â”œâ”€â”€ certbot/
â”‚   â”‚   â””â”€â”€ Dockerfile           (was Dockerfile.certbot)
â”‚   â””â”€â”€ test/
â”‚       â””â”€â”€ Dockerfile           (was Dockerfile.test)
â”œâ”€â”€ compose/
â”‚   â”œâ”€â”€ base.yml                 (was docker-compose.yml)
â”‚   â”œâ”€â”€ development.yml          (was docker-compose-development.yml)
â”‚   â”œâ”€â”€ production.yml           (was docker-compose-prod.yml)
â”‚   â””â”€â”€ test.yml                 (was docker-compose-test.yml)
â”œâ”€â”€ nginx/
â”‚   â”œâ”€â”€ nginx.conf               (was deployment/nginx.conf)
â”‚   â””â”€â”€ aquillm.conf.template    (was deployment/aquillm.conf.template)
â””â”€â”€ scripts/
    â”œâ”€â”€ run.sh                   (was deployment/run.sh)
    â”œâ”€â”€ vllm_start.sh            (was deployment/vllm_start.sh)
    â”œâ”€â”€ healthcheck.sh           (was deployment/healthcheck.sh)
    â”œâ”€â”€ start_dev.sh             (was deployment/start_dev.sh)
    â”œâ”€â”€ get_certs.sh             (was deployment/get_certs.sh)
    â”œâ”€â”€ get_certs.cron           (was deployment/get_certs.cron)
    â”œâ”€â”€ start_mem0_local.sh      (was deployment/start_mem0_local.sh)
    â”œâ”€â”€ relaunch_mem0_oss.sh     (was deployment/relaunch_mem0_oss.sh)
    â”œâ”€â”€ nginx_start.sh           (was deployment/nginx_start.sh)
    â”œâ”€â”€ install.sh               (was deployment/install.sh)
    â”œâ”€â”€ aquillm.service          (was deployment/aquillm.service)
    â””â”€â”€ dev/
        â”œâ”€â”€ run.sh               (was dev/run.sh)
        â”œâ”€â”€ create_buckets.sh    (was dev/create_buckets.sh)
        â””â”€â”€ reload_tailwind.sh   (was dev/reload_tailwind.sh)
```

#### Files Updated with New Paths

| File | Changes |
|------|---------|
| Dockerfiles | Updated CMD/COPY paths to new locations |
| Compose files | Updated dockerfile paths, volume mounts for nginx/certs |
| deploy/scripts/start_dev.sh | Updated default compose file path |
| deploy/scripts/relaunch_mem0_oss.sh | Updated default compose file path |
| deploy/scripts/install.sh | Updated PROJECT_ROOT calculation and paths |
| deploy/scripts/aquillm.service | Updated compose and script paths |
| deploy/scripts/dev/run.sh | Updated reload_tailwind.sh path |
| README.md | Updated all deployment command references |

#### Removed Directories
- `deployment/` - moved to `deploy/`
- `dev/` - moved to `deploy/scripts/dev/`

## Chunk 7 Completed Work

### Test Migration

Moved all test files from `aquillm/aquillm/tests/` and `aquillm/chat/tests.py`, `aquillm/ingest/tests.py` to domain-specific locations:

#### lib/ Tests (Pure Python)

| Old Location | New Location |
|--------------|--------------|
| `aquillm/aquillm/tests/test_llm_tool_choice_serialization.py` | `lib/llm/tests/test_tool_choice_serialization.py` |
| `aquillm/aquillm/tests/test_embedding_context_limit_handling.py` | `lib/embeddings/tests/test_context_limit_handling.py` |
| `aquillm/aquillm/tests/test_ocr_provider_selection.py` | `lib/ocr/tests/test_provider_selection.py` |
| `aquillm/aquillm/tests/test_mem0_oss_mode.py` | `lib/memory/tests/test_mem0_oss_mode.py` |

#### apps/ Tests (Django)

| Old Location | New Location |
|--------------|--------------|
| `aquillm/chat/tests.py` | `apps/chat/tests/test_messages.py` |
| `aquillm/aquillm/tests/models_test.py` | `apps/collections/tests/test_collection_model.py` |
| `aquillm/aquillm/tests/test_multimodal_chunk_position_uniqueness.py` | `apps/documents/tests/test_multimodal_chunk_position_uniqueness.py` |
| `aquillm/aquillm/tests/test_figure_extraction.py` | `apps/ingestion/tests/test_figure_extraction.py` |
| `aquillm/aquillm/tests/test_unified_ingestion_api.py` | `apps/ingestion/tests/test_unified_ingestion_api.py` |
| `aquillm/aquillm/tests/test_unified_ingestion_parsers.py` | `apps/ingestion/tests/test_unified_ingestion_parsers.py` |
| `aquillm/aquillm/tests/test_ingestion_monitor_includes_non_pdf.py` | `apps/ingestion/tests/test_ingestion_monitor_includes_non_pdf.py` |
| `aquillm/aquillm/tests/test_multimodal_ingestion_media_storage.py` | `apps/ingestion/tests/test_multimodal_ingestion_media_storage.py` |
| `aquillm/ingest/tests.py` | `apps/ingestion/tests/test_handwritten_ingest.py` |
| `aquillm/aquillm/tests/test_transcribe_provider_selection.py` | `apps/ingestion/tests/test_transcribe_provider_selection.py` |

#### Integration Tests (Cross-cutting)

| Old Location | New Location |
|--------------|--------------|
| `aquillm/aquillm/tests/test_deployment_run_script.py` | `tests/integration/test_deployment_run_script.py` |
| `aquillm/aquillm/tests/test_dev_launch_script.py` | `tests/integration/test_dev_launch_script.py` |
| `aquillm/aquillm/tests/test_compose_multimodal_services.py` | `tests/integration/test_compose_multimodal_services.py` |
| `aquillm/aquillm/tests/test_ingest_row_ajax_header.py` | `tests/integration/test_ingest_row_ajax_header.py` |

#### Updates Made

- lib/ tests updated to import from `lib.*` instead of `aquillm.*`
- Integration tests updated with new deployment paths (`deploy/` instead of `deployment/`)
- Removed `assert False` from collections test
- Deleted empty `aquillm/aquillm/tests/` directory

### Verification

- âœ… All 18 test files pass py_compile
- âœ… No tests left in old locations

## Chunk 8 Completed Work

### Frontend Directory Structure

Created feature-based directory structure:

```
react/src/
â”œâ”€â”€ features/
â”‚   â”œâ”€â”€ chat/
â”‚   â”‚   â”œâ”€â”€ components/
â”‚   â”‚   â”‚   â”œâ”€â”€ index.ts
â”‚   â”‚   â”‚   â”œâ”€â”€ MessageBubble.tsx
â”‚   â”‚   â”‚   â”œâ”€â”€ ToolCallGroup.tsx
â”‚   â”‚   â”‚   â””â”€â”€ RatingButtons.tsx
â”‚   â”‚   â”œâ”€â”€ types/
â”‚   â”‚   â”‚   â””â”€â”€ index.ts
â”‚   â”‚   â””â”€â”€ utils/
â”‚   â”‚       â”œâ”€â”€ index.ts
â”‚   â”‚       â””â”€â”€ messageGrouping.ts
â”‚   â”‚
â”‚   â”œâ”€â”€ ingestion/
â”‚   â”‚   â”œâ”€â”€ components/
â”‚   â”‚   â”‚   â”œâ”€â”€ index.ts
â”‚   â”‚   â”‚   â”œâ”€â”€ DocTypeToggle.tsx
â”‚   â”‚   â”‚   â”œâ”€â”€ IngestRow.tsx
â”‚   â”‚   â”‚   â””â”€â”€ forms/
â”‚   â”‚   â”‚       â”œâ”€â”€ index.ts
â”‚   â”‚   â”‚       â”œâ”€â”€ UploadsForm.tsx
â”‚   â”‚   â”‚       â”œâ”€â”€ PDFForm.tsx
â”‚   â”‚   â”‚       â”œâ”€â”€ VTTForm.tsx
â”‚   â”‚   â”‚       â”œâ”€â”€ WebpageForm.tsx
â”‚   â”‚   â”‚       â”œâ”€â”€ ArxivForm.tsx
â”‚   â”‚   â”‚       â””â”€â”€ HandwrittenForm.tsx
â”‚   â”‚   â””â”€â”€ types/
â”‚   â”‚       â””â”€â”€ index.ts
â”‚   â”‚
â”‚   â”œâ”€â”€ collections/components/
â”‚   â”œâ”€â”€ documents/components/
â”‚   â””â”€â”€ platform_admin/components/
â”‚
â””â”€â”€ shared/
    â”œâ”€â”€ components/
    â”‚   â”œâ”€â”€ index.ts
    â”‚   â”œâ”€â”€ Collapsible.tsx
    â”‚   â”œâ”€â”€ ToolResult.tsx
    â”‚   â””â”€â”€ logos/
    â”‚       â”œâ”€â”€ index.ts
    â”‚       â”œâ”€â”€ AquillmLogo.tsx
    â”‚       â”œâ”€â”€ UserLogo.tsx
    â”‚       â””â”€â”€ ArxivLogo.tsx
    â”œâ”€â”€ hooks/
    â”œâ”€â”€ utils/
    â””â”€â”€ types/
```

### ChatComponent.tsx Split

Original: 1066 lines â†’ Main component: ~450 lines + extracted modules

| Extracted Module | New Location |
|-----------------|--------------|
| Types (Message, Conversation, etc.) | `features/chat/types/index.ts` |
| MessageBubble component | `features/chat/components/MessageBubble.tsx` |
| ToolCallGroup component | `features/chat/components/ToolCallGroup.tsx` |
| RatingButtons component | `features/chat/components/RatingButtons.tsx` |
| groupMessages, shouldShowSpinner | `features/chat/utils/messageGrouping.ts` |
| AquillmLogo | `shared/components/logos/AquillmLogo.tsx` |
| UserLogo | `shared/components/logos/UserLogo.tsx` |
| Collapsible | `shared/components/Collapsible.tsx` |
| ToolResult, ToolValue | `shared/components/ToolResult.tsx` |

### IngestRow.tsx Split

Original: 967 lines â†’ Main component: ~300 lines + extracted modules

| Extracted Module | New Location |
|-----------------|--------------|
| Types (DocType, IngestRowData, etc.) | `features/ingestion/types/index.ts` |
| DocTypeToggle component | `features/ingestion/components/DocTypeToggle.tsx` |
| IngestRow component | `features/ingestion/components/IngestRow.tsx` |
| UploadsForm | `features/ingestion/components/forms/UploadsForm.tsx` |
| PDFForm | `features/ingestion/components/forms/PDFForm.tsx` |
| VTTForm | `features/ingestion/components/forms/VTTForm.tsx` |
| WebpageForm | `features/ingestion/components/forms/WebpageForm.tsx` |
| ArxivForm | `features/ingestion/components/forms/ArxivForm.tsx` |
| HandwrittenForm | `features/ingestion/components/forms/HandwrittenForm.tsx` |
| ArxivLogo | `shared/components/logos/ArxivLogo.tsx` |

### Backward Compatibility

Original component files now import from new locations:
- `components/ChatComponent.tsx` imports from `features/chat/` and `shared/`
- `components/IngestRow.tsx` imports from `features/ingestion/` and re-exports `DocType`

### Verification

- âœ… `npm run build` passes successfully
- âœ… All modules correctly imported
- âœ… Build output: 628.15 kB (193.94 kB gzipped)

## What's Next

### Chunk 9: Cleanup

Final cleanup phase:
- Remove any unused imports or files
- Verify all backward compatibility wrappers work
- Run full test suite
- Docker validation on remote

## Technical Reference

### Migration Pattern (Used for All Apps)

1. New app migration: `SeparateDatabaseAndState(state_operations=[CreateModel(...)], database_operations=[])`
2. Removal migration: `SeparateDatabaseAndState(state_operations=[DeleteModel(name='X')], database_operations=[])`
3. Dependencies: All new app migrations depend on `('aquillm', '0017_document_figure_model')`
4. Removal migration depends on all new app migrations

### Model â†’ Table Name Mapping

All models use `db_table = 'aquillm_<lowercase_model_name>'` to preserve existing tables.

### App Labels

| App | Label |
|-----|-------|
| apps.collections | apps_collections |
| apps.documents | apps_documents |
| apps.chat | apps_chat |
| apps.ingestion | apps_ingestion |
| apps.memory | apps_memory |
| apps.platform_admin | apps_platform_admin |
| apps.core | apps_core |
| apps.integrations.zotero | apps_integrations_zotero |

### Backward Compatibility

`aquillm/models.py` imports and re-exports all models from new locations:
```python
from apps.documents.models import TextChunk, PDFDocument, ...
from apps.chat.models import WSConversation, Message, ...
# etc.
```

This ensures `from aquillm.models import TextChunk` continues to work.

Similarly, `aquillm/memory.py`, `aquillm/utils.py`, `aquillm/ocr_utils.py`, and `aquillm/ingestion/parsers.py` re-export from lib/ modules for backward compatibility.

### Docker Validation (At End of Refactor)

Will be run on remote machine:
```bash
docker compose -f deploy/compose/development.yml run --rm web python manage.py check
docker compose -f deploy/compose/development.yml run --rm web python manage.py migrate --plan
```

## Files Changed This Session (Chunk 5)

### New Files Created

apps/collections/views/:
- `apps/collections/views/api.py`
- `apps/collections/views/pages.py`
- `apps/collections/views/__init__.py`
- `apps/collections/urls.py`

apps/documents/views/:
- `apps/documents/views/api.py`
- `apps/documents/views/pages.py`
- `apps/documents/views/__init__.py`
- `apps/documents/urls.py`

apps/ingestion/views/:
- `apps/ingestion/views/api.py`
- `apps/ingestion/views/pages.py`
- `apps/ingestion/views/__init__.py`
- `apps/ingestion/urls.py`

apps/platform_admin/views/:
- `apps/platform_admin/views/api.py`
- `apps/platform_admin/views/pages.py`
- `apps/platform_admin/views/__init__.py`
- `apps/platform_admin/urls.py`

apps/chat/:
- `apps/chat/views/api.py`
- `apps/chat/views/pages.py`
- `apps/chat/views/__init__.py`
- `apps/chat/urls.py`
- `apps/chat/routing.py`
- `apps/chat/consumers/chat.py`
- `apps/chat/consumers/__init__.py`

apps/core/views/:
- `apps/core/views/api.py`
- `apps/core/views/pages.py`
- `apps/core/views/__init__.py`
- `apps/core/urls.py`

### Files Modified (Backward Compatibility)

- `aquillm/api_views.py` - Now thin wrapper importing from apps/
- `aquillm/views.py` - Now thin wrapper importing from apps/
- `chat/consumers.py` - Now thin wrapper importing from apps/chat/consumers/

---

## Files Changed (Chunk 4)

### New Files Created

lib/memory/:
- `lib/memory/__init__.py`
- `lib/memory/config.py`
- `lib/memory/types.py`
- `lib/memory/formatting.py`
- `lib/memory/mem0/__init__.py`
- `lib/memory/mem0/client.py`
- `lib/memory/mem0/operations.py`
- `lib/memory/extraction/__init__.py`
- `lib/memory/extraction/stable_facts.py`

lib/embeddings/:
- `lib/embeddings/__init__.py`
- `lib/embeddings/config.py`
- `lib/embeddings/local.py`
- `lib/embeddings/cohere.py`
- `lib/embeddings/multimodal.py`
- `lib/embeddings/utils.py`

lib/ocr/:
- `lib/ocr/__init__.py`
- `lib/ocr/config.py`
- `lib/ocr/image_utils.py`
- `lib/ocr/tesseract.py`
- `lib/ocr/qwen.py`
- `lib/ocr/gemini.py`

lib/parsers/:
- `lib/parsers/__init__.py`
- `lib/parsers/config.py`
- `lib/parsers/text_utils.py`
- `lib/parsers/documents/__init__.py`
- `lib/parsers/documents/pdf.py`
- `lib/parsers/documents/html.py`
- `lib/parsers/documents/docx.py`
- `lib/parsers/documents/epub.py`
- `lib/parsers/spreadsheets/__init__.py`
- `lib/parsers/spreadsheets/xlsx.py`
- `lib/parsers/spreadsheets/xls.py`
- `lib/parsers/spreadsheets/ods.py`
- `lib/parsers/spreadsheets/csv_parser.py`
- `lib/parsers/presentations/__init__.py`
- `lib/parsers/presentations/pptx.py`
- `lib/parsers/presentations/odp.py`
- `lib/parsers/structured/__init__.py`
- `lib/parsers/structured/json_parser.py`
- `lib/parsers/structured/xml_parser.py`
- `lib/parsers/structured/yaml_parser.py`
- `lib/parsers/media/__init__.py`
- `lib/parsers/media/vtt.py`

### Files Modified

- `aquillm/aquillm/memory.py` - Now imports from lib/memory
- `aquillm/aquillm/utils.py` - Now imports from lib/embeddings
- `aquillm/aquillm/ocr_utils.py` - Now imports from lib/ocr
- `aquillm/aquillm/ingestion/parsers.py` - Now imports from lib/parsers

## Files Changed (Chunk 6)

### New Files Created (Moved)

deploy/docker/:
- `deploy/docker/web/Dockerfile` (from Dockerfile)
- `deploy/docker/web/Dockerfile.prod` (from Dockerfile.prod)
- `deploy/docker/vllm/Dockerfile` (from Dockerfile.vllm)
- `deploy/docker/vllm/chat_templates/qwen3_vl_reranker.jinja`
- `deploy/docker/certbot/Dockerfile` (from Dockerfile.certbot)
- `deploy/docker/test/Dockerfile` (from Dockerfile.test)

deploy/compose/:
- `deploy/compose/base.yml` (from docker-compose.yml)
- `deploy/compose/development.yml` (from docker-compose-development.yml)
- `deploy/compose/production.yml` (from docker-compose-prod.yml)
- `deploy/compose/test.yml` (from docker-compose-test.yml)

deploy/nginx/:
- `deploy/nginx/nginx.conf`
- `deploy/nginx/aquillm.conf.template`

deploy/scripts/:
- `deploy/scripts/run.sh`
- `deploy/scripts/vllm_start.sh`
- `deploy/scripts/healthcheck.sh`
- `deploy/scripts/start_dev.sh`
- `deploy/scripts/get_certs.sh`
- `deploy/scripts/get_certs.cron`
- `deploy/scripts/start_mem0_local.sh`
- `deploy/scripts/relaunch_mem0_oss.sh`
- `deploy/scripts/nginx_start.sh`
- `deploy/scripts/install.sh`
- `deploy/scripts/aquillm.service`
- `deploy/scripts/dev/run.sh`
- `deploy/scripts/dev/create_buckets.sh`
- `deploy/scripts/dev/reload_tailwind.sh`

### Files Deleted (Moved)

- `Dockerfile` â†’ `deploy/docker/web/Dockerfile`
- `Dockerfile.prod` â†’ `deploy/docker/web/Dockerfile.prod`
- `Dockerfile.vllm` â†’ `deploy/docker/vllm/Dockerfile`
- `Dockerfile.certbot` â†’ `deploy/docker/certbot/Dockerfile`
- `Dockerfile.test` â†’ `deploy/docker/test/Dockerfile`
- `docker-compose.yml` â†’ `deploy/compose/base.yml`
- `docker-compose-development.yml` â†’ `deploy/compose/development.yml`
- `docker-compose-prod.yml` â†’ `deploy/compose/production.yml`
- `docker-compose-test.yml` â†’ `deploy/compose/test.yml`
- `deployment/` directory (moved to `deploy/`)
- `dev/` directory (moved to `deploy/scripts/dev/`)

### Files Modified

- `README.md` - Updated all deployment command references

## Files Changed (Chunk 8)

### New Files Created

react/src/features/chat/:
- `react/src/features/chat/types/index.ts`
- `react/src/features/chat/components/index.ts`
- `react/src/features/chat/components/MessageBubble.tsx`
- `react/src/features/chat/components/ToolCallGroup.tsx`
- `react/src/features/chat/components/RatingButtons.tsx`
- `react/src/features/chat/utils/index.ts`
- `react/src/features/chat/utils/messageGrouping.ts`

react/src/features/ingestion/:
- `react/src/features/ingestion/types/index.ts`
- `react/src/features/ingestion/components/index.ts`
- `react/src/features/ingestion/components/DocTypeToggle.tsx`
- `react/src/features/ingestion/components/IngestRow.tsx`
- `react/src/features/ingestion/components/forms/index.ts`
- `react/src/features/ingestion/components/forms/UploadsForm.tsx`
- `react/src/features/ingestion/components/forms/PDFForm.tsx`
- `react/src/features/ingestion/components/forms/VTTForm.tsx`
- `react/src/features/ingestion/components/forms/WebpageForm.tsx`
- `react/src/features/ingestion/components/forms/ArxivForm.tsx`
- `react/src/features/ingestion/components/forms/HandwrittenForm.tsx`

react/src/shared/:
- `react/src/shared/components/index.ts`
- `react/src/shared/components/Collapsible.tsx`
- `react/src/shared/components/ToolResult.tsx`
- `react/src/shared/components/logos/index.ts`
- `react/src/shared/components/logos/AquillmLogo.tsx`
- `react/src/shared/components/logos/UserLogo.tsx`
- `react/src/shared/components/logos/ArxivLogo.tsx`

### Files Modified

- `react/src/components/ChatComponent.tsx` - Now imports from features/chat/ and shared/
- `react/src/components/IngestRow.tsx` - Now imports from features/ingestion/ and shared/

## Constraints

- **Local validation:** `python -m py_compile <file>` âœ… All files pass
- **Docker validation:** `manage.py check`, `migrate --plan` â³ Requires Docker
- **Backward compatibility:** Old imports from aquillm modules must continue working
- **No DB changes:** All migrations use SeparateDatabaseAndState
- **lib/ purity:** lib/ modules should not import from apps/ (verified) or Django models (mostly achieved)



