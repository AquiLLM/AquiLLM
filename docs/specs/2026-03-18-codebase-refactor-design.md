# AquiLLM Codebase Refactor Design

**Date:** 2026-03-18  
**Status:** Draft  
**Goal:** Comprehensive refactor for auditability, maintainability, and onboarding

## Overview

Restructure the AquiLLM codebase from a flat, bloated file structure to a clean, domain-driven architecture with clear separation between Django apps, reusable library code, and deployment configuration.

### Principles

1. **No file over 300 lines** (target, with exceptions like `settings.py`) - Split large files by responsibility
2. **Domain-driven organization** - Group by business domain, not technical layer
3. **Extractable libraries** - `lib/` contains pure Python that could become packages
4. **Frontend mirrors backend** - Easy to find related code across stack
5. **No code duplication** - Shared utilities in appropriate locations
6. **Future-ready** - Structure supports agents, skills, MCP expansion

## Current State

### Bloated Files (Python)

| File | Lines | Issue |
|------|-------|-------|
| `llm.py` | 1988 | Message types + 3 LLM providers + tools in one file |
| `models.py` | 1752 | 25+ Django models in one file |
| `consumers.py` | 825 | WebSocket handler + all tool definitions |
| `api_views.py` | 761 | All API endpoints |
| `memory.py` | 706 | Memory backends + extraction logic |
| `views.py` | 559 | Mixed template and API views |
| `parsers.py` | 483 | All document parsers |
| `utils.py` | 399 | Embedding utilities |
| `ocr_utils.py` | 388 | OCR providers |

### Bloated Files (React)

| File | Lines | Issue |
|------|-------|-------|
| `ChatComponent.tsx` | 1001 | Chat UI + WebSocket + messages + tools |
| `IngestRow.tsx` | 913 | Container + 6 form types + toggle |
| `CollectionView.tsx` | 634 | View + settings + tree |
| `FileSystemViewer.tsx` | 515 | Viewer + document list |
| `UserManagementModal.tsx` | 454 | Management + table + edit |

### Current Structure Issues

- Flat directory structure in `aquillm/aquillm/`
- Mixed concerns in large files
- Duplicate code (`insert_one_from_arxiv` in two files)
- Inconsistent test organization
- Deployment files scattered at root

## Target Architecture

### Top-Level Structure

```
aquillm/
├── aquillm/                 # Django project config ONLY
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py
│   ├── wsgi.py
│   └── celery.py
├── apps/                    # Django applications
│   ├── chat/
│   ├── documents/
│   ├── collections/
│   ├── ingestion/
│   ├── memory/
│   ├── platform_admin/
│   └── core/
├── lib/                     # Reusable, non-Django code
│   ├── llm/
│   ├── tools/
│   ├── memory/
│   ├── embeddings/
│   ├── ocr/
│   ├── parsers/
│   ├── integrations/
│   ├── agents/              # Future
│   ├── skills/              # Future
│   └── mcp/                 # Future
├── tests/                   # Integration tests
├── deploy/                  # All deployment config
│   ├── docker/
│   ├── compose/
│   ├── nginx/
│   ├── scripts/
│   └── k8s/                 # Future
├── react/                   # Frontend
└── docs/
```

### Backend: `lib/` Structure

```
lib/
├── llm/
│   ├── __init__.py              # Public API exports
│   ├── types/
│   │   ├── __init__.py
│   │   ├── messages.py          # UserMessage, AssistantMessage, ToolMessage
│   │   ├── conversation.py      # Conversation class
│   │   ├── tools.py             # LLMTool, ToolChoice, ToolResultDict
│   │   └── response.py          # LLMResponse
│   ├── decorators/
│   │   └── tool.py              # @llm_tool decorator
│   ├── providers/
│   │   ├── __init__.py          # Provider factory
│   │   ├── base.py              # LLMInterface ABC
│   │   ├── claude.py            # ClaudeInterface
│   │   ├── openai.py            # OpenAIInterface
│   │   └── gemini.py            # GeminiInterface
│   └── utils/
│       └── tokens.py            # Token counting
│
├── tools/
│   ├── __init__.py              # Tool registry
│   ├── base.py                  # Tool utilities
│   ├── search/
│   │   ├── vector_search.py
│   │   ├── single_document.py
│   │   └── context.py
│   ├── documents/
│   │   ├── list_ids.py
│   │   └── fetch.py
│   ├── astronomy/
│   │   ├── sky_subtraction.py
│   │   ├── flat_fielding.py
│   │   └── point_source.py
│   └── debug/
│       └── weather.py
│
├── memory/
│   ├── __init__.py              # get_memory_backend()
│   ├── types.py                 # RetrievedEpisodicMemory
│   ├── base.py                  # MemoryBackend ABC
│   ├── local.py                 # LocalMemoryBackend
│   ├── mem0/
│   │   ├── client.py
│   │   ├── search.py
│   │   └── write.py
│   └── extraction/
│       ├── stable_facts.py
│       ├── heuristics.py
│       └── formatting.py
│
├── embeddings/
│   ├── __init__.py              # get_embedding, get_embeddings
│   ├── base.py
│   ├── local.py
│   ├── cohere.py
│   ├── multimodal.py
│   └── utils.py
│
├── ocr/
│   ├── __init__.py              # extract_text_from_image
│   ├── base.py
│   ├── tesseract.py
│   ├── qwen.py
│   ├── gemini.py
│   └── utils.py
│
├── parsers/
│   ├── __init__.py              # extract_text_payloads
│   ├── base.py
│   ├── documents/
│   │   ├── pdf.py
│   │   ├── docx.py
│   │   ├── html.py
│   │   └── epub.py
│   ├── spreadsheets/
│   │   ├── xlsx.py
│   │   ├── xls.py
│   │   ├── ods.py
│   │   └── csv.py
│   ├── presentations/
│   │   ├── pptx.py
│   │   └── odp.py
│   ├── structured/
│   │   ├── json.py
│   │   ├── xml.py
│   │   └── yaml.py
│   ├── media/
│   │   ├── vtt.py
│   │   ├── srt.py
│   │   ├── image.py
│   │   └── audio_video.py
│   ├── archive.py
│   └── utils.py
│
├── integrations/
│   └── zotero/
│       ├── __init__.py
│       └── client.py             # Pure API client only
│
├── agents/                      # Future: Agentic frameworks
│   ├── __init__.py
│   ├── base.py
│   ├── orchestrator.py
│   └── strategies/
│
├── skills/                      # Future: Skill system
│   ├── __init__.py
│   ├── base.py
│   ├── registry.py
│   └── builtin/
│
└── mcp/                         # Future: Model Context Protocol
    ├── __init__.py
    ├── client.py
    ├── server.py
    └── adapters/
```

### Backend: `apps/` Structure

```
apps/
├── chat/
│   ├── __init__.py
│   ├── apps.py
│   ├── urls.py
│   ├── routing.py
│   ├── consumers/
│   │   ├── __init__.py
│   │   ├── chat.py              # ChatConsumer
│   │   └── utils.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── conversation.py      # WSConversation
│   │   ├── message.py           # Message
│   │   └── file.py              # ConversationFile
│   ├── views/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   └── pages.py
│   ├── refs.py                  # CollectionsRef, ChatRef
│   ├── admin.py
│   └── tests/
│
├── documents/
│   ├── __init__.py
│   ├── apps.py
│   ├── urls.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── document.py          # Document base
│   │   ├── document_types/
│   │   │   ├── __init__.py
│   │   │   ├── pdf.py
│   │   │   ├── tex.py
│   │   │   ├── vtt.py
│   │   │   ├── handwritten.py
│   │   │   ├── image.py
│   │   │   ├── media.py
│   │   │   ├── raw_text.py
│   │   │   └── figure.py
│   │   ├── chunks.py            # TextChunk
│   │   └── exceptions.py
│   ├── views/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   └── pages.py
│   ├── admin.py
│   └── tests/
│
├── collections/
│   ├── __init__.py
│   ├── apps.py
│   ├── urls.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── collection.py        # Collection
│   │   └── permission.py        # CollectionPermission
│   ├── views/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   └── pages.py
│   ├── admin.py
│   └── tests/
│
├── ingestion/
│   ├── __init__.py
│   ├── apps.py
│   ├── urls.py
│   ├── routing.py
│   ├── consumers.py             # Ingestion status WebSocket
│   ├── models/
│   │   ├── __init__.py
│   │   └── batch.py             # IngestionBatch, IngestionBatchItem
│   ├── views/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   └── monitor.py
│   ├── services/
│   │   ├── __init__.py
│   │   └── arxiv.py             # insert_one_from_arxiv (deduplicated)
│   ├── figure_extraction/       # Existing, keep structure
│   ├── tasks.py
│   ├── admin.py
│   └── tests/
│
├── memory/
│   ├── __init__.py
│   ├── apps.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── facts.py             # UserMemoryFact
│   │   └── episodic.py          # EpisodicMemory
│   ├── tasks.py
│   ├── admin.py
│   └── tests/
│
├── platform_admin/              # Named to avoid confusion with django.contrib.admin
│   ├── __init__.py
│   ├── apps.py
│   ├── urls.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── whitelist.py         # EmailWhitelist
│   │   └── usage.py             # GeminiAPIUsage
│   ├── views/
│   │   ├── __init__.py
│   │   ├── users.py
│   │   ├── whitelist.py
│   │   └── monitoring.py
│   └── admin.py
│
├── core/
│   ├── __init__.py
│   ├── apps.py
│   ├── urls.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── user_settings.py     # UserSettings
│   ├── views/
│   │   ├── __init__.py
│   │   ├── index.py
│   │   ├── search.py
│   │   └── settings.py
│   └── context_processors.py
│
└── integrations/
    └── zotero/                  # Django app for Zotero (has models/views)
        ├── __init__.py
        ├── apps.py
        ├── models.py            # ZoteroConnection
        ├── views.py
        ├── oauth.py
        ├── tasks.py
        ├── urls.py
        └── admin.py
```

### Deployment Structure

```
deploy/
├── docker/
│   ├── web/
│   │   ├── Dockerfile
│   │   └── Dockerfile.prod
│   ├── worker/
│   │   └── Dockerfile
│   ├── vllm/
│   │   └── Dockerfile.vllm
│   └── certbot/
│       └── Dockerfile.certbot
├── compose/
│   ├── base.yml
│   ├── development.yml
│   ├── production.yml
│   └── test.yml
├── nginx/
│   ├── nginx.conf
│   └── aquillm.conf.template
├── scripts/
│   ├── run.sh
│   ├── start_dev.sh
│   ├── healthcheck.sh
│   ├── vllm_start.sh
│   ├── relaunch_mem0_oss.sh
│   └── ...
└── k8s/                         # Future
```

### Frontend Structure

```
react/src/
├── app/
│   ├── App.tsx
│   ├── main.tsx
│   └── routes.tsx
│
├── features/
│   ├── chat/
│   │   ├── components/
│   │   │   ├── index.ts
│   │   │   ├── Chat.tsx
│   │   │   ├── MessageList.tsx
│   │   │   ├── MessageBubble.tsx
│   │   │   ├── MessageInput.tsx
│   │   │   ├── ToolCallGroup.tsx
│   │   │   ├── ToolResult.tsx
│   │   │   ├── Collapsible.tsx
│   │   │   ├── RatingButtons.tsx
│   │   │   └── CollectionSelector.tsx
│   │   ├── hooks/
│   │   │   ├── index.ts
│   │   │   ├── useChatWebSocket.ts
│   │   │   ├── useMessages.ts
│   │   │   └── useScrollToBottom.ts
│   │   ├── types/
│   │   │   └── index.ts
│   │   └── utils/
│   │       └── messageGrouping.ts
│   │
│   ├── ingestion/
│   │   ├── components/
│   │   │   ├── index.ts
│   │   │   ├── IngestRowsContainer.tsx
│   │   │   ├── IngestRow.tsx
│   │   │   ├── DocTypeToggle.tsx
│   │   │   ├── forms/
│   │   │   │   ├── UploadsForm.tsx
│   │   │   │   ├── PDFForm.tsx
│   │   │   │   ├── VTTForm.tsx
│   │   │   │   ├── WebpageForm.tsx
│   │   │   │   ├── ArxivForm.tsx
│   │   │   │   └── HandwrittenForm.tsx
│   │   │   └── PDFMonitor.tsx
│   │   ├── hooks/
│   │   │   └── useIngestionStatus.ts
│   │   └── types/
│   │       └── index.ts
│   │
│   ├── collections/
│   │   ├── components/
│   │   │   ├── index.ts
│   │   │   ├── CollectionsPage.tsx
│   │   │   ├── CollectionView.tsx
│   │   │   ├── CollectionHeader.tsx
│   │   │   ├── CollectionTree.tsx
│   │   │   ├── CollectionSettings.tsx
│   │   │   ├── MoveCollectionModal.tsx
│   │   │   └── CreateCollectionModal.tsx
│   │   ├── hooks/
│   │   │   └── useCollections.ts
│   │   └── types/
│   │       └── index.ts
│   │
│   ├── documents/
│   │   ├── components/
│   │   │   ├── index.ts
│   │   │   ├── FileSystemViewer.tsx
│   │   │   ├── DocumentList.tsx
│   │   │   ├── DocumentRow.tsx
│   │   │   └── DocumentPreview.tsx
│   │   ├── hooks/
│   │   │   └── useDocuments.ts
│   │   └── types/
│   │       └── index.ts
│   │
│   ├── platform_admin/
│   │   ├── components/
│   │   │   ├── index.ts
│   │   │   ├── UserManagement.tsx
│   │   │   ├── UserTable.tsx
│   │   │   ├── UserEditModal.tsx
│   │   │   ├── UserSettings.tsx
│   │   │   └── WhitelistEmails.tsx
│   │   └── hooks/
│   │       └── useUsers.ts
│   │
│   └── search/
│       └── components/
│           └── SearchPage.tsx
│
├── shared/
│   ├── components/
│   │   ├── index.ts
│   │   ├── Button.tsx
│   │   ├── Modal.tsx
│   │   ├── ContextMenu.tsx
│   │   ├── Spinner.tsx
│   │   └── logos/
│   │       ├── AquillmLogo.tsx
│   │       ├── UserLogo.tsx
│   │       └── ArxivLogo.tsx
│   ├── hooks/
│   │   ├── index.ts
│   │   ├── useCsrf.ts
│   │   └── useAuth.ts
│   ├── utils/
│   │   ├── index.ts
│   │   ├── formatUrl.ts
│   │   └── csrf.ts
│   └── types/
│       └── index.ts
│
└── assets/
    └── icons/
```

## Dependency Rules

To avoid circular imports between `lib/` and `apps/`:

```
apps/ → lib/        ✓ ALLOWED (apps can import from lib)
lib/ → apps/        ✗ FORBIDDEN (lib should never import from apps)
lib/ → lib/         ✓ ALLOWED (lib modules can import each other)
apps/ → apps/       ✓ ALLOWED (apps can import from other apps)
```

**For tools that need database access:**
- Tools receive model instances or query results as parameters
- Tools do NOT import Django models directly
- The consumer/view creates the query, passes results to tools

Example:
```python
# lib/tools/search/vector_search.py
def create(user_id: int, collection_ids: list[int], search_func):
    """Factory that receives a search function, not Django imports."""
    @llm_tool(...)
    def vector_search(search_string: str, top_k: int) -> ToolResultDict:
        results = search_func(search_string, top_k)  # Injected dependency
        return {"result": results}
    return vector_search

# apps/chat/consumers/chat.py
from apps.documents.models import TextChunk

def get_search_func(user, collection_ids):
    def search(query, top_k):
        return TextChunk.text_chunk_search(query, top_k, ...)
    return search

# Consumer injects the dependency
tools = [create(user.id, col_ids, get_search_func(user, col_ids))]
```

## Django Migrations Strategy

Moving models across apps requires careful handling of Django migrations.

### Approach: Use `db_table` Meta Option

Keep existing database table names to avoid data migration:

```python
# apps/collections/models/collection.py
class Collection(models.Model):
    class Meta:
        db_table = 'aquillm_collection'  # Keep original table name
```

### Migration Steps

1. **Before moving models:**
   - Run `./manage.py makemigrations` to ensure current state is clean
   - Commit all existing migrations

2. **Create new app structure:**
   - Create new apps with empty `models.py`
   - Add apps to `INSTALLED_APPS`

3. **Move models one at a time:**
   - Copy model to new location with `db_table` meta
   - Update all imports to new location
   - Create migration in new app using `migrations.SeparateDatabaseAndState`:
   ```python
   operations = [
       migrations.SeparateDatabaseAndState(
           state_operations=[
               migrations.CreateModel(
                   name='Collection',
                   fields=[...],
                   options={'db_table': 'aquillm_collection'},
               ),
           ],
           database_operations=[],  # No DB changes - table already exists
       ),
   ]
   ```
   - Create migration in old app to remove model from state (not DB)
   - Delete model from old location

4. **Handle foreign keys:**
   - Models with FKs to moved models need migration updates
   - Use `migrations.AlterField` to update FK references

5. **Verify:**
   - Run `./manage.py migrate --plan` to check migration order
   - Run `./manage.py migrate` on fresh DB clone first

### Rollback Plan

- Keep old migrations as backup
- Use git branches per phase
- Test on DB clone before production

## Migration Strategy

### Phase 1: Create Structure (No Code Changes)
1. Create all new directories (`apps/`, `lib/`, `deploy/`)
2. Create `__init__.py` files with proper exports
3. Create placeholder `apps.py` for each new Django app
4. Add new apps to `INSTALLED_APPS` (pointing to empty apps)
5. **Checkpoint:** `git commit -m "Phase 1: Create structure"`

### Phase 2: Move Models (Database-Safe)
1. Move models to new apps one domain at a time:
   - `Collection`, `CollectionPermission` → `apps/collections/`
   - `Document`, document types, `TextChunk` → `apps/documents/`
   - `WSConversation`, `Message`, `ConversationFile` → `apps/chat/`
   - `IngestionBatch`, `IngestionBatchItem` → `apps/ingestion/`
   - `UserMemoryFact`, `EpisodicMemory` → `apps/memory/`
   - `EmailWhitelist`, `GeminiAPIUsage` → `apps/platform_admin/`
   - `UserSettings` → `apps/core/`
   - `ZoteroConnection` → `apps/integrations/zotero/`
2. Use `db_table` to preserve table names
3. Create proper migrations using `SeparateDatabaseAndState`
4. Update all model imports throughout codebase
5. Run full test suite
6. **Checkpoint:** `git commit -m "Phase 2: Move models"`

### Phase 3: Extract `lib/` (No Django Dependencies)
1. Split `llm.py` → `lib/llm/`
2. Split `memory.py` → `lib/memory/` (backend logic only)
3. Split `utils.py` → `lib/embeddings/`
4. Split `ocr_utils.py` → `lib/ocr/`
5. Move `ingestion/parsers.py` → `lib/parsers/`
6. Extract tools from `consumers.py` → `lib/tools/`
7. Move Zotero client → `lib/integrations/zotero/client.py`
8. Update imports, ensure no `lib/` → `apps/` imports
9. Run full test suite
10. **Checkpoint:** `git commit -m "Phase 3: Extract lib"`

### Phase 4: Restructure Views/Consumers
1. Split `api_views.py` and `views.py` across apps
2. Split `consumers.py` → `apps/chat/consumers/`
3. Update URL routing in each app
4. Update main `urls.py` to include app URLs
5. Run full test suite
6. **Checkpoint:** `git commit -m "Phase 4: Restructure views"`

### Phase 5: Deployment Restructure
1. Move Dockerfiles → `deploy/docker/`
2. Move compose files → `deploy/compose/`
3. Move scripts → `deploy/scripts/`
4. Update all path references in:
   - GitHub Actions workflows
   - README.md
   - Any scripts that reference compose files
5. Test Docker builds
6. **Checkpoint:** `git commit -m "Phase 5: Restructure deployment"`

### Phase 6: Frontend Restructure
1. Create feature directories
2. Split `ChatComponent.tsx` into smaller components
3. Split `IngestRow.tsx` into smaller components
4. Split remaining large components
5. Extract shared components to `shared/`
6. Update all imports
7. Run frontend tests/build
8. **Checkpoint:** `git commit -m "Phase 6: Restructure frontend"`

### Phase 7: Cleanup
1. Run full test suite (backend + frontend)
2. Fix any broken imports
3. Remove empty old files
4. Update documentation
5. Final review
6. **Checkpoint:** `git commit -m "Phase 7: Cleanup"`

## Test Migration

### Current Test Locations
```
aquillm/aquillm/tests/           # 17 test files
aquillm/chat/tests.py            # Chat tests
aquillm/ingest/tests.py          # Ingest tests
```

### New Test Locations
| Current File | New Location |
|--------------|--------------|
| `tests/test_figure_extraction.py` | `apps/ingestion/tests/test_figure_extraction.py` |
| `tests/test_unified_ingestion_*.py` | `apps/ingestion/tests/` |
| `tests/test_multimodal_*.py` | `apps/documents/tests/` |
| `tests/test_embedding_*.py` | `lib/embeddings/tests/` |
| `tests/test_ocr_*.py` | `lib/ocr/tests/` |
| `tests/test_transcribe_*.py` | `lib/parsers/tests/` |
| `tests/test_mem0_*.py` | `lib/memory/tests/` |
| `tests/test_llm_*.py` | `lib/llm/tests/` |
| `tests/test_compose_*.py` | `tests/integration/` |
| `tests/test_dev_*.py` | `tests/integration/` |
| `tests/test_deployment_*.py` | `tests/integration/` |
| `chat/tests.py` | `apps/chat/tests/` |
| `ingest/tests.py` | `apps/ingestion/tests/` |

## Import Updates

### Example: Old → New

```python
# Old
from aquillm.llm import UserMessage, Conversation, LLMInterface
from aquillm.models import Collection, TextChunk, WSConversation

# New
from lib.llm import UserMessage, Conversation, LLMInterface
from apps.collections.models import Collection
from apps.documents.models import TextChunk
from apps.chat.models import WSConversation
```

### Django Settings Update

```python
# settings.py
INSTALLED_APPS = [
    # ...
    'apps.chat',
    'apps.documents',
    'apps.collections',
    'apps.ingestion',
    'apps.memory',
    'apps.platform_admin',           # Named to avoid confusion with django.contrib.admin
    'apps.core',
    'apps.integrations.zotero',
]
```

## Testing Strategy

### Unit Tests
- Live alongside code: `apps/chat/tests/`, `lib/llm/tests/`
- Run with: `pytest apps/ lib/`

### Integration Tests
- Separate directory: `tests/`
- Cross-app and end-to-end tests
- Run with: `pytest tests/`

### Migration Verification
- Run full test suite after each phase
- Verify all imports resolve
- Check Docker builds
- Manual smoke test of key features

## Success Criteria

1. **No file over 300 lines** (exceptions: `settings.py`, complex provider implementations)
2. **All tests pass** (backend and frontend)
3. **Docker builds work** (dev and prod)
4. **No duplicate code** (shared utilities properly extracted)
5. **Clear import paths** (easy to understand where things live)
6. **Easy to find any code by domain** (consistent structure)
7. **No `lib/` → `apps/` imports** (dependency direction enforced)
8. **Each phase has a working commit** (rollback possible)

## Future Extensibility

This structure supports:

- **New LLM providers**: Add to `lib/llm/providers/`
- **New tools by domain**: Add to `lib/tools/<domain>/`
- **New document parsers**: Add to `lib/parsers/<category>/`
- **Agentic frameworks**: Implement in `lib/agents/`
- **Skill system**: Implement in `lib/skills/`
- **MCP integration**: Implement in `lib/mcp/`
- **Microservice extraction**: `lib/` modules are already isolated
