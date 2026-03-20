# AquiLLM Codebase Refactor Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure AquiLLM from flat file structure to domain-driven architecture with clear separation between Django apps, reusable library code, and deployment configuration.

**Architecture:** Hybrid domain/layer approach with `apps/` for Django apps, `lib/` for reusable pure Python, and `deploy/` for all deployment config. Frontend mirrors backend domains in `features/`.

**Tech Stack:** Django 5.1, React, PostgreSQL, Celery, Docker Compose

**Spec:** `docs/superpowers/specs/2026-03-18-codebase-refactor-design.md`

---

> **VALIDATION CONSTRAINT:** Full build (Docker, vLLM, integration tests) requires remote pipeline. However, local validation IS available for: Python syntax, linting, import verification, and Django configuration checks. Use local checks after each phase; save full integration testing for final remote push.

---

## Chunk 1: Phase 1 - Create Directory Structure

### Task 1.1: Create Backend Directory Structure

**Files:**
- Create: `aquillm/apps/__init__.py`
- Create: `aquillm/apps/chat/__init__.py`
- Create: `aquillm/apps/documents/__init__.py`
- Create: `aquillm/apps/collections/__init__.py`
- Create: `aquillm/apps/ingestion/__init__.py`
- Create: `aquillm/apps/memory/__init__.py`
- Create: `aquillm/apps/platform_admin/__init__.py`
- Create: `aquillm/apps/core/__init__.py`
- Create: `aquillm/apps/integrations/__init__.py`
- Create: `aquillm/apps/integrations/zotero/__init__.py`
- Create: `aquillm/lib/__init__.py`
- Create: `aquillm/lib/llm/__init__.py`
- Create: `aquillm/lib/tools/__init__.py`
- Create: `aquillm/lib/memory/__init__.py`
- Create: `aquillm/lib/embeddings/__init__.py`
- Create: `aquillm/lib/ocr/__init__.py`
- Create: `aquillm/lib/parsers/__init__.py`
- Create: `aquillm/lib/integrations/__init__.py`

- [ ] **Step 1: Create apps directory structure**

```powershell
cd aquillm
# Create all app directories
$appDirs = @(
    "apps/chat/consumers", "apps/chat/models", "apps/chat/views", "apps/chat/tests",
    "apps/documents/models/document_types", "apps/documents/views", "apps/documents/tests",
    "apps/collections/models", "apps/collections/views", "apps/collections/tests",
    "apps/ingestion/models", "apps/ingestion/views", "apps/ingestion/services", "apps/ingestion/tests",
    "apps/memory/models", "apps/memory/tests",
    "apps/platform_admin/models", "apps/platform_admin/views", "apps/platform_admin/tests",
    "apps/core/models", "apps/core/views", "apps/core/tests",
    "apps/integrations", "apps/integrations/zotero"
)
$appDirs | ForEach-Object { New-Item -ItemType Directory -Path $_ -Force }
```

- [ ] **Step 2: Create lib directory structure**

```powershell
$libDirs = @(
    "lib/llm/types", "lib/llm/providers", "lib/llm/decorators", "lib/llm/utils",
    "lib/tools/search", "lib/tools/documents", "lib/tools/astronomy", "lib/tools/debug",
    "lib/memory/mem0", "lib/memory/extraction",
    "lib/embeddings",
    "lib/ocr",
    "lib/parsers/documents", "lib/parsers/spreadsheets", "lib/parsers/presentations", "lib/parsers/structured", "lib/parsers/media",
    "lib/integrations/zotero"
)
$libDirs | ForEach-Object { New-Item -ItemType Directory -Path $_ -Force }
```

- [ ] **Step 3: Create all __init__.py files**

```powershell
# Create empty __init__.py files for all directories
Get-ChildItem -Path apps, lib -Directory -Recurse | ForEach-Object { New-Item -Path "$($_.FullName)/__init__.py" -ItemType File -Force }
```

- [ ] **Step 4: Verify structure**

```powershell
tree apps /F
tree lib /F
```

Expected: Directory trees showing all created folders with `__init__.py` files

- [ ] **Step 5: Commit**

```bash
git add apps/ lib/
git commit -m "chore: create apps/ and lib/ directory structure"
```

### Task 1.2: Create Django App Configuration Files

**Files:**
- Create: `aquillm/apps/chat/apps.py`
- Create: `aquillm/apps/documents/apps.py`
- Create: `aquillm/apps/collections/apps.py`
- Create: `aquillm/apps/ingestion/apps.py`
- Create: `aquillm/apps/memory/apps.py`
- Create: `aquillm/apps/platform_admin/apps.py`
- Create: `aquillm/apps/core/apps.py`
- Create: `aquillm/apps/integrations/zotero/apps.py`

- [ ] **Step 1: Create apps.py for chat**

```python
# apps/chat/apps.py
from django.apps import AppConfig

class ChatConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.chat'
    label = 'apps_chat'
```

- [ ] **Step 2: Create apps.py for documents**

```python
# apps/documents/apps.py
from django.apps import AppConfig

class DocumentsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.documents'
    label = 'apps_documents'
```

- [ ] **Step 3: Create apps.py for collections**

```python
# apps/collections/apps.py
from django.apps import AppConfig

class CollectionsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.collections'
    label = 'apps_collections'
```

- [ ] **Step 4: Create apps.py for ingestion**

```python
# apps/ingestion/apps.py
from django.apps import AppConfig

class IngestionConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.ingestion'
    label = 'apps_ingestion'
```

- [ ] **Step 5: Create apps.py for memory**

```python
# apps/memory/apps.py
from django.apps import AppConfig

class MemoryConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.memory'
    label = 'apps_memory'
```

- [ ] **Step 6: Create apps.py for platform_admin**

```python
# apps/platform_admin/apps.py
from django.apps import AppConfig

class PlatformAdminConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.platform_admin'
    label = 'apps_platform_admin'
```

- [ ] **Step 7: Create apps.py for core**

```python
# apps/core/apps.py
from django.apps import AppConfig

class CoreConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.core'
    label = 'apps_core'
```

- [ ] **Step 8: Create apps.py for zotero**

```python
# apps/integrations/zotero/apps.py
from django.apps import AppConfig

class ZoteroConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.integrations.zotero'
    label = 'apps_integrations_zotero'
```

- [ ] **Step 9: Commit**

```bash
git add apps/*/apps.py apps/integrations/*/apps.py
git commit -m "chore: add Django app configuration files"
```

### Task 1.3: Create Deployment Directory Structure

**Files:**
- Create: `deploy/docker/web/`
- Create: `deploy/docker/vllm/`
- Create: `deploy/docker/certbot/`
- Create: `deploy/compose/`
- Create: `deploy/nginx/`
- Create: `deploy/scripts/`

- [ ] **Step 1: Create deploy directories**

```powershell
Set-Location ..  # Back to repo root
$deployDirs = @(
    "deploy/docker/web", "deploy/docker/vllm", "deploy/docker/certbot",
    "deploy/compose", "deploy/nginx", "deploy/scripts"
)
$deployDirs | ForEach-Object { New-Item -ItemType Directory -Path $_ -Force }
```

- [ ] **Step 2: Verify structure**

```powershell
tree deploy /F
```

- [ ] **Step 3: Commit**

```bash
git add deploy/
git commit -m "chore: create deploy/ directory structure"
```

### Task 1.4: Create Frontend Directory Structure

**Files:**
- Create: `react/src/app/`
- Create: `react/src/features/chat/`
- Create: `react/src/features/ingestion/`
- Create: `react/src/features/collections/`
- Create: `react/src/features/documents/`
- Create: `react/src/features/admin/`
- Create: `react/src/shared/`

- [ ] **Step 1: Create frontend directories**

```powershell
Set-Location react/src
$frontendDirs = @(
    "app",
    "features/chat/components", "features/chat/hooks", "features/chat/types", "features/chat/utils",
    "features/ingestion/components/forms", "features/ingestion/hooks", "features/ingestion/types",
    "features/collections/components", "features/collections/hooks", "features/collections/types",
    "features/documents/components", "features/documents/hooks", "features/documents/types",
    "features/admin/components", "features/admin/hooks",
    "features/search/components",
    "shared/components/logos", "shared/hooks", "shared/utils", "shared/types"
)
$frontendDirs | ForEach-Object { New-Item -ItemType Directory -Path $_ -Force }
```

- [ ] **Step 2: Verify structure**

```powershell
tree features /F
tree shared /F
```

- [ ] **Step 3: Commit**

```bash
git add features/ shared/ app/
git commit -m "chore: create frontend feature directory structure"
```

---

## Chunk 2: Phase 2 - Move Models (Database-Safe)

> **Critical:** This phase moves Django models while preserving database tables using `db_table` meta option and `SeparateDatabaseAndState` migrations.

### Task 2.1: Move Collection Models to apps/collections/

**Files:**
- Create: `aquillm/apps/collections/models/collection.py`
- Create: `aquillm/apps/collections/models/permission.py`
- Create: `aquillm/apps/collections/models/__init__.py`
- Modify: `aquillm/aquillm/models.py` (remove Collection, CollectionPermission)

- [ ] **Step 1: Create collection model file**

Copy Collection and CollectionQuerySet from `aquillm/models.py` to new file:

```python
# apps/collections/models/collection.py
from django.db import models
from django.contrib.auth.models import User
from django.db.models import Q

class CollectionQuerySet(models.QuerySet):
    def accessible_by(self, user):
        from .permission import CollectionPermission
        accessible_ids = CollectionPermission.objects.filter(user=user).values_list('collection_id', flat=True)
        return self.filter(id__in=accessible_ids)

class Collection(models.Model):
    name = models.CharField(max_length=255, blank=False)
    parent = models.ForeignKey('self', on_delete=models.CASCADE, null=True, blank=True, related_name='children')
    
    objects = CollectionQuerySet.as_manager()
    
    class Meta:
        db_table = 'aquillm_collection'  # Preserve existing table name
    
    # ... copy all methods from original Collection class ...
```

- [ ] **Step 2: Create permission model file**

```python
# apps/collections/models/permission.py
from django.db import models
from django.contrib.auth.models import User

class CollectionPermission(models.Model):
    PERMISSION_CHOICES = [
        ('viewer', 'Viewer'),
        ('editor', 'Editor'),
        ('owner', 'Owner'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    collection = models.ForeignKey('collection.Collection', on_delete=models.CASCADE)
    permission_type = models.CharField(max_length=20, choices=PERMISSION_CHOICES, default='viewer')
    
    class Meta:
        db_table = 'aquillm_collectionpermission'  # Preserve existing table name
        unique_together = ['user', 'collection']
    
    # ... copy all methods from original CollectionPermission class ...
```

- [ ] **Step 3: Create models __init__.py**

```python
# apps/collections/models/__init__.py
from .collection import Collection, CollectionQuerySet
from .permission import CollectionPermission

__all__ = ['Collection', 'CollectionQuerySet', 'CollectionPermission']
```

- [ ] **Step 4: Create migration using SeparateDatabaseAndState**

```bash
cd aquillm
python manage.py makemigrations apps_collections --empty --name initial_collections
```

Then edit the migration:

```python
# apps/collections/migrations/0001_initial_collections.py
from django.db import migrations, models
import django.db.models.deletion
from django.conf import settings

class Migration(migrations.Migration):
    initial = True
    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            state_operations=[
                migrations.CreateModel(
                    name='Collection',
                    fields=[
                        ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                        ('name', models.CharField(max_length=255)),
                        ('parent', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='children', to='apps_collections.collection')),
                    ],
                    options={
                        'db_table': 'aquillm_collection',
                    },
                ),
                migrations.CreateModel(
                    name='CollectionPermission',
                    fields=[
                        ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                        ('permission_type', models.CharField(choices=[('viewer', 'Viewer'), ('editor', 'Editor'), ('owner', 'Owner')], default='viewer', max_length=20)),
                        ('collection', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='apps_collections.collection')),
                        ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
                    ],
                    options={
                        'db_table': 'aquillm_collectionpermission',
                        'unique_together': {('user', 'collection')},
                    },
                ),
            ],
            database_operations=[],  # No DB changes - tables already exist
        ),
    ]
```

- [ ] **Step 5: Update INSTALLED_APPS**

```python
# aquillm/settings.py - add to INSTALLED_APPS
'apps.collections',
```

- [ ] **Step 6: Run migration plan check**

```bash
python manage.py migrate --plan
```

Expected: Shows migration for apps_collections with no DB operations

- [ ] **Step 7: Run migration**

```bash
python manage.py migrate apps_collections
```

- [ ] **Step 8: Update imports across codebase**

Search and replace:
- `from aquillm.models import Collection` → `from apps.collections.models import Collection`
- `from aquillm.models import CollectionPermission` → `from apps.collections.models import CollectionPermission`

```bash
# Find all files that need updating
rg "from aquillm.models import.*Collection" --files-with-matches
```

- [ ] **Step 9: Run tests**

```bash
python manage.py test
```

Expected: All tests pass

- [ ] **Step 10: Commit**

```bash
git add apps/collections/ aquillm/settings.py
git commit -m "refactor: move Collection models to apps/collections/"
```

### Task 2.2: Move Document Models to apps/documents/

**Files:**
- Create: `aquillm/apps/documents/models/document.py`
- Create: `aquillm/apps/documents/models/document_types/*.py`
- Create: `aquillm/apps/documents/models/chunks.py`
- Create: `aquillm/apps/documents/models/exceptions.py`

> **Note:** This task follows the same pattern as Task 2.1. For brevity, showing key differences only.

- [ ] **Step 1: Create document base model**

Copy Document class with `db_table = 'aquillm_document'`

- [ ] **Step 2: Create document type files**

Each document type (PDFDocument, VTTDocument, etc.) goes in its own file under `document_types/`:
- `pdf.py` - `db_table = 'aquillm_pdfdocument'`
- `tex.py` - `db_table = 'aquillm_texdocument'`
- `vtt.py` - `db_table = 'aquillm_vttdocument'`
- `handwritten.py` - `db_table = 'aquillm_handwrittennotesdocument'`
- `image.py` - `db_table = 'aquillm_imageuploaddocument'`
- `media.py` - `db_table = 'aquillm_mediauploaddocument'`
- `raw_text.py` - `db_table = 'aquillm_rawtextdocument'`
- `figure.py` - `db_table = 'aquillm_documentfigure'`

- [ ] **Step 3: Create chunks model**

Copy TextChunk and TextChunkQuerySet with `db_table = 'aquillm_textchunk'`

- [ ] **Step 4: Create exceptions file**

```python
# apps/documents/models/exceptions.py
from django.core.exceptions import ValidationError

class DuplicateDocumentError(ValidationError):
    pass
```

- [ ] **Step 5: Create models __init__.py**

```python
# apps/documents/models/__init__.py
from .document import Document, DocumentChild
from .document_types.pdf import PDFDocument
from .document_types.tex import TeXDocument
from .document_types.vtt import VTTDocument
from .document_types.handwritten import HandwrittenNotesDocument
from .document_types.image import ImageUploadDocument
from .document_types.media import MediaUploadDocument
from .document_types.raw_text import RawTextDocument
from .document_types.figure import DocumentFigure
from .chunks import TextChunk, TextChunkQuerySet
from .exceptions import DuplicateDocumentError

__all__ = [
    'Document', 'DocumentChild',
    'PDFDocument', 'TeXDocument', 'VTTDocument',
    'HandwrittenNotesDocument', 'ImageUploadDocument', 'MediaUploadDocument',
    'RawTextDocument', 'DocumentFigure',
    'TextChunk', 'TextChunkQuerySet',
    'DuplicateDocumentError',
]
```

- [ ] **Step 6: Create migration with SeparateDatabaseAndState**

- [ ] **Step 7: Update INSTALLED_APPS**

- [ ] **Step 8: Run migration**

- [ ] **Step 9: Update imports across codebase**

- [ ] **Step 10: Run tests**

- [ ] **Step 11: Commit**

```bash
git commit -m "refactor: move Document models to apps/documents/"
```

### Task 2.3: Move Chat Models to apps/chat/

**Files:**
- Create: `aquillm/apps/chat/models/conversation.py` (WSConversation)
- Create: `aquillm/apps/chat/models/message.py` (Message)
- Create: `aquillm/apps/chat/models/file.py` (ConversationFile)

Follow same pattern as Tasks 2.1-2.2 with appropriate `db_table` values.

### Task 2.4: Move Ingestion Models to apps/ingestion/

**Files:**
- Create: `aquillm/apps/ingestion/models/batch.py` (IngestionBatch, IngestionBatchItem)

### Task 2.5: Move Memory Models to apps/memory/

**Files:**
- Create: `aquillm/apps/memory/models/facts.py` (UserMemoryFact)
- Create: `aquillm/apps/memory/models/episodic.py` (EpisodicMemory)

### Task 2.6: Move Admin Models to apps/platform_admin/

**Files:**
- Create: `aquillm/apps/platform_admin/models/whitelist.py` (EmailWhitelist)
- Create: `aquillm/apps/platform_admin/models/usage.py` (GeminiAPIUsage)

### Task 2.7: Move Core Models to apps/core/

**Files:**
- Create: `aquillm/apps/core/models/user_settings.py` (UserSettings)

### Task 2.8: Move Zotero Models to apps/integrations/zotero/

**Files:**
- Create: `aquillm/apps/integrations/zotero/models.py` (ZoteroConnection)

### Task 2.9: Phase 2 Verification

- [ ] **Step 1: Run full test suite**

```bash
python manage.py test
```

- [ ] **Step 2: Verify database unchanged**

```bash
python manage.py dbshell
\dt  # List tables - should be unchanged
```

- [ ] **Step 3: Commit phase checkpoint**

```bash
git commit -m "refactor: Phase 2 complete - models moved to apps/"
```

---

## Chunk 3: Phase 3 - Extract lib/ (LLM Module)

### Task 3.1: Extract LLM Types

**Files:**
- Create: `aquillm/lib/llm/types/messages.py`
- Create: `aquillm/lib/llm/types/conversation.py`
- Create: `aquillm/lib/llm/types/tools.py`
- Create: `aquillm/lib/llm/types/response.py`
- Modify: `aquillm/aquillm/llm.py` (lines 1-338 → lib/)

- [ ] **Step 1: Create messages.py**

Extract from `llm.py` lines 154-276:

```python
# lib/llm/types/messages.py
from typing import Literal, Optional, Any
from pydantic import BaseModel, Field, model_validator
from abc import ABC
import uuid

from .tools import LLMTool, ToolChoice, ToolResultDict

class __LLMMessage(BaseModel, ABC):
    role: Literal['user', 'tool', 'assistant']
    content: str
    tools: Optional[list[LLMTool]] = None
    tool_choice: Optional[ToolChoice] = None
    rating: Literal[None, 1,2,3,4,5] = None
    feedback_text: Optional[str] = None
    files: Optional[list[tuple[str, int]]] = None
    message_uuid: uuid.UUID = Field(default_factory=uuid.uuid4)
    
    @classmethod
    @model_validator(mode='after')
    def validate_tools(cls, data: Any) -> Any:
        if (data.tools and not data.tool_choice) or (data.tool_choice and not data.tools):
            raise ValueError("Both tools and tool_choice must be populated if tools are used")
        return data

    def render(self, *args, **kwargs) -> dict:
        ret = self.model_dump(*args, **kwargs)
        if self.files:
            ret['content'] = ret['content'] + "\n\nFiles:\n" + "\n".join([f'name: {file[0]}, id: {file[1]}' for file in self.files])
        return ret

class UserMessage(__LLMMessage):
    role: Literal['user'] = 'user'

class ToolMessage(__LLMMessage):
    role: Literal['tool'] = 'tool'
    tool_name: str
    arguments: Optional[dict] = None
    for_whom: Literal['assistant', 'user']
    result_dict: ToolResultDict = {}
    
    def has_images(self) -> bool:
        return bool(self.result_dict and self.result_dict.get("_images"))
    
    def get_images(self) -> list[dict]:
        if not self.result_dict:
            return []
        return self.result_dict.get("_images", [])
    
    # ... rest of ToolMessage methods ...

class AssistantMessage(__LLMMessage):
    role: Literal['assistant'] = 'assistant'
    model: Optional[str] = None
    stop_reason: str
    tool_call_id: Optional[str] = None
    tool_call_name: Optional[str] = None
    tool_call_input: Optional[dict] = None
    usage: int = 0
    
    # ... rest of AssistantMessage ...

LLM_Message = UserMessage | ToolMessage | AssistantMessage

__all__ = ['UserMessage', 'ToolMessage', 'AssistantMessage', 'LLM_Message']
```

- [ ] **Step 2: Create tools.py**

Extract LLMTool, ToolChoice, ToolResultDict from `llm.py` lines 34-152:

```python
# lib/llm/types/tools.py
from typing import Literal, Optional, Dict, List, Callable, Any
from pydantic import BaseModel, model_validator

type __ToolResultDictInner = str | int | bool | float | Dict[str, '__ToolResultDictInner' | List['__ToolResultDictInner']] | list[tuple[str, int]] | list[dict]
type ToolResultDict = Dict[Literal['exception', 'result', 'files', '_images', '_image_instruction'], __ToolResultDictInner]

class LLMTool(BaseModel):
    llm_definition: dict
    for_whom: Literal['user', 'assistant']
    _function: Callable[..., ToolResultDict]
    
    def __init__(self, **data):
        super().__init__(**data)
        self._function = data.get("_function")

    def __call__(self, *args, **kwargs):
        return self._function(*args, **kwargs)
    
    @property
    def name(self) -> str:
        return self.llm_definition['name']

class ToolChoice(BaseModel):
    type: Literal['auto', 'any', 'tool']
    name: Optional[str] = None

    @model_validator(mode='after')
    @classmethod
    def validate_name(cls, data: Any) -> Any:
        if data.type == 'tool' and data.name is None:
            raise ValueError("name is required when type is 'tool'")
        if data.type != 'tool' and data.name is not None:
            raise ValueError("name should only be set when type is 'tool'")
        return data

__all__ = ['LLMTool', 'ToolChoice', 'ToolResultDict']
```

- [ ] **Step 3: Create conversation.py**

Extract Conversation class from `llm.py` lines 278-328:

```python
# lib/llm/types/conversation.py
from pydantic import BaseModel
from .messages import LLM_Message, UserMessage, AssistantMessage, ToolMessage
from .tools import LLMTool

class Conversation(BaseModel):
    system: str
    messages: list[LLM_Message] = []

    def __len__(self):
        return len(self.messages)
    
    def __getitem__(self, index: int):
        return self.messages[index]
    
    def __iter__(self):
        return iter(self.messages)
    
    def __add__(self, other) -> 'Conversation':
        if isinstance(other, (list, Conversation)):
            return Conversation(system=self.system, messages=self.messages + list(other))
        if isinstance(other, (UserMessage, AssistantMessage, ToolMessage)):
            return Conversation(system=self.system, messages=self.messages + [other])
        return NotImplemented

    def rebind_tools(self, tools: list[LLMTool]) -> None:
        # ... copy method implementation ...
        pass

__all__ = ['Conversation']
```

- [ ] **Step 4: Create response.py**

```python
# lib/llm/types/response.py
from typing import Optional
from pydantic import BaseModel

class LLMResponse(BaseModel):
    text: Optional[str]
    tool_call: dict
    stop_reason: str
    input_usage: int
    output_usage: int
    model: Optional[str] = None

__all__ = ['LLMResponse']
```

- [ ] **Step 5: Create types __init__.py**

```python
# lib/llm/types/__init__.py
from .messages import UserMessage, ToolMessage, AssistantMessage, LLM_Message
from .tools import LLMTool, ToolChoice, ToolResultDict
from .conversation import Conversation
from .response import LLMResponse

__all__ = [
    'UserMessage', 'ToolMessage', 'AssistantMessage', 'LLM_Message',
    'LLMTool', 'ToolChoice', 'ToolResultDict',
    'Conversation',
    'LLMResponse',
]
```

- [ ] **Step 6: Run tests**

```bash
python -c "from lib.llm.types import UserMessage, Conversation, LLMTool"
```

Expected: No import errors

- [ ] **Step 7: Commit**

```bash
git add lib/llm/types/
git commit -m "refactor: extract LLM types to lib/llm/types/"
```

### Task 3.2: Extract LLM Tool Decorator

**Files:**
- Create: `aquillm/lib/llm/decorators/tool.py`

- [ ] **Step 1: Create tool.py**

Extract `@llm_tool` decorator from `llm.py` lines 52-125

- [ ] **Step 2: Commit**

```bash
git commit -m "refactor: extract @llm_tool decorator to lib/llm/decorators/"
```

### Task 3.3: Extract LLM Providers

**Files:**
- Create: `aquillm/lib/llm/providers/base.py`
- Create: `aquillm/lib/llm/providers/claude.py`
- Create: `aquillm/lib/llm/providers/openai.py`
- Create: `aquillm/lib/llm/providers/gemini.py`

- [ ] **Step 1: Create base.py with LLMInterface ABC**

Extract from `llm.py` lines 339-989

- [ ] **Step 2: Create claude.py**

Extract ClaudeInterface from `llm.py` lines 991-1055

- [ ] **Step 3: Create openai.py**

Extract OpenAIInterface from `llm.py` lines 1058-1857

- [ ] **Step 4: Create gemini.py**

Extract GeminiInterface from `llm.py` lines 1858-end

- [ ] **Step 5: Create providers __init__.py**

```python
# lib/llm/providers/__init__.py
from .base import LLMInterface
from .claude import ClaudeInterface
from .openai import OpenAIInterface
from .gemini import GeminiInterface

def get_provider(provider_name: str, **kwargs) -> LLMInterface:
    providers = {
        'claude': ClaudeInterface,
        'openai': OpenAIInterface,
        'gemini': GeminiInterface,
    }
    if provider_name not in providers:
        raise ValueError(f"Unknown provider: {provider_name}")
    return providers[provider_name](**kwargs)

__all__ = ['LLMInterface', 'ClaudeInterface', 'OpenAIInterface', 'GeminiInterface', 'get_provider']
```

- [ ] **Step 6: Commit**

```bash
git commit -m "refactor: extract LLM providers to lib/llm/providers/"
```

### Task 3.4: Create lib/llm Public API

**Files:**
- Create: `aquillm/lib/llm/__init__.py`

- [ ] **Step 1: Create __init__.py**

```python
# lib/llm/__init__.py
from .types import (
    UserMessage, ToolMessage, AssistantMessage, LLM_Message,
    LLMTool, ToolChoice, ToolResultDict,
    Conversation,
    LLMResponse,
)
from .decorators.tool import llm_tool
from .providers import LLMInterface, get_provider

__all__ = [
    'UserMessage', 'ToolMessage', 'AssistantMessage', 'LLM_Message',
    'LLMTool', 'ToolChoice', 'ToolResultDict',
    'Conversation',
    'LLMResponse',
    'llm_tool',
    'LLMInterface', 'get_provider',
]
```

- [ ] **Step 2: Update imports across codebase**

```bash
# Find files importing from aquillm.llm
rg "from aquillm.llm import" --files-with-matches
rg "from aquillm import llm" --files-with-matches
```

Replace with: `from lib.llm import ...`

- [ ] **Step 3: Run tests**

```bash
python manage.py test
```

- [ ] **Step 4: Commit**

```bash
git commit -m "refactor: complete lib/llm extraction"
```

---

## Chunk 4: Phase 3 Continued - Extract Remaining lib/ Modules

### Task 4.1: Extract lib/tools/

**Files:**
- Create: `aquillm/lib/tools/base.py`
- Create: `aquillm/lib/tools/search/vector_search.py`
- Create: `aquillm/lib/tools/documents/fetch.py`
- Create: `aquillm/lib/tools/astronomy/sky_subtraction.py`

Extract tool definitions from `chat/consumers.py` following dependency injection pattern.

### Task 4.2: Extract lib/memory/

**Files:**
- Create: `aquillm/lib/memory/base.py`
- Create: `aquillm/lib/memory/local.py`
- Create: `aquillm/lib/memory/mem0/client.py`
- Create: `aquillm/lib/memory/extraction/stable_facts.py`

Extract from `aquillm/memory.py`.

### Task 4.3: Extract lib/embeddings/

**Files:**
- Create: `aquillm/lib/embeddings/base.py`
- Create: `aquillm/lib/embeddings/local.py`
- Create: `aquillm/lib/embeddings/cohere.py`

Extract from `aquillm/utils.py`.

### Task 4.4: Extract lib/ocr/

**Files:**
- Create: `aquillm/lib/ocr/base.py`
- Create: `aquillm/lib/ocr/tesseract.py`
- Create: `aquillm/lib/ocr/qwen.py`
- Create: `aquillm/lib/ocr/gemini.py`

Extract from `aquillm/ocr_utils.py`.

### Task 4.5: Extract lib/parsers/

**Files:**
- Create: `aquillm/lib/parsers/documents/pdf.py`
- Create: `aquillm/lib/parsers/spreadsheets/xlsx.py`
- Create: `aquillm/lib/parsers/media/vtt.py`

Extract from `aquillm/ingestion/parsers.py`.

### Task 4.6: Phase 3 Verification

- [ ] **Step 1: Verify no lib/ → apps/ imports**

```bash
rg "from apps\." lib/
```

Expected: No matches (lib/ should not import from apps/)

- [ ] **Step 2: Run full test suite**

```bash
python manage.py test
```

- [ ] **Step 3: Commit phase checkpoint**

```bash
git commit -m "refactor: Phase 3 complete - lib/ extraction done"
```

---

## Chunk 5: Phase 4 - Restructure Views/Consumers

### Task 5.1: Split api_views.py into Domain Apps

Move views from `aquillm/api_views.py` to appropriate apps:
- Collection views → `apps/collections/views/api.py`
- Document views → `apps/documents/views/api.py`
- Ingestion views → `apps/ingestion/views/api.py`
- Admin views → `apps/platform_admin/views/`

### Task 5.2: Split views.py into Domain Apps

Move views from `aquillm/views.py` to appropriate apps.

### Task 5.3: Split consumers.py

Move ChatConsumer to `apps/chat/consumers/chat.py`.

### Task 5.4: Update URL Routing

Create `urls.py` for each app and update main `aquillm/urls.py`.

### Task 5.5: Phase 4 Verification

- [ ] **Step 1: Run full test suite**

- [ ] **Step 2: Verify all routes work**

```bash
python manage.py show_urls
```

- [ ] **Step 3: Commit phase checkpoint**

```bash
git commit -m "refactor: Phase 4 complete - views/consumers restructured"
```

---

## Chunk 6: Phase 5 - Deployment Restructure

### Task 6.1: Move Dockerfiles

- [ ] **Step 1: Move files**

```bash
mv Dockerfile deploy/docker/web/
mv Dockerfile.prod deploy/docker/web/
mv Dockerfile.vllm deploy/docker/vllm/
mv Dockerfile.certbot deploy/docker/certbot/
mv Dockerfile.test deploy/docker/web/
```

- [ ] **Step 2: Update docker-compose files**

Update build context paths in all compose files.

### Task 6.2: Move Compose Files

- [ ] **Step 1: Move and rename**

```bash
mv docker-compose.yml deploy/compose/base.yml
mv docker-compose-development.yml deploy/compose/development.yml
mv docker-compose-prod.yml deploy/compose/production.yml
mv docker-compose-test.yml deploy/compose/test.yml
```

- [ ] **Step 2: Create wrapper scripts at root**

```bash
# docker-compose.yml at root (for backward compatibility)
# This just extends the compose files in deploy/
```

### Task 6.3: Move Scripts

```bash
mv deployment/* deploy/scripts/
rmdir deployment
```

### Task 6.4: Update README

Update paths in README.md.

### Task 6.5: Phase 5 Verification

- [ ] **Step 1: Test Docker build**

```bash
docker compose -f deploy/compose/development.yml build
```

- [ ] **Step 2: Commit phase checkpoint**

```bash
git commit -m "refactor: Phase 5 complete - deployment restructured"
```

---

## Chunk 7: Phase 5.5 - Test Migration

### Task 7.1: Move Unit Tests to Domain Apps

**Files per spec (docs/superpowers/specs/2026-03-18-codebase-refactor-design.md lines 683-707):**

| Current File | New Location |
|--------------|--------------|
| `aquillm/tests/test_figure_extraction.py` | `apps/ingestion/tests/` |
| `aquillm/tests/test_unified_ingestion_*.py` | `apps/ingestion/tests/` |
| `aquillm/tests/test_ingestion_monitor_includes_non_pdf.py` | `apps/ingestion/tests/` |
| `aquillm/tests/test_multimodal_*.py` | `apps/documents/tests/` |
| `aquillm/tests/test_embedding_*.py` | `lib/embeddings/tests/` |
| `aquillm/tests/test_ocr_*.py` | `lib/ocr/tests/` |
| `aquillm/tests/test_transcribe_*.py` | `lib/parsers/tests/` |
| `aquillm/tests/test_mem0_*.py` | `lib/memory/tests/` |
| `aquillm/tests/test_llm_*.py` | `lib/llm/tests/` |
| `aquillm/tests/models_test.py` | `apps/documents/tests/` |
| `chat/tests.py` | `apps/chat/tests/test_chat.py` |
| `ingest/tests.py` | `apps/ingestion/tests/test_ingest.py` |

- [ ] **Step 1: Create test directories in lib/**

```powershell
$testDirs = @(
    "lib/llm/tests", "lib/memory/tests", "lib/embeddings/tests",
    "lib/ocr/tests", "lib/parsers/tests"
)
$testDirs | ForEach-Object { New-Item -ItemType Directory -Path $_ -Force }
$testDirs | ForEach-Object { New-Item -ItemType File -Path "$_/__init__.py" -Force }
```

- [ ] **Step 2: Move ingestion tests**

```powershell
Move-Item aquillm/aquillm/tests/test_figure_extraction.py apps/ingestion/tests/
Move-Item aquillm/aquillm/tests/test_unified_ingestion_api.py apps/ingestion/tests/
Move-Item aquillm/aquillm/tests/test_unified_ingestion_parsers.py apps/ingestion/tests/
Move-Item aquillm/aquillm/tests/test_ingestion_monitor_includes_non_pdf.py apps/ingestion/tests/
```

- [ ] **Step 3: Move document tests**

```powershell
Move-Item aquillm/aquillm/tests/test_multimodal_chunk_position_uniqueness.py apps/documents/tests/
Move-Item aquillm/aquillm/tests/test_multimodal_ingestion_media_storage.py apps/documents/tests/
Move-Item aquillm/aquillm/tests/models_test.py apps/documents/tests/
```

- [ ] **Step 4: Move lib tests**

```powershell
Move-Item aquillm/aquillm/tests/test_embedding_context_limit_handling.py lib/embeddings/tests/
Move-Item aquillm/aquillm/tests/test_ocr_provider_selection.py lib/ocr/tests/
Move-Item aquillm/aquillm/tests/test_transcribe_provider_selection.py lib/parsers/tests/
Move-Item aquillm/aquillm/tests/test_mem0_oss_mode.py lib/memory/tests/
Move-Item aquillm/aquillm/tests/test_llm_tool_choice_serialization.py lib/llm/tests/
```

- [ ] **Step 5: Move app-level test files**

```powershell
Move-Item aquillm/chat/tests.py apps/chat/tests/test_chat.py
Move-Item aquillm/ingest/tests.py apps/ingestion/tests/test_ingest.py
```

- [ ] **Step 6: Update test imports**

Update import paths in all moved test files:
- `from aquillm.models import` → `from apps.documents.models import` (etc.)
- `from aquillm.llm import` → `from lib.llm import`

- [ ] **Step 7: Run tests to verify**

```powershell
python manage.py test
```

Expected: All tests pass

- [ ] **Step 8: Commit**

```bash
git add apps/*/tests/ lib/*/tests/
git commit -m "refactor: move unit tests to domain locations"
```

### Task 7.2: Move Integration Tests

**Files:**
- Create: `tests/integration/`
- Move: Deployment and compose tests

- [ ] **Step 1: Create integration test directory**

```powershell
New-Item -ItemType Directory -Path tests/integration -Force
New-Item -ItemType File -Path tests/__init__.py -Force
New-Item -ItemType File -Path tests/integration/__init__.py -Force
```

- [ ] **Step 2: Move integration tests**

```powershell
Move-Item aquillm/aquillm/tests/test_compose_multimodal_services.py tests/integration/
Move-Item aquillm/aquillm/tests/test_dev_launch_script.py tests/integration/
Move-Item aquillm/aquillm/tests/test_deployment_run_script.py tests/integration/
```

- [ ] **Step 3: Update pytest.ini**

```ini
# pytest.ini
[pytest]
DJANGO_SETTINGS_MODULE = aquillm.settings
python_files = tests.py test_*.py *_tests.py
testpaths = apps lib tests
```

- [ ] **Step 4: Run full test suite**

```powershell
pytest apps/ lib/ tests/
```

- [ ] **Step 5: Commit**

```bash
git add tests/ pytest.ini
git commit -m "refactor: move integration tests to tests/"
```

---

## Chunk 8: Phase 6 - Frontend Restructure

### Task 7.1: Split ChatComponent.tsx

**Files:**
- Create: `react/src/features/chat/components/Chat.tsx`
- Create: `react/src/features/chat/components/MessageBubble.tsx`
- Create: `react/src/features/chat/components/ToolCallGroup.tsx`
- Create: `react/src/features/chat/hooks/useChatWebSocket.ts`

### Task 7.2: Split IngestRow.tsx

**Files:**
- Create: `react/src/features/ingestion/components/IngestRowsContainer.tsx`
- Create: `react/src/features/ingestion/components/forms/UploadsForm.tsx`

### Task 7.3: Split Remaining Components

Split CollectionView.tsx, FileSystemViewer.tsx, UserManagementModal.tsx.

### Task 7.4: Extract Shared Components

Move reusable components to `shared/components/`.

### Task 7.5: Phase 6 Verification

- [ ] **Step 1: Run frontend build**

```bash
cd react && npm run build
```

- [ ] **Step 2: Commit phase checkpoint**

```bash
git commit -m "refactor: Phase 6 complete - frontend restructured"
```

---

## Chunk 9: Phase 7 - Cleanup

### Task 8.1: Remove Old Files

Delete now-empty original files after verifying everything works.

### Task 8.2: Final Test Suite

- [ ] **Step 1: Backend tests**

```bash
python manage.py test
```

- [ ] **Step 2: Frontend tests**

```bash
cd react && npm test
```

- [ ] **Step 3: Docker build**

```bash
docker compose -f deploy/compose/development.yml up --build
```

### Task 8.3: Update Documentation

Update README.md and any other docs with new paths.

### Task 8.4: Final Commit

```bash
git commit -m "refactor: Phase 7 complete - codebase refactor finished"
```

---

## Execution Notes

**Estimated Tasks:** ~55 main tasks across 9 chunks
**Estimated Time:** 4-8 hours of focused work
**Critical Path:** Phase 2 (migrations) is the riskiest

**Validation Strategy:**
- **Local (after each phase):** Python syntax, linting, imports, `manage.py check`
- **Remote (final push):** Full Docker build, vLLM deployment, integration tests

Local validation catches most issues. Remote build confirms full system works.

**Import Path Note:**
Within the Django project, imports should use full paths:
- `from aquillm.lib.llm import ...` (not `from lib.llm import ...`)
- Alternatively, add `aquillm/` to Python path and use `from lib.llm import ...`

**Pre-Push Verification Checklist:**
Before pushing to trigger remote build:
- [ ] All `__init__.py` files exist in new directories
- [ ] All import statements updated (search for old paths)
- [ ] All `db_table` meta options match existing table names
- [ ] All `apps.py` files have correct `name` and `label`
- [ ] `INSTALLED_APPS` includes all new apps
- [ ] URL routing updated
- [ ] No circular imports (lib/ does not import from apps/)
- [ ] Docker build context paths updated in compose files
- [ ] Frontend imports updated

**Local Verification Commands (run after each phase):**
```powershell
# Check for old import paths that need updating
rg "from aquillm\.models import" --files-with-matches
rg "from aquillm\.llm import" --files-with-matches
rg "from aquillm\.memory import" --files-with-matches
rg "from aquillm\.utils import" --files-with-matches

# Check Python syntax
Get-ChildItem -Path aquillm/apps, aquillm/lib -Filter "*.py" -Recurse | ForEach-Object { python -m py_compile $_.FullName }

# Check for missing __init__.py
Get-ChildItem -Path aquillm/apps, aquillm/lib -Directory -Recurse | Where-Object { -not (Test-Path "$($_.FullName)/__init__.py") }

# Verify Django configuration (catches import errors, bad app configs)
cd aquillm
python manage.py check
python manage.py makemigrations --dry-run  # Verify migration state

# Lint (if available)
ruff check aquillm/apps aquillm/lib
```

**Remote Verification (final push):**
- Docker build
- vLLM service startup  
- Integration tests
- Full application smoke test

**Rollback Points:**
- Each phase has a checkpoint commit
- Can revert to any phase boundary if remote build fails
- Phase 2 migrations use `SeparateDatabaseAndState` to avoid DB changes
