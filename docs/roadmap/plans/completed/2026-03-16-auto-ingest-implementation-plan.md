# Unified Document Ingestion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement source-agnostic ingestion so users can upload many mixed documents (including folder-as-zip) without choosing types, and extend ingestion to Zotero + arXiv via the same pipeline.

**Architecture:** Add a unified ingestion service + type detector + parser registry. Expose a single multipart API endpoint (`/api/ingest_files/`) used by a new â€œUpload documentsâ€ modal. Refactor Zotero sync/manual import and arXiv ingestion to call the same service. Persist new uploads into new generic `DocumentV2`/`DocumentAsset` models while keeping legacy models readable.

**Tech Stack:** Django, Celery, Channels/websockets, `pypdf`, Pillow, `chardet`, existing `aquillm/aquillm/vtt.py`, existing `aquillm/aquillm/ocr_utils.py`, pytest/pytest-django.

---

## Conventions used in this plan

- **New ingestion models** are named `DocumentV2`/`DocumentAsset` to avoid colliding with the existing abstract `Document` base in `aquillm/aquillm/models.py`.
- **Unified ingestion code** lives under `aquillm/aquillm/ingestion/` (new package).
- **Endpoint** is added to `aquillm/aquillm/api_views.py` and included in URL patterns there.
- **UI** is implemented in Django templates (existing pattern) and invoked from Collections page.

### Caveats and edge cases (see design doc)

Full behavior is in **`docs/specs/2026-03-16-auto-ingest-design.md`** under **â€œCaveats and edge casesâ€**. Summary for implementation:

- **Per-item status values:** `accepted` | `skipped_no_file` | `skipped_duplicate` | `skipped_unsupported_type` | `error`. Use these consistently in API responses.
- **Zotero:** Handle `skipped_no_file` when `download_file` returns nothing (link-only items); `skipped_duplicate` when a document already exists for that attachment key. Report counts: ingested, skipped (no file), skipped (duplicate).
- **Uploads / zip:** Handle `skipped_unsupported_type`, empty file, ZipSlip, zip size limits, parser failures; fail per file, not whole batch.
- **arXiv:** Handle 404, no PDF/TeX, and `skipped_duplicate` by `source_id`.
- **General:** 403 when user cannot edit collection; 400/404 for invalid collection; partial batch success with per-item `status` and `error`.

## Task 1: Add `DocumentV2` + `DocumentAsset` models

**Files:**
- Modify: `aquillm/aquillm/models.py`
- Create: `aquillm/aquillm/migrations/00xx_documentv2_and_assets.py`
- Test: `aquillm/ingest/tests.py` (or create `aquillm/aquillm/tests/test_documentv2_models.py`)

**Step 1: Write failing model tests**

```python
import uuid
import pytest
from django.contrib.auth.models import User
from aquillm.models import Collection, CollectionPermission

@pytest.mark.django_db
def test_documentv2_unique_by_source_provider_and_id():
    user = User.objects.create_user(username="u", password="p")
    col = Collection.objects.create(name="c")
    CollectionPermission.objects.create(user=user, collection=col, permission="MANAGE")

    from aquillm.models import DocumentV2
    d1 = DocumentV2.objects.create(
        title="t1",
        full_text="hello",
        collection=col,
        ingested_by=user,
        doc_type="text",
        source_provider="zotero",
        source_id="ABCD1234",
    )
    with pytest.raises(Exception):
        DocumentV2.objects.create(
            title="t2",
            full_text="hello2",
            collection=col,
            ingested_by=user,
            doc_type="text",
            source_provider="zotero",
            source_id="ABCD1234",
        )
```

**Step 2: Run test to verify it fails**

Run:
- `pytest -q`

Expected: failure because `DocumentV2` does not exist yet.

**Step 3: Implement models**

Implement:
- `DocumentV2` fields:
  - `id` UUID primary identifier (can reuse current abstract style: `pkid` BigAutoField + `id` UUIDField indexed)
  - `title`, `full_text`, `full_text_hash`
  - `doc_type` choices: `pdf`, `vtt`, `image_ocr`, `text`, `html`, `tex`, `unknown`, `zip_container`
  - `collection`, `ingested_by`, timestamps, `ingestion_complete`
  - provenance: `source_provider` (nullable), `source_id` (nullable), `source_metadata` JSON (default `{}`)
  - DB constraint: unique `(source_provider, source_id)` when both non-null (partial unique constraint)
- `DocumentAsset`:
  - FK to `DocumentV2`
  - `role` choices: `original`, `audio`, `other`
  - `file` FileField (upload_to `document_assets/`), `mime`, `size_bytes`, `client_path` (for zip path display)

**Step 4: Add migration**

Run:
- `python aquillm/manage.py makemigrations`

Verify migration creates new tables + constraints.

**Step 5: Run tests**

Run:
- `pytest -q`

Expected: PASS.

**Step 6: Commit**

```bash
git add aquillm/aquillm/models.py aquillm/aquillm/migrations aquillm/**/test*.py
git commit -m "feat: add generic DocumentV2 + DocumentAsset models"
```

## Task 2: Add ingestion types + detector

**Files:**
- Create: `aquillm/aquillm/ingestion/__init__.py`
- Create: `aquillm/aquillm/ingestion/types.py`
- Create: `aquillm/aquillm/ingestion/detect.py`
- Test: `aquillm/aquillm/tests/test_detect.py`

**Step 1: Write failing tests for detection**

```python
import pytest
from aquillm.ingestion.detect import detect_bytes

def test_detect_pdf_header():
    result = detect_bytes(b"%PDF-1.7\\n...")
    assert result.doc_type == "pdf"

def test_detect_vtt_header():
    result = detect_bytes(b"WEBVTT\\n\\n00:00:00.000 --> 00:00:01.000\\nHello\\n")
    assert result.doc_type == "vtt"

def test_detect_zip_header():
    result = detect_bytes(b"PK\\x03\\x04....")
    assert result.doc_type == "zip"
```

**Step 2: Run tests**

Run: `pytest aquillm/aquillm/tests/test_detect.py -q`

Expected: FAIL (module missing).

**Step 3: Implement detection**

Implement:
- `DetectionResult(doc_type, mime, ext, confidence, is_container)`
- `detect_uploaded_file(uploaded_file)` that peeks a small prefix (without consuming the file permanently)
- `detect_bytes(data)` for tests
- Heuristics from the design doc: `%PDF`, `WEBVTT`, `PK\\x03\\x04`, image via Pillow open, otherwise text via decode+`chardet`.

**Step 4: Run tests**

Expected: PASS.

**Step 5: Commit**

```bash
git add aquillm/aquillm/ingestion aquillm/aquillm/tests/test_detect.py
git commit -m "feat: add file-type detection for unified ingestion"
```

## Task 3: Implement parser registry + initial parsers

**Files:**
- Create: `aquillm/aquillm/ingestion/parsers/__init__.py`
- Create: `aquillm/aquillm/ingestion/parsers/base.py`
- Create: `aquillm/aquillm/ingestion/parsers/pdf.py`
- Create: `aquillm/aquillm/ingestion/parsers/vtt.py`
- Create: `aquillm/aquillm/ingestion/parsers/image_ocr.py`
- Create: `aquillm/aquillm/ingestion/parsers/text.py`
- (Optional) Create: `aquillm/aquillm/ingestion/parsers/html.py`
- Test: `aquillm/aquillm/tests/test_parsers_smoke.py`

**Step 1: Write failing smoke tests**

```python
import pytest
from aquillm.ingestion.parsers import get_parser_for

def test_parser_registry_has_pdf():
    parser = get_parser_for("pdf")
    assert parser is not None
```

**Step 2: Run tests (fail)**

Run: `pytest aquillm/aquillm/tests/test_parsers_smoke.py -q`

**Step 3: Implement parsers**

- PDF: reuse `pypdf.PdfReader` extraction logic currently in `PDFDocument.extract_text()`.\n+- VTT: reuse `aquillm/aquillm/vtt.py` `parse â†’ coalesce â†’ to_text`.\n+- Image OCR: call `extract_text_from_image` from `aquillm/aquillm/ocr_utils.py` with option `ocr_convert_to_latex`.\n+- Text: decode bytes to string; store.\n+- HTML optional: use `trafilatura` for extraction if present, else BeautifulSoup fallback.\n+
Return a consistent `ParseResult(full_text, metadata, assets)`.\n+
**Step 4: Run tests**
\n+Run: `pytest -q`\n+Expected: PASS.\n+
**Step 5: Commit**
\n+```bash\n+git add aquillm/aquillm/ingestion aquillm/aquillm/tests/test_parsers_smoke.py\n+git commit -m \"feat: add ingestion parser registry and core parsers\"\n+```\n+\n+## Task 4: Implement zip expansion with safety limits\n+\n+**Files:**\n+- Create: `aquillm/aquillm/ingestion/zip_expand.py`\n+- Modify: `aquillm/aquillm/ingestion/service.py` (created in Task 5)\n+- Test: `aquillm/aquillm/tests/test_zip_expand.py`\n+\n+**Step 1: Write failing tests**\n+\n+```python\n+import io\n+import zipfile\n+import pytest\n+from aquillm.ingestion.zip_expand import expand_zip_bytes\n+\n+def make_zip(files: dict[str, bytes]) -> bytes:\n+    buf = io.BytesIO()\n+    with zipfile.ZipFile(buf, \"w\") as z:\n+        for name, content in files.items():\n+            z.writestr(name, content)\n+    return buf.getvalue()\n+\n+def test_zip_slip_rejected():\n+    data = make_zip({\"../evil.txt\": b\"no\"})\n+    with pytest.raises(ValueError):\n+        list(expand_zip_bytes(data))\n+\n+def test_zip_skips_ds_store():\n+    data = make_zip({\".DS_Store\": b\"x\", \"ok.txt\": b\"hi\"})\n+    entries = list(expand_zip_bytes(data))\n+    assert len(entries) == 1\n+    assert entries[0].client_path == \"ok.txt\"\n+```\n+\n+**Step 2: Run tests (fail)**\n+\n+Run: `pytest aquillm/aquillm/tests/test_zip_expand.py -q`\n+\n+**Step 3: Implement**\n+\n+Implement `expand_zip_bytes` that yields `ZipEntryInput` with:\n+- ZipSlip checks\n+- entry filtering\n+- limits via env vars (reasonable defaults):\n+  - `APP_INGEST_ZIP_MAX_FILES` (e.g. 1000)\n+  - `APP_INGEST_ZIP_MAX_TOTAL_BYTES` (e.g. 500MB)\n+  - `APP_INGEST_ZIP_MAX_FILE_BYTES` (e.g. 50MB)\n+\n+**Step 4: Run tests**\n+\n+Expected: PASS.\n+\n+**Step 5: Commit**\n+\n+```bash\n+git add aquillm/aquillm/ingestion/zip_expand.py aquillm/aquillm/tests/test_zip_expand.py\n+git commit -m \"feat: expand zip uploads with safety limits\"\n+```\n+\n+## Task 5: UnifiedIngestService (persist + enqueue chunking)\n+\n+**Files:**\n+- Create: `aquillm/aquillm/ingestion/service.py`\n+- Modify: `aquillm/aquillm/models.py` (if needed to expose chunk enqueue helper)\n+- Test: `aquillm/aquillm/tests/test_ingest_service.py`\n+\n+**Step 1: Write failing test (minimal end-to-end)**\n+\n+```python\n+import pytest\n+from django.contrib.auth.models import User\n+from aquillm.models import Collection, CollectionPermission, DocumentV2\n+from aquillm.ingestion.service import ingest_bytes\n+\n+@pytest.mark.django_db\n+def test_ingest_bytes_creates_documentv2(monkeypatch):\n+    user = User.objects.create_user(username=\"u\", password=\"p\")\n+    col = Collection.objects.create(name=\"c\")\n+    CollectionPermission.objects.create(user=user, collection=col, permission=\"MANAGE\")\n+\n+    # avoid running celery in test: stub enqueue\n+    monkeypatch.setattr(\"aquillm.ingestion.service.enqueue_chunking\", lambda doc_id: None)\n+\n+    result = ingest_bytes(\n+        data=b\"hello\",\n+        display_name=\"hello.txt\",\n+        collection=col,\n+        user=user,\n+        options={},\n+        source_provider=\"upload\",\n+        source_id=\"test-1\",\n+    )\n+    assert result.document_id\n+    doc = DocumentV2.objects.get(id=result.document_id)\n+    assert \"hello\" in doc.full_text\n+```\n+\n+**Step 2: Run test (fail)**\n+\n+Run: `pytest aquillm/aquillm/tests/test_ingest_service.py -q`\n+\n+**Step 3: Implement service**\n+\n+Implement:\n+- `ingest_inputs(inputs, collection, user, options) -> list[IngestResult]`\n+- `ingest_uploaded_files(files, ...)` convenience for request.FILES\n+- `ingest_bytes(...)` for tests and Zotero/arXiv\n+- Persistence:\n+  - create `DocumentV2`\n+  - save `DocumentAsset(role=\"original\")` where applicable\n+  - compute `full_text_hash`\n+- Enqueue chunking:\n+  - call existing Celery `create_chunks.delay(str(doc.id))` (reuse)\n+\n+**Step 4: Run tests**\n+\n+Expected: PASS.\n+\n+**Step 5: Commit**\n+\n+```bash\n+git add aquillm/aquillm/ingestion/service.py aquillm/aquillm/tests/test_ingest_service.py\n+git commit -m \"feat: add unified ingest service for DocumentV2\"\n+```\n+\n+## Task 6: Add `/api/ingest_files/` endpoint (multipart, multi-file, zip)\n+\n+**Files:**\n+- Modify: `aquillm/aquillm/api_views.py`\n+- Test: `aquillm/aquillm/tests/test_api_ingest_files.py`\n+\n+**Step 1: Write failing API test**\n+\n+```python\n+import io\n+import zipfile\n+import pytest\n+from django.core.files.uploadedfile import SimpleUploadedFile\n+\n+@pytest.mark.django_db\n+def test_api_ingest_files_accepts_multiple(client, django_user_model):\n+    user = django_user_model.objects.create_user(username=\"u\", password=\"p\")\n+    client.force_login(user)\n+\n+    from aquillm.models import Collection, CollectionPermission\n+    col = Collection.objects.create(name=\"c\")\n+    CollectionPermission.objects.create(user=user, collection=col, permission=\"MANAGE\")\n+\n+    f1 = SimpleUploadedFile(\"a.txt\", b\"hello\", content_type=\"text/plain\")\n+    f2 = SimpleUploadedFile(\"b.vtt\", b\"WEBVTT\\n\\n00:00:00.000 --> 00:00:01.000\\nHi\\n\", content_type=\"text/vtt\")\n+\n+    resp = client.post(\n+        \"/aquillm/api/ingest_files/\",\n+        data={\"collection_id\": str(col.id), \"files\": [f1, f2]},\n+    )\n+    assert resp.status_code in (200, 202)\n+    body = resp.json()\n+    assert len(body[\"results\"]) == 2\n+```\n+\n+**Step 2: Run failing test**\n+\n+Run: `pytest aquillm/aquillm/tests/test_api_ingest_files.py -q`\n+\n+**Step 3: Implement endpoint**\n+\n+Add view:\n+- validate collection + permissions\n+- accept `request.FILES.getlist('files')`\n+- parse `options` JSON if present\n+- call unified ingest service\n+- return results JSON\n+\n+**Step 4: Run tests**\n+\n+Expected: PASS.\n+\n+**Step 5: Commit**\n+\n+```bash\n+git add aquillm/aquillm/api_views.py aquillm/aquillm/tests/test_api_ingest_files.py\n+git commit -m \"feat: add unified multi-file ingest API endpoint\"\n+```\n+\n+## Task 7: Collections UI â€” new â€œUpload documentsâ€ modal\n+\n+**Files:**\n+- Create: `aquillm/templates/aquillm/ingest_files_modal.html`\n+- Modify: `aquillm/templates/aquillm/collection.html`\n+- (Optional) Modify: `aquillm/aquillm/context_processors.py` if it controls modal routes\n+- Modify: `aquillm/aquillm/views.py` (add a view to render the modal if needed)\n+\n+**Step 1: Add modal template**\n+\n+- `<input type=\"file\" multiple>`\n+- `<input type=\"file\" accept=\".zip\" ...>` or allow zip within same input\n+- checkbox: `ocr_convert_to_latex`\n+- JS:\n+  - assemble `FormData` with `files[]`, `collection_id`, `options`\n+  - call `/aquillm/api/ingest_files/`\n+  - render result list and link to document pages\n+\n+**Step 2: Wire modal entrypoint on collection page**\n+\n+- Add an â€œUpload documentsâ€ button next to existing ingest actions\n+\n+**Step 3: Manual smoke**\n+\n+- Upload 2 files mixed types\n+- Upload a zip with nested dirs\n+- Verify results returned and ingestion dashboard shows progress\n+\n+**Step 4: Commit**\n+\n+```bash\n+git add aquillm/templates/aquillm/ingest_files_modal.html aquillm/templates/aquillm/collection.html aquillm/aquillm/views.py\n+git commit -m \"feat: add collections multi-file upload modal\"\n+```\n+\n+## Task 8: Zotero manual import (new) using unified ingest\n+\n+**Files:**\n+- Modify: `aquillm/aquillm/zotero_views.py` (add manual import view + form)\n+- Modify: `aquillm/aquillm/api_views.py` (optional API endpoint)\n+- Modify: `aquillm/templates/zotero/settings.html` (add manual import UI) OR add modal in collections\n+\n+**Step 1: Implement a minimal API endpoint**\n+\n+- `POST /api/ingest_zotero_item/` with `collection_id`, `zotero_item_key_or_url`\n+- Parse key from URL\n+- Use `ZoteroAPIClient.get_item_children` + `download_file` to get attachments\n+- For each downloaded attachment, call `ingest_bytes(... source_provider=\"zotero\", source_id=attachment_key ...)`\n+\n+**Step 2: Add test for key parsing + ingestion call**\n+\n+- Stub Zotero client network calls via monkeypatch\n+\n+**Step 3: Commit**\n+\n+```bash\n+git add aquillm/aquillm/api_views.py aquillm/aquillm/tests/test_api_ingest_zotero.py\n+git commit -m \"feat: add Zotero manual import via unified ingest\"\n+```\n+\n+## Task 9: Refactor Zotero sync task to create `DocumentV2`\n+\n+**Files:**\n+- Modify: `aquillm/aquillm/zotero_tasks.py`\n+\n+**Step 1: Replace PDFDocument creation**\n+\n+- Where code does `PDFDocument(...).save()`, replace with unified ingest service call\n+- Preserve dedupe behavior:\n+  - before download, check `DocumentV2` exists with `(source_provider=\"zotero\", source_id=attachment_key)`\n+\n+**Step 2: Add regression test**\n+\n+- monkeypatch `ZoteroAPIClient` methods to return known items + pdf bytes\n+- verify `DocumentV2` rows created and no duplicates on second run\n+\n+**Step 3: Commit**\n+\n+```bash\n+git add aquillm/aquillm/zotero_tasks.py aquillm/aquillm/tests/test_zotero_sync_documentv2.py\n+git commit -m \"refactor: Zotero sync uses unified ingest pipeline\"\n+```\n+\n+## Task 10: Refactor arXiv ingestion to unified ingest\n+\n+**Files:**\n+- Modify: `aquillm/aquillm/api_views.py`\n+\n+**Step 1: Refactor `insert_one_from_arxiv`**\n+\n+- After fetching PDF bytes or TeX text:\n+  - call unified ingest with `source_provider=\"arxiv\"`, `source_id=f\"{arxiv_id}:pdf\"` or `:tex`\n+- Return results compatible with existing API contract\n+\n+**Step 2: Add tests**\n+\n+- monkeypatch `requests.get` to return PDF bytes\n+- assert `DocumentV2` created with correct source id\n+\n+**Step 3: Commit**\n+\n+```bash\n+git add aquillm/aquillm/api_views.py aquillm/aquillm/tests/test_api_ingest_arxiv_v2.py\n+git commit -m \"refactor: arXiv ingestion uses unified ingest pipeline\"\n+```\n+\n+## Task 11: Compatibility in listing/search (minimal v1)\n+\n+**Files:**\n+- Modify: `aquillm/aquillm/api_views.py` collection listing endpoints\n+- Modify: `aquillm/aquillm/views.py` collection page context (if it lists docs)\n+\n+**Step 1: Include DocumentV2 in collection documents list**\n+\n+- Wherever documents are gathered from `DESCENDED_FROM_DOCUMENT`, also include `DocumentV2.objects.filter(collection=collection)`\n+- Add `type` to response: `\"DocumentV2\"` or `doc_type`\n+\n+**Step 2: Commit**\n+\n+```bash\n+git add aquillm/aquillm/api_views.py aquillm/aquillm/views.py\n+git commit -m \"feat: include DocumentV2 in collection listings\"\n+```\n+\n+## Task 12: End-to-end verification\n+\n+**Step 1: Run unit tests**\n+\n+Run:\n+- `pytest -q`\n+\n+Expected: PASS.\n+\n+**Step 2: Manual smoke checklist**\n+\n+- Upload multi-file (txt + vtt)\n+- Upload zip with nested directories\n+- Ensure per-file results returned\n+- Confirm chunking progresses (websocket monitor)\n+- Trigger Zotero sync and confirm docs appear as `DocumentV2`\n+- Import arXiv and confirm docs appear as `DocumentV2`\n+\n+**Step 3: Final commit(s) if fixes needed**\n+\n+## Execution handoff\n+\n+Plan complete and saved to `docs/roadmap/plans/completed/2026-03-16-auto-ingest-implementation-plan.md`.\n+\n+Two execution options:\n+\n+1. **Subagent-Driven (this session)** â€” I dispatch a fresh subagent per task, review between tasks, fast iteration.\n+2. **Parallel Session (separate)** â€” Open a new session in a worktree and execute with `superpowers:executing-plans`.\n+\n+Which approach?\n+



