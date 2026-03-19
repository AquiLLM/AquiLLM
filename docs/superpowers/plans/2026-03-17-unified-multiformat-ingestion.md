# Unified Multi-Format Ingestion Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single multi-file upload ingestion pipeline that auto-detects and parses broad document, image, audio, video, and data formats, then feeds extracted text into AquiLLM's existing chunk/embed/rerank flow with concurrent background processing.

**Architecture:** Keep the existing `Document` model family and chunking pipeline as the indexing core. Add a parser/strategy registry to normalize many file types into extracted text, and add a unified API endpoint that enqueues one Celery task per uploaded file for concurrency. Preserve existing `arXiv` and webpage APIs as separate import paths.

**Tech Stack:** Django, Celery, PostgreSQL/pgvector, existing AquiLLM models/views/api_views, React + Vite frontend, Python parsing libraries per file format

---

## Chunk 1: Backend Parsing Foundation

### Task 1: Add parser registry and extraction contract

**Files:**
- Create: `aquillm/aquillm/ingestion/parsers.py`
- Create: `aquillm/aquillm/ingestion/types.py`
- Create: `aquillm/aquillm/tests/test_unified_ingestion_parsers.py`

- [ ] **Step 1: Write failing tests for parser routing and support matrix** (@superpowers:test-driven-development)

```python
def test_detect_type_for_known_extensions():
    assert detect_ingest_type("paper.pdf", "application/pdf") == "pdf"
    assert detect_ingest_type("sheet.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet") == "xlsx"
    assert detect_ingest_type("clip.mp4", "video/mp4") == "video"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q aquillm/aquillm/tests/test_unified_ingestion_parsers.py::test_detect_type_for_known_extensions`
Expected: FAIL due to missing parser module/functions.

- [ ] **Step 3: Implement parser registry and normalized extraction type**

```python
SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".xlsx": "xlsx",
    ".csv": "csv",
    ".mp4": "video",
    ".mp3": "audio",
}
```

- [ ] **Step 4: Run parser tests to verify pass**

Run: `python -m pytest -q aquillm/aquillm/tests/test_unified_ingestion_parsers.py`
Expected: PASS.

- [ ] **Step 5: Commit parser routing base**

```bash
git add aquillm/aquillm/ingestion/types.py aquillm/aquillm/ingestion/parsers.py aquillm/aquillm/tests/test_unified_ingestion_parsers.py
git commit -m "feat: add unified ingestion parser registry and type detection"
```

### Task 2: Implement extraction strategies for text-like and office/data formats

**Files:**
- Modify: `aquillm/aquillm/ingestion/parsers.py`
- Create: `aquillm/aquillm/tests/test_unified_ingestion_extractors.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Write failing extractor tests for core formats**

```python
def test_csv_extracts_text_rows(tmp_path):
    file_path = tmp_path / "data.csv"
    file_path.write_text("a,b\\n1,2\\n", encoding="utf-8")
    result = extract_text_for_file(file_path, "text/csv")
    assert "a,b" in result.full_text
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest -q aquillm/aquillm/tests/test_unified_ingestion_extractors.py`
Expected: FAIL on unimplemented extractors.

- [ ] **Step 3: Implement minimal extractors**

Implement per format family:
- plaintext/markdown/html/json/xml/yaml: decode and normalize text
- csv/tsv/xls/xlsx/ods: table-to-text conversion
- doc/docx/odt/rtf/ppt/pptx/odp/epub: text extraction adapters
- vtt/srt: caption parsing to plain text

- [ ] **Step 4: Add required parsing dependencies**

Update `requirements.txt` for selected parser libs used by implementation.

- [ ] **Step 5: Run extractor tests**

Run: `python -m pytest -q aquillm/aquillm/tests/test_unified_ingestion_extractors.py`
Expected: PASS.

- [ ] **Step 6: Commit extractor support**

```bash
git add requirements.txt aquillm/aquillm/ingestion/parsers.py aquillm/aquillm/tests/test_unified_ingestion_extractors.py
git commit -m "feat: add text, office, and data extractors for unified ingestion"
```

### Task 3: Add image OCR, audio transcription, video transcription, and zip expansion adapters

**Files:**
- Modify: `aquillm/aquillm/ingestion/parsers.py`
- Create: `aquillm/aquillm/ingestion/media.py`
- Create: `aquillm/aquillm/tests/test_unified_ingestion_media.py`

- [ ] **Step 1: Write failing tests for OCR/transcription dispatch behavior**

```python
def test_image_file_routes_to_ocr_extractor():
    assert resolve_strategy(".png") == "image_ocr"

def test_video_file_routes_to_video_transcription():
    assert resolve_strategy(".mp4") == "video_transcription"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest -q aquillm/aquillm/tests/test_unified_ingestion_media.py`
Expected: FAIL before media adapters exist.

- [ ] **Step 3: Implement media adapters with provider abstraction**

Add `TranscriptionProvider` abstraction and implement default adapter path for audio/video transcription and OCR routing.

- [ ] **Step 4: Implement zip expansion guardrails**

Add:
- max expanded files
- max expanded bytes
- recurse only supported extensions

- [ ] **Step 5: Run media tests**

Run: `python -m pytest -q aquillm/aquillm/tests/test_unified_ingestion_media.py`
Expected: PASS.

- [ ] **Step 6: Commit media and archive support**

```bash
git add aquillm/aquillm/ingestion/parsers.py aquillm/aquillm/ingestion/media.py aquillm/aquillm/tests/test_unified_ingestion_media.py
git commit -m "feat: add OCR, transcription, and zip expansion for unified ingestion"
```

## Chunk 2: Unified API + Concurrent Processing

### Task 4: Add batch and per-file ingestion status models

**Files:**
- Create: `aquillm/aquillm/migrations/0015_unified_ingestion_status.py`
- Modify: `aquillm/aquillm/models.py`
- Create: `aquillm/aquillm/tests/test_unified_ingestion_models.py`

- [ ] **Step 1: Write failing tests for batch/item status transitions**
- [ ] **Step 2: Run tests and confirm failure**
- [ ] **Step 3: Add `IngestionBatch` and `IngestionBatchItem` models with status enum**
- [ ] **Step 4: Create and apply migration**
- [ ] **Step 5: Run tests and confirm pass**
- [ ] **Step 6: Commit status models**

### Task 5: Add unified upload API endpoint and validation

**Files:**
- Modify: `aquillm/aquillm/api_views.py`
- Modify: `aquillm/aquillm/context_processors.py`
- Create: `aquillm/aquillm/tests/test_unified_ingestion_api.py`

- [ ] **Step 1: Write failing API tests for multi-file upload and per-file queue response**

```python
def test_upload_endpoint_queues_each_file(client, user, collection):
    response = client.post(
        "/api/ingest_uploads/",
        data={"collection": str(collection.id), "files": [pdf_file, csv_file]},
    )
    assert response.status_code == 202
    assert len(response.json()["items"]) == 2
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest -q aquillm/aquillm/tests/test_unified_ingestion_api.py`
Expected: FAIL before endpoint exists.

- [ ] **Step 3: Implement `POST /api/ingest_uploads/`**

Behavior:
- auth + collection edit permission check
- batch limits (count + bytes)
- per-file detection + queued item creation
- enqueue task per item
- return `202` with batch/item metadata

- [ ] **Step 4: Add route exposure in API URLs map**
- [ ] **Step 5: Run API tests and confirm pass**
- [ ] **Step 6: Commit unified upload endpoint**

### Task 6: Implement Celery per-file worker task

**Files:**
- Create: `aquillm/aquillm/ingestion/tasks.py`
- Modify: `aquillm/aquillm/celery.py`
- Create: `aquillm/aquillm/tests/test_unified_ingestion_tasks.py`

- [ ] **Step 1: Write failing task tests for success/failure status updates**
- [ ] **Step 2: Run tests to confirm failure**
- [ ] **Step 3: Implement `ingest_uploaded_file_task(item_id)`**

Behavior:
- mark item `processing`
- parse file to extracted text
- persist as corresponding `Document` subtype or `RawTextDocument`
- rely on existing `Document.save()` to enqueue chunking
- mark item `success` or `error`

- [ ] **Step 4: Run task tests and confirm pass**
- [ ] **Step 5: Commit ingestion worker task**

## Chunk 3: Frontend Unified Upload UX

### Task 7: Replace type-picker upload with single batch file uploader

**Files:**
- Modify: `react/src/components/IngestRow.tsx`
- Modify: `react/src/components/CollectionView.tsx`
- Create: `react/src/components/UnifiedIngestUploader.tsx`
- Create: `react/src/components/__tests__/UnifiedIngestUploader.test.tsx`

- [ ] **Step 1: Write failing frontend tests for multi-select and payload shape**
- [ ] **Step 2: Run tests to confirm failure**
- [ ] **Step 3: Implement unified uploader component**

Behavior:
- drag/drop + file picker
- multi-file list with optional title overrides
- one submit to `/api/ingest_uploads/`
- per-file row status rendering

- [ ] **Step 4: Wire collection view to new uploader**
- [ ] **Step 5: Run frontend tests/typecheck**

Run: `npm --prefix react run typecheck`
Expected: PASS.

- [ ] **Step 6: Commit unified upload UI**

## Chunk 4: Status API, Documentation, and Final Verification

### Task 8: Add ingestion batch status endpoint

**Files:**
- Modify: `aquillm/aquillm/api_views.py`
- Create: `aquillm/aquillm/tests/test_unified_ingestion_status_api.py`

- [ ] **Step 1: Write failing tests for status polling**
- [ ] **Step 2: Run tests to confirm failure**
- [ ] **Step 3: Implement `GET /api/ingest_uploads/<batch_id>/`**
- [ ] **Step 4: Run tests to confirm pass**
- [ ] **Step 5: Commit status API**

### Task 9: Update user docs and env docs

**Files:**
- Modify: `README.md`
- Modify: `.env.example`

- [ ] **Step 1: Document new supported format matrix**
- [ ] **Step 2: Document batch limits and media parsing behavior**
- [ ] **Step 3: Add configurable env vars for limits/transcription provider**
- [ ] **Step 4: Commit docs and env updates**

### Task 10: End-to-end verification

**Files:**
- Modify: `react/tests/collections.spec.ts` (or add dedicated e2e spec)
- Create: `react/tests/unified-ingest.spec.ts`

- [ ] **Step 1: Add e2e scenario for mixed-format multi-file batch upload**
- [ ] **Step 2: Run backend tests**

Run:
`python -m pytest -q aquillm/aquillm/tests/test_unified_ingestion_parsers.py aquillm/aquillm/tests/test_unified_ingestion_extractors.py aquillm/aquillm/tests/test_unified_ingestion_media.py aquillm/aquillm/tests/test_unified_ingestion_api.py aquillm/aquillm/tests/test_unified_ingestion_tasks.py aquillm/aquillm/tests/test_unified_ingestion_status_api.py`

Expected: PASS.

- [ ] **Step 3: Run frontend checks**

Run:
`npm --prefix react run typecheck`

Expected: PASS.

- [ ] **Step 4: Run e2e**

Run:
`npm --prefix react run test:e2e -- unified-ingest.spec.ts`

Expected: PASS.

- [ ] **Step 5: Commit verification and e2e coverage**

```bash
git add react/tests/unified-ingest.spec.ts react/tests/collections.spec.ts
git commit -m "test: add e2e coverage for unified multi-format ingestion"
```

