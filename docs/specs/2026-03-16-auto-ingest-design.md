# 2026-03-16 — Unified, Source-Agnostic Document Ingestion (Design)

## Goal

Make AquiLLM support **“upload a bunch of documents without thinking about type”**, including:

- **Multi-file uploads** (mixed types)
- **Folder ingestion** that is **cross-browser reliable**
  - v1: **upload a `.zip`** containing a folder tree
  - optional progressive enhancement: native folder pick where supported
- **Extensible ingestion sources** beyond local uploads:
  - **Zotero**: manual import + background sync
  - **arXiv**: import by identifier/URL
  - (future) additional sources implemented without adding new “type-specific” ingestion endpoints

This design focuses on **abstracting file type selection and parsing**, not on new chunking or retrieval behavior.

## Non-goals (v1)

- Perfect parsing for every file type “under the sun”
- Cross-browser native folder APIs (not available reliably)
- Replacing chunking/search pipeline (reuse existing `create_chunks` task behavior)
- Forcing immediate migration of all legacy document rows (we’ll provide a phased path)

## Current state (repository reality)

Today ingestion is type-specific:

- HTML views: `ingest_pdf`, `ingest_vtt`, `ingest_handwritten_notes` in `aquillm/aquillm/views.py`
- API views: `ingest_pdf`, `ingest_vtt`, `ingest_arxiv` in `aquillm/aquillm/api_views.py`
- Zotero sync: `sync_zotero_library` in `aquillm/aquillm/zotero_tasks.py` downloads PDFs and creates `PDFDocument`
- Parsing is coupled to per-type models in `aquillm/aquillm/models.py`:
  - `PDFDocument.extract_text()` uses `pypdf`
  - VTT parsing uses `aquillm/aquillm/vtt.py`
  - OCR uses `aquillm/aquillm/ocr_utils.py`
- Chunking is centralized via Celery task `create_chunks(doc_id)` in `aquillm/aquillm/models.py`

## Key idea

Split ingestion into **two orthogonal layers**:

1. **Sources** (where bytes come from): Uploads / Zip container / Zotero / arXiv / Webpage crawler / …
2. **Parsers** (how bytes become text + metadata): PDF / VTT / Image OCR / Plain text / HTML / …

Everything flows through a single **UnifiedIngestService** that:

- normalizes inputs
- detects type
- selects parser
- produces `full_text` + metadata + assets
- persists a document record
- triggers chunking asynchronously (existing mechanism)

This removes the need for any “file type selection” UI.

## User experience (Collections UI)

### “Upload documents” modal (v1)

Single modal that supports:

- Drag & drop multiple files
- Multi-select files
- Upload a `.zip` produced from a folder (cross-browser reliable folder ingestion)
- Optional global toggle(s) that apply when relevant:
  - Example: “Convert math to LaTeX (images)” (only affects OCR)

Modal output:

- Per-file status row: `Queued → Uploaded → Parsing → Chunking → Complete` (or error)
- Show folder path for zip entries (e.g. `course/week1/notes.pdf`)

### Folder support: why zip

Native folder selection/drag-drop is inconsistent across browsers. For cross-browser reliability, v1 uses:

- **Zip upload** as the guaranteed path
- optional progressive enhancement where supported later

## API surface

### Unified upload endpoint

`POST /api/ingest_files/` (multipart/form-data)

- `collection_id` (required)
- `files[]` (required; can include a `.zip`)
- `options` (optional JSON string)
  - `ocr_convert_to_latex: bool` (applies to images)
  - future: per-file options keyed by client path/name

Response:

```json
{
  "results": [
    {
      "client_name": "notes/week1.pdf",
      "detected_type": "pdf",
      "document_id": "uuid",
      "status": "accepted",
      "error": null
    }
  ]
}
```

Notes:

- The endpoint returns `accepted` quickly after persistence + enqueue, not after chunking completes.
- Errors are per-file and do not abort the whole batch.

### Manual Zotero import endpoint (new)

`POST /api/ingest_zotero_item/`

- `collection_id`
- `zotero_item_key_or_url`
- options (e.g. “attachments only”, “prefer PDF”)

Returns per-attachment results using the same shape as file uploads.

#### Zotero ingestion caveats (user-facing and implementation)

**Link-only vs file attachments:** Ingestion depends on how items were added to Zotero. If the user attached the actual PDF (e.g. a paywalled paper they have access to), we can download and ingest it. If they only added a *link* or reference without attaching a file, there is nothing to download—those items cannot be ingested. The system must:

- Skip link-only / snapshot-only attachments gracefully (no file available from Zotero API).
- Return a clear per-item status, e.g. `skipped_no_file` or `error: "No file attached in Zotero"`, so the user understands which items were not ingested and why.

**Duplicates:** Zotero libraries can contain duplicates (same paper added twice, or same attachment linked from different items). The implementation must:

- **Dedupe by Zotero attachment key:** Before downloading, check if we already have a document with `source_provider="zotero"` and `source_id=<attachment_key>`. If so, skip and return status `skipped_duplicate` (or link to existing document).
- Optionally, in sync results or UI, report counts: “X ingested, Y skipped (no file), Z skipped (already in library).”

### arXiv import endpoint (existing -> refactor)

Keep `POST /api/ingest_arxiv/` but refactor it to call UnifiedIngestService rather than instantiating `PDFDocument`/`TeXDocument` directly.

## Core backend components

### 1) IngestionInput (normalized input objects)

All sources normalize to a small set of input types:

- `UploadedFileInput(file, client_path, display_name, source_ref)`
- `ZipEntryInput(bytes, client_path, display_name, source_ref)`
- `RemoteBytesInput(bytes, display_name, source_ref)`
- `RemoteTextInput(text, display_name, source_ref)`

`source_ref` is a small dict for provenance and idempotency, e.g.:

- Zotero: `{provider:"zotero", attachment_key:"ABCD1234", item_key:"WXYZ9876", library_id:"personal|<group>" }`
- arXiv: `{provider:"arxiv", arxiv_id:"2101.12345", variant:"pdf|tex" }`
- Upload: `{provider:"upload", upload_session:"uuid", client_path:"..." }`

### 2) Type detection (no user selection)

Detector uses content sniffing first, then extension/MIME fallback:

- PDF: starts with `%PDF`
- VTT: first non-empty line `WEBVTT`
- Image: Pillow can open
- Zip: signature `PK\x03\x04` or extension `.zip`
- Text: decodable bytes (try utf-8, fallback `chardet`)
- HTML: looks like `<!doctype html>`/`<html` (fallback only)

Returns: `detected_type`, `mime`, `ext`, `confidence`, plus any hints.

### 3) Parser registry (pluggable)

Interface:

- Input: normalized bytes/stream + detection result + options
- Output:
  - `full_text: str`
  - `metadata: dict`
  - `assets: list[AssetToAttach]` (e.g. original binary, audio sidecar)

Initial parsers (reuse existing code):

- PDF parser: `pypdf` extraction (existing behavior)
- VTT parser: `aquillm/aquillm/vtt.py`
- Image OCR parser: `aquillm/aquillm/ocr_utils.py`
- Text/Markdown parser: decode + store as-is
- HTML parser (optional in v1): `trafilatura` or BeautifulSoup (already dependency present)

### 4) Zip container handling

Zip is treated as a *container source*:

- Unpack safely (streaming if possible; otherwise temp storage)
- Convert each file entry into an `IngestionInput` with `client_path`
- Feed each entry into the same ingestion pipeline

**Safety constraints (must-have):**

- ZipSlip protection: reject entries with absolute paths or `..`
- Limits:
  - max file count
  - max per-file bytes
  - max total uncompressed bytes
- Skip macOS/Windows metadata files (`__MACOSX/`, `.DS_Store`, `Thumbs.db`)

### 5) UnifiedIngestService (single ingestion workflow)

Responsibilities:

1. Validate permission (user can edit target collection)
2. Normalize inputs (including expanding zip)
3. For each input:
   - detect type
   - parse via registry
   - persist document + assets
   - enqueue chunking (existing Celery task)
4. Return per-input results

**Idempotency / dedupe:**

- Use `source_ref` to avoid re-importing the same remote artifact:
  - Zotero: attachment key is stable; do `get_or_create` by `(provider, provider_id)`
  - arXiv: `(arxiv_id, variant)`
- Use `full_text_hash` as a secondary dedupe mechanism (optional; existing logic is currently partially disabled for debugging—do not rely solely on it).

## Data model strategy (phased)

We want to converge toward a **single generic Document model** long-term, but the safest path is phased.

### Phase 1 (v1): add new generic models, don’t break legacy

Add:

- `DocumentV2` (concrete model):
  - uuid `id`, `title`, `full_text`, `full_text_hash`
  - `collection`, `ingested_by`, timestamps, `ingestion_complete`
  - `doc_type` (enum)
  - provenance: `source_provider`, `source_id`, optional `source_metadata` JSON
- `DocumentAsset`:
  - FK to `DocumentV2`
  - `role` (`original`, `audio`, etc.)
  - `file`, `mime`, `size_bytes`

New ingestion creates `DocumentV2` only.

Legacy reads remain intact.

### Phase 2: read compatibility layer

Update listing/retrieval code to show both:

- legacy docs (current per-type models)
- new `DocumentV2` docs

### Phase 3: backfill legacy docs (optional)

Provide a migration/command to convert legacy docs into `DocumentV2` rows, preserving:

- UUIDs, titles, collection, user, timestamps
- full_text/full_text_hash
- associated binaries copied/moved into `DocumentAsset`

## Progress reporting

Reuse current approach:

- After persistence, enqueue chunking (existing Celery task)
- Use existing websocket progress events keyed by `doc_id`

For batches (many files), the UI shows multiple rows listening for their respective `doc_id`.

## Error handling principles

- Fail per document, not the whole batch
- Errors should be user-actionable:
  - “Unsupported file type”
  - “Zip contained 3,200 files; limit is 1,000”
  - “PDF text extraction failed”
- Store enough metadata for operators to debug:
  - source provenance
  - detection + parser chosen
  - exception string (bounded)

## Security considerations

- Zip safety (ZipSlip, size limits)
- Upload size limits at Django/web server level
- Avoid executing embedded content; parsers must treat inputs as data
- HTML extraction should not execute JS; must be server-side static extraction only

## Testing plan

- Unit tests:
  - type detection (PDF/VTT/image/zip/text)
  - ZipSlip protection + size limit enforcement
  - parser outputs for small fixtures
- Integration tests:
  - multipart with mixed files
  - multipart with zip containing nested folders and mixed types
- Regression test:
  - Zotero sync still works but creates `DocumentV2` (not `PDFDocument`) and dedupes on attachment key
  - arXiv ingestion still works via unified path

## Rollout plan (recommended)

1. Add backend ingestion abstractions + API endpoint
2. Add Collections modal that calls the endpoint (multi-file + zip)
3. Wire Zotero manual import + refactor sync task to call unified ingest
4. Refactor arXiv path to call unified ingest
5. (Optional) backfill legacy docs

## Caveats and edge cases

This section consolidates source-specific limitations and the status/error cases the system must handle. Implementations must return the documented status values and messages so the UI can show clear, actionable feedback.

### Status and result contract

Per-item result shape:

- `status`: one of `accepted` | `skipped_no_file` | `skipped_duplicate` | `skipped_unsupported_type` | `error`
- `document_id`: set when `status === "accepted"`
- `existing_document_id`: set when `status === "skipped_duplicate"` (optional, for linking)
- `error`: short user-facing message when `status === "error"` or `skipped_*` (e.g. reason for skip)
- `detected_type`: when relevant (e.g. uploads)

### Zotero

| Case | Condition | Behavior | Status | User-facing message / note |
|------|-----------|----------|--------|----------------------------|
| Link-only / no file | User added only a link or snapshot in Zotero; no file attached. `download_file` returns `None` or empty. | Do not create a document. | `skipped_no_file` | e.g. "No file attached in Zotero" |
| Duplicate | Document already exists with `source_provider="zotero"` and `source_id=attachment_key`. | Skip ingest. Optionally return `existing_document_id`. | `skipped_duplicate` | e.g. "Already in library" |
| Success | File downloaded and parsed. | Create DocumentV2, enqueue chunking. | `accepted` | — |
| API / network failure | Zotero API error or timeout. | Do not create document. | `error` | e.g. "Zotero API error: …" |

**User-facing caveat:** Ingestion depends on how items were added to Zotero. If the user attached the actual PDF (e.g. a paywalled paper they have access to), we can ingest it. If they only added a link or reference without attaching a file, there is nothing to download—those items cannot be ingested. The UI should explain this (e.g. in Zotero sync/import help text).

### Uploads (files and zip entries)

| Case | Condition | Behavior | Status | User-facing message / note |
|------|-----------|----------|--------|----------------------------|
| Unsupported type | Detector returns a type with no registered parser, or type `unknown` and no fallback. | Skip this file only. | `skipped_unsupported_type` | e.g. "Unsupported file type" |
| Empty file | File size 0 or empty content after read. | Skip. | `error` or `skipped_*` | e.g. "File is empty" |
| Zip: too many files | Entry count &gt; configured max (e.g. 1000). | Reject entire zip. | `error` (per zip) | e.g. "Zip contains too many files (limit 1000)" |
| Zip: ZipSlip / path escape | Entry path contains `..` or absolute path. | Reject entire zip (or reject that entry and continue, per security policy). | `error` | e.g. "Invalid path in zip" |
| Zip: size limits | Per-file or total uncompressed size exceeds limit. | Reject zip or skip oversized entries; document which. | `error` or per-entry skip | e.g. "File exceeds size limit" |
| Parser failure | Parser throws (e.g. corrupted PDF, invalid VTT). | Do not create document for this file. | `error` | e.g. "PDF text extraction failed" |
| Success | Type detected, parser succeeded, document saved. | Create DocumentV2, enqueue chunking. | `accepted` | — |

### arXiv

| Case | Condition | Behavior | Status | User-facing message / note |
|------|-----------|----------|--------|----------------------------|
| Not found | 404 from arXiv (bad id or URL). | No document created. | `error` | e.g. "arXiv ID not found" |
| No PDF or TeX | Neither `/pdf/` nor `/src/` returns a usable response. | No document created. | `error` | e.g. "No PDF or source available for this arXiv ID" |
| Duplicate | Document already exists with `source_provider="arxiv"` and same `source_id` (e.g. `2101.12345:pdf`). | Skip. | `skipped_duplicate` | e.g. "Already ingested" |
| Success | PDF or TeX fetched and ingested. | Create DocumentV2, enqueue chunking. | `accepted` | — |

### General (all sources)

| Case | Condition | Behavior |
|------|-----------|----------|
| No permission | User cannot edit the target collection. | Return 403; do not ingest. |
| Collection missing | `collection_id` invalid or not found. | Return 400/404; do not ingest. |
| Batch partial failure | Some items fail (e.g. one PDF corrupted). | Return 200 with `results` array; failed items have `status: "error"` and `error` set. Do not abort the whole batch. |

### Sync / UI summary (Zotero and batch uploads)

Where applicable (e.g. Zotero sync, multi-file upload modal), report aggregate counts so users can see what happened at a glance:

- **Ingested:** count of `accepted`
- **Skipped (no file):** count of `skipped_no_file` (Zotero only)
- **Skipped (duplicate):** count of `skipped_duplicate`
- **Skipped (unsupported type):** count of `skipped_unsupported_type`
- **Errors:** count of `error`, with per-item details in the results list

## Open questions (intentionally deferred)

- Per-file options UI (e.g. different OCR flags per image in a batch)
- Support for Office formats (DOCX/PPTX) — can be added as parsers later
- Richer chunking strategies per doc_type (out of scope for v1)

