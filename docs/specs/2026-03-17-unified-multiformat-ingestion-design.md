# Unified Multi-Format Ingestion Design

**Date:** 2026-03-17

## Goal

Replace file-type-specific upload flows with one unified upload experience that auto-detects file type, supports multi-file uploads, and ingests files concurrently so indexing remains fast.

## Scope

- Add one API endpoint for file uploads that accepts multiple files in one request.
- Auto-detect file type per uploaded file and route to the appropriate parser/transcriber/OCR path.
- Support broad format coverage in a single ingestion pipeline.
- Dispatch ingestion per file asynchronously to allow concurrent processing.
- Return per-file status for mixed-success batches.
- Keep current link-based imports (`arXiv`, webpage crawl) as separate paths.

## Supported File Types

- Documents: `.pdf`, `.doc`, `.docx`, `.odt`, `.rtf`, `.txt`, `.md`, `.html`, `.htm`, `.epub`
- Spreadsheets/tabular: `.csv`, `.tsv`, `.xls`, `.xlsx`, `.ods`
- Presentations: `.ppt`, `.pptx`, `.odp`
- Structured data: `.json`, `.jsonl`, `.xml`, `.yaml`, `.yml`
- Captions/transcripts: `.vtt`, `.srt`
- Images (OCR): `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`, `.webp`, `.heic`, `.heif`
- Audio (transcribe): `.mp3`, `.wav`, `.m4a`, `.aac`, `.flac`, `.ogg`, `.opus`
- Video (extract audio + transcribe): `.mp4`, `.mov`, `.m4v`, `.webm`, `.mkv`, `.avi`, `.mpeg`, `.mpg`
- Archives: `.zip` (expand and ingest only supported file types inside)

## Non-Goals

- No change to chat interaction semantics.
- No change to existing `arXiv` and webpage import APIs.
- No attempt to index binary formats we cannot parse safely.
- No retrieval model changes beyond feeding extracted text into existing embedding/rerank flow.

## Design

### 1. Unified Upload Entry Point

Introduce `POST /api/ingest_uploads/` for file uploads. The endpoint validates collection permissions once, validates batch limits (count/size), and enqueues each file for asynchronous ingestion.

### 2. File Type Detection and Routing

Use a parser registry keyed by normalized extension and MIME hint:

1. Normalize extension and MIME.
2. Resolve ingestion strategy via registry.
3. Fallback to lightweight content sniffing for ambiguous types.
4. If unsupported: return explicit per-file error (`unsupported_file_type`).

### 3. Text Extraction Contract

Every strategy outputs a common payload:

- `title`
- `normalized_type` (e.g., `pdf`, `docx`, `audio_transcript`)
- `full_text`
- `optional_native_file` (for model-specific document rows where needed)
- `metadata` (parser info, source filename, warnings)

The extracted text then enters the existing `Document` + chunk + embedding + rerank pipeline.

### 4. Concurrency Model

- API call enqueues one Celery task per file (`ingest_uploaded_file_task`).
- Worker concurrency processes files in parallel.
- Large media parsing/transcription runs in background workers, not request thread.
- Response returns quickly with batch and per-file queued states.

### 5. Batch Status and Observability

- Persist per-file ingestion status (`queued`, `processing`, `success`, `error`).
- Expose status in API response and allow polling by batch id.
- Reuse existing ingestion dashboard concepts where possible.

### 6. Error Handling

- Per-file isolation: one file failure does not fail whole batch.
- Clear structured errors (`invalid_extension`, `parse_failed`, `transcription_failed`, `too_large`, `unsupported_file_type`).
- Keep source file metadata for failed attempts for debugging.

### 7. Security and Resource Controls

- Enforce max file size per file and max files per batch.
- For archives, enforce max expanded file count and total expanded size.
- Reject executable/script files by default.
- Require authenticated user and collection edit permissions.

## Verification

- Upload a mixed multi-file batch (5+ types) in one request and confirm independent success/failure statuses.
- Confirm each successful file generates a document and chunks with embeddings.
- Confirm audio/video files generate transcript-backed documents.
- Confirm unsupported types return explicit per-file errors without breaking other files in batch.
- Confirm concurrent workers process batch files in parallel.

