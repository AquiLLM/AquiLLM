# Full Multimodal Support And Storage Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver end-to-end multimodal ingestion/retrieval by running dedicated OCR and transcription model services, storing original media plus extracted text, and retrieving across both modalities.

**Architecture:** Split model-serving by workload (`vllm`, `vllm_ocr`, `vllm_transcribe`, `vllm_embed`, `vllm_rerank`) with separate GPU budgets. Ingestion stores raw media documents (image/video/audio) and derived OCR/transcript text, then creates modality-aware chunks for embedding and rerank. Retrieval fuses text and image evidence while preserving source traceability.

**Tech Stack:** Django, Celery, pgvector, vLLM OpenAI-compatible APIs, Docker Compose, React ingestion UI.

---

## File Structure Map

- Infra/deploy:
  - Modify: `docker-compose.yml`
  - Modify: `docker-compose-development.yml`
  - Modify: `docker-compose-prod.yml`
  - Modify: `deployment/start_dev.sh`
  - Modify: `deployment/relaunch_mem0_oss.sh`
- Environment/docs:
  - Modify: `.env.example`
  - Modify: `.env.multimodal`
  - Modify: `README.md`
- OCR/transcribe routing:
  - Modify: `aquillm/aquillm/ocr_utils.py`
  - Modify: `aquillm/aquillm/ingestion/media.py`
- Ingestion data model/pipeline:
  - Modify: `aquillm/aquillm/models.py`
  - Modify: `aquillm/aquillm/tasks.py`
  - Modify: `aquillm/aquillm/ingestion/types.py`
  - Modify: `aquillm/aquillm/ingestion/parsers.py`
  - Create: `aquillm/aquillm/migrations/<next>_multimodal_ingestion_media_docs.py`
- API/UI status:
  - Modify: `aquillm/aquillm/api_views.py`
  - Modify: `react/src/components/IngestRow.tsx`
  - Modify: `react/src/components/IngestionDashboard.tsx`
- Tests:
  - Modify: `aquillm/aquillm/tests/test_ocr_provider_selection.py`
  - Create: `aquillm/aquillm/tests/test_transcribe_provider_selection.py`
  - Create: `aquillm/aquillm/tests/test_multimodal_ingestion_media_storage.py`
  - Modify/Create: `aquillm/aquillm/tests/test_unified_ingestion_api.py`
  - Modify: `aquillm/aquillm/tests/test_dev_launch_script.py`

---

## Chunk 1: Dedicated Model Services (OCR + Transcribe)

### Task 1: Add `vllm_ocr` and `vllm_transcribe` compose services

**Files:**
- Modify: `docker-compose.yml`
- Modify: `docker-compose-development.yml`
- Modify: `docker-compose-prod.yml`

- [ ] **Step 1: Write failing tests/assertions for service presence**
  - Extend existing compose/script tests to assert services exist (or script waits for them).
- [ ] **Step 2: Add `vllm_ocr` service**
  - Clone existing vLLM service pattern with OCR-specific env keys:
  - `OCR_VLLM_MODEL`, `OCR_VLLM_SERVED_MODEL_NAME`, `OCR_VLLM_TOKENIZER`, `OCR_VLLM_GPU_MEMORY_UTILIZATION`, `OCR_VLLM_MAX_MODEL_LEN`, `OCR_VLLM_EXTRA_ARGS`, `OCR_VLLM_TRUST_REMOTE_CODE`.
- [ ] **Step 3: Add `vllm_transcribe` service**
  - Clone existing vLLM service pattern with ASR-specific env keys:
  - `TRANSCRIBE_VLLM_MODEL`, `TRANSCRIBE_VLLM_SERVED_MODEL_NAME`, `TRANSCRIBE_VLLM_GPU_MEMORY_UTILIZATION`, `TRANSCRIBE_VLLM_MAX_MODEL_LEN`, `TRANSCRIBE_VLLM_EXTRA_ARGS`.
- [ ] **Step 4: Wire health checks and service dependencies**
  - `web`/`worker` depend on new services as optional health-gated dependencies (same pattern used today).
- [ ] **Step 5: Verify compose config renders**
  - Run: `docker compose -f <file> config` for all modified files.

### Task 2: Update startup orchestration scripts

**Files:**
- Modify: `deployment/start_dev.sh`
- Modify: `deployment/relaunch_mem0_oss.sh`
- Test: `aquillm/aquillm/tests/test_dev_launch_script.py`

- [ ] **Step 1: Write failing test for startup sequence**
  - Assert dev startup waits for `vllm_ocr` and `vllm_transcribe` before app services.
- [ ] **Step 2: Update startup order**
  - `vllm -> vllm_ocr -> vllm_transcribe -> vllm_embed -> vllm_rerank -> web/worker`.
- [ ] **Step 3: Update relaunch helper (optional model relaunch set)**
  - Include new services when relaunching model stack.
- [ ] **Step 4: Re-run script tests**

---

## Chunk 2: Environment Profiles And Routing

### Task 3: Finalize env contracts for OCR/transcription services

**Files:**
- Modify: `.env.example`
- Modify: `.env.multimodal`
- Modify: `README.md`

- [ ] **Step 1: Add OCR service env keys**
  - `APP_OCR_PROVIDER=qwen`
  - `APP_OCR_QWEN_BASE_URL=http://vllm_ocr:8000/v1`
  - `APP_OCR_QWEN_MODEL=<served OCR model>`
  - `APP_OCR_QWEN_API_KEY=EMPTY`
  - `APP_OCR_QWEN_TIMEOUT_SECONDS=...`
- [ ] **Step 2: Add transcription service env keys**
  - `INGEST_TRANSCRIBE_PROVIDER=openai`
  - `INGEST_TRANSCRIBE_OPENAI_BASE_URL=http://vllm_transcribe:8000/v1`
  - `INGEST_TRANSCRIBE_OPENAI_API_KEY=EMPTY`
  - `INGEST_TRANSCRIBE_MODEL=<served transcribe model>`
- [ ] **Step 3: Document tuning knobs**
  - GPU utilization, context, and extra args for each service.
- [ ] **Step 4: Add operator command examples**
  - profile startup, force-recreate just OCR/transcribe services.

### Task 4: Route OCR/transcribe clients to dedicated services

**Files:**
- Modify: `aquillm/aquillm/ocr_utils.py`
- Modify: `aquillm/aquillm/ingestion/media.py`

- [ ] **Step 1: Write failing tests for default service base URLs**
- [ ] **Step 2: Set OCR qwen default base URL to `vllm_ocr`**
- [ ] **Step 3: Set transcribe default base URL to `vllm_transcribe`**
- [ ] **Step 4: Add clear error text when service/model endpoint does not support requested task**
- [ ] **Step 5: Verify existing fallback behavior remains deterministic**

---

## Chunk 3: Dual-Save Multimodal Ingestion (Raw Media + Text)

### Task 5: Extend ingestion payload contract for media persistence

**Files:**
- Modify: `aquillm/aquillm/ingestion/types.py`
- Modify: `aquillm/aquillm/ingestion/parsers.py`

- [ ] **Step 1: Add payload fields for media blobs/metadata**
  - Include modality, media filename/content-type, and extracted text fields.
- [ ] **Step 2: Keep image extraction dual output**
  - OCR text + media metadata (not text-only abstraction).
- [ ] **Step 3: Keep audio/video extraction dual output**
  - Transcript text + media metadata.
- [ ] **Step 4: Preserve current behavior for purely textual file types**

### Task 6: Add persistent media-backed document models

**Files:**
- Modify: `aquillm/aquillm/models.py`
- Create: `aquillm/aquillm/migrations/<next>_multimodal_ingestion_media_docs.py`

- [ ] **Step 1: Add image upload document model**
  - Store image file + normalized OCR text + provider metadata.
- [ ] **Step 2: Add media upload document model (audio/video)**
  - Store source file + transcript text + provider metadata.
- [ ] **Step 3: Include models in `DESCENDED_FROM_DOCUMENT`**
  - Ensure monitor/search/document listing includes new docs.
- [ ] **Step 4: Migration for schema + indexes**

### Task 7: Update ingestion task to write both raw media docs and text content

**Files:**
- Modify: `aquillm/aquillm/tasks.py`
- Modify: `aquillm/aquillm/ingestion/parsers.py`

- [ ] **Step 1: Write failing tests for image dual-save behavior**
- [ ] **Step 2: For image payloads, create media-backed image document**
- [ ] **Step 3: For video/audio payloads, create media-backed media document**
- [ ] **Step 4: Keep textual docs as `RawTextDocument` where appropriate**
- [ ] **Step 5: Persist document IDs + per-item metadata in batch item**

---

## Chunk 4: Multimodal Chunking And Retrieval

### Task 8: Ensure chunking emits modality-aware chunks for new media docs

**Files:**
- Modify: `aquillm/aquillm/models.py`

- [ ] **Step 1: Write failing tests for image/media chunk modality**
- [ ] **Step 2: Emit IMAGE chunks for image docs with linked source data URL**
- [ ] **Step 3: Emit TEXT chunks for OCR/transcript fields**
- [ ] **Step 4: Verify embed/rerank payload generation supports both chunk types**

### Task 9: Retrieval fusion policy for dual-evidence docs

**Files:**
- Modify: `aquillm/aquillm/models.py`

- [ ] **Step 1: Keep text-based retrieval path for OCR/transcripts**
- [ ] **Step 2: Keep multimodal vector retrieval path for image chunks**
- [ ] **Step 3: Add balanced score merge policy**
  - default balanced weighting; env-configurable.
- [ ] **Step 4: Add debug logs for contribution by modality**

---

## Chunk 5: API/UI And Monitoring

### Task 10: Expose modality + provider metadata in ingestion APIs

**Files:**
- Modify: `aquillm/aquillm/api_views.py`

- [ ] **Step 1: Include per-item modality in upload status response**
- [ ] **Step 2: Include OCR/transcribe provider and error details**
- [ ] **Step 3: Keep response backward-compatible for existing frontend fields**

### Task 11: Update frontend ingestion display

**Files:**
- Modify: `react/src/components/IngestRow.tsx`
- Modify: `react/src/components/IngestionDashboard.tsx`

- [ ] **Step 1: Show modality badge (text/image/audio/video/archive)**
- [ ] **Step 2: Show raw-media-saved + text-extracted status indicators**
- [ ] **Step 3: Keep stale-row cleanup behavior and success refresh hooks**

---

## Chunk 6: Verification And Test Plan

### Task 12: Backend tests

**Files:**
- Modify: `aquillm/aquillm/tests/test_ocr_provider_selection.py`
- Create: `aquillm/aquillm/tests/test_transcribe_provider_selection.py`
- Create: `aquillm/aquillm/tests/test_multimodal_ingestion_media_storage.py`
- Modify/Create: `aquillm/aquillm/tests/test_unified_ingestion_api.py`

- [ ] **Step 1: OCR route tests target `vllm_ocr` defaults**
- [ ] **Step 2: Transcribe route tests target `vllm_transcribe` defaults**
- [ ] **Step 3: Dual-save tests ensure both raw media and extracted text are persisted**
- [ ] **Step 4: End-to-end item status tests for image/video files**

### Task 13: Infra/script tests and smoke checks

**Files:**
- Modify: `aquillm/aquillm/tests/test_dev_launch_script.py`

- [ ] **Step 1: assert launch script includes `vllm_ocr` + `vllm_transcribe` waits**
- [ ] **Step 2: run compose config smoke checks for all compose files**
- [ ] **Step 3: run compile/type checks for touched python/ts files**

---

## Rollout Sequence

1. Land compose/service/env routing changes first and validate model services become healthy.
2. Land ingestion dual-save model/migration/task changes.
3. Land chunking/retrieval fusion updates.
4. Land API/UI updates.
5. Run full regression + ingest sample suite (png, jpg, mp4, mp3, pdf).

## Operational Guardrails

- Keep `.env` as baseline and `.env.multimodal` as explicit profile.
- Rotate `MEM0_COLLECTION_NAME` when embed model/dims change.
- If `APP_EMBED_DIMS` and embedding output dims differ, either align schema or keep controlled truncation with explicit warning budget.
- Treat OCR/transcribe service scaling independently from chat service to protect latency.

## Execution Hand-off

Plan complete and saved to `docs/superpowers/plans/2026-03-17-full-multimodal-storage-and-ingestion.md`. Ready to execute.
