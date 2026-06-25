# WhisperX Transcription Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a **local WhisperX HTTP service** that **enhances** the existing **vLLM Whisper** (`vllm_transcribe`) deployment: primary ASR stays OpenAI-compatible; an **optional second stage** calls WhisperX for alignment, optional diarization, and richer transcript text/metadata.

**Architecture:** (1) `transcribe_media_bytes` keeps calling vLLM via the existing client when `INGEST_TRANSCRIBE_PROVIDER=openai`. (2) If `INGEST_TRANSCRIBE_WHISPERX_ENHANCE` is enabled and `INGEST_TRANSCRIBE_WHISPERX_BASE_URL` is set, POST the **same media bytes** (and **baseline transcript** from step 1) to the local WhisperX service; use its JSON `text` as the final transcript (subject to failure policy). WhisperX runs in its **own container** on the Compose network (e.g. `whisperx` or `whisperx_transcribe`).

**Tech Stack:** Python (Django ingestion), Docker/Compose, WhisperX in a sidecar image, HTTP client + mocks in tests.

**Depends on:**

- `docs/specs/2026-03-30-whisperx-transcription-design.md`

---

## Scope

### In scope

- **`media.py`**: After successful vLLM transcription, conditionally call the WhisperX **enhance** endpoint; implement **fail-open** (fallback to vLLM text) or **strict** failure per env (match design).
- **Environment variables**: `INGEST_TRANSCRIBE_WHISPERX_ENHANCE`, `INGEST_TRANSCRIBE_WHISPERX_BASE_URL`, timeouts, max bytes, optional `INGEST_TRANSCRIBE_WHISPERX_STRICT`, diarization/language helpers, documented in `.env.example`.
- **Local service**: Dockerfile + small HTTP app (e.g. FastAPI) under `deploy/docker/whisperx/` implementing health + enhance route; pin WhisperX/CUDA base image versions.
- **Compose**: New service **after** / alongside `vllm_transcribe` on internal network; same GPU profile pattern as other vLLM services; document VRAM when both run on one host.
- **Tests**: Unit tests for “enhance off”, “enhance on” with mocked WhisperX HTTP, and failure paths (fallback vs strict).
- **Docs**: README + `.env.example`; spec already indexed in `docs/specs/README.md`.

### Out of scope

- Removing `vllm_transcribe` or changing the default `INGEST_TRANSCRIBE_PROVIDER` away from `openai`.
- Kubernetes manifests (follow-up unless done in the same release).

---

## Proposed file structure

| Path | Action | Responsibility |
|------|--------|------------------|
| `aquillm/aquillm/ingestion/media.py` | Modify | vLLM first; optional WhisperX enhance; failure policy. |
| `aquillm/aquillm/ingestion/parsers.py` | Modify (optional) | Attach `metadata` from WhisperX response if returned. |
| `deploy/docker/whisperx/` | Create | Dockerfile + FastAPI (or similar) + WhisperX pipeline. |
| `deploy/compose/base.yml` (and env-specific) | Modify | `whisperx` service; `depends_on` ordering vs `vllm_transcribe` if app needs both URLs (app may only need network access). |
| `.env.example` | Modify | `INGEST_TRANSCRIBE_WHISPERX_*` vars. |
| `README.md` | Modify | “vLLM Whisper + optional local WhisperX enhance” subsection. |
| `aquillm/aquillm/ingestion/tests/` | Modify/Create | Enhancement and fallback tests. |

---

## Chunk 1: WhisperX service API and container

### Task 1: Define enhance endpoint and Dockerfile

**Files:**

- Create: `deploy/docker/whisperx/` with `Dockerfile` + app implementing:
  - `GET /health`
  - `POST /v1/enhance` (or `/v1/transcribe`) — multipart: `file` (required), `baseline_transcript` (optional string), `language` (optional), `diarize` (optional bool)
  - Response JSON: `{ "text": "...", "model": "...", "segments": [...] }` — `segments` optional in v1

- [ ] **Step 1:** Implement WhisperX pipeline inside the handler (transcribe + align + optional diarization per library API). If alignment **requires** internal ASR segments, document how **baseline_transcript** is used (e.g. merge, or vLLM used only when enhancement disabled — align with design open questions).
- [ ] **Step 2:** Request size limit + server-side timeout; return 413/504 with clear JSON errors.
- [ ] **Step 3:** `docker build` smoke test + `curl` health.

---

## Chunk 2: Compose and environment

### Task 2: Wire local `whisperx` service

**Files:**

- `deploy/compose/base.yml`, `development.yml`, `production.yml` as needed
- `.env.example`

- [ ] **Step 1:** Add service (e.g. `whisperx`) with GPU reservation, HF cache volume, internal port, profile `vllm` (or shared inference profile).
- [ ] **Step 2:** Document `INGEST_TRANSCRIBE_WHISPERX_BASE_URL=http://whisperx:<port>` (match service name/port).
- [ ] **Step 3:** Document dual-GPU vs single-GPU constraints when `vllm_transcribe` and `whisperx` both run.

---

## Chunk 3: Ingestion — vLLM then WhisperX

### Task 3: Implement enhancement hook in `media.py`

**Files:**

- `aquillm/aquillm/ingestion/media.py`
- `aquillm/aquillm/ingestion/parsers.py` (if metadata threading)

- [ ] **Step 1:** Keep existing `transcribe_media_bytes` flow for vLLM; capture `baseline = text` from OpenAI-compatible response.
- [ ] **Step 2:** If `INGEST_TRANSCRIBE_WHISPERX_ENHANCE` is truthy and base URL is non-empty, POST `file` + `baseline_transcript` to WhisperX; parse `text`.
- [ ] **Step 3:** On WhisperX failure: if not strict, return `baseline`; if strict, raise `RuntimeError` with actionable message.
- [ ] **Step 4:** Optionally merge `segments` into extraction metadata (extend return path or a small result type — prefer minimal signature change).

---

## Chunk 4: Tests

### Task 4: Pytest coverage

- [ ] **Step 1:** Enhancement disabled — no HTTP to WhisperX (mock asserts not called).
- [ ] **Step 2:** Enhancement enabled — mock WhisperX returns `{"text": "enhanced"}`; assert final transcript is enhanced.
- [ ] **Step 3:** WhisperX fails + fail-open — assert fallback to baseline.
- [ ] **Step 4:** WhisperX fails + strict — assert error.
- [ ] **Step 5:** Run `pytest` on ingestion tests.

---

## Chunk 5: Docs handoff

### Task 5: README and operator note

- [ ] **Step 1:** README: describe **local** WhisperX as an **enhancer** for the existing Whisper (vLLM) stack.
- [ ] **Step 2:** Note HF token for diarization and failure policy defaults.

---

## Exit criteria

- [ ] vLLM-only path unchanged when enhancement is off.
- [ ] With enhancement on, local WhisperX is invoked and its `text` is used when successful.
- [ ] Compose brings up `vllm_transcribe` + `whisperx` on the same network for manual smoke.
- [ ] No secrets in repo; `.env.example` documents flags only.
