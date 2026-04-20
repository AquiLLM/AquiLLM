# WhisperX Transcription Design

**Date:** 2026-03-30  
**Status:** Planned  
**Related:**

- [WhisperX transcription implementation plan](../roadmap/plans/pending/2026-03-30-whisperx-transcription-implementation.md)

## Problem

Media ingestion today transcribes audio and video through an **OpenAI-compatible** HTTP API (`audio.transcriptions.create`), typically backed by **vLLM** serving a Whisper-family model (`vllm_transcribe` in Compose). That path yields a **single plain transcript** string. It does **not** provide the extras that [WhisperX](https://github.com/m-bain/whisperX) is known for: **accurate word-level alignment**, optional **speaker diarization**, and structured segments that often **improve perceived quality** on challenging audio.

Operators want a **local WhisperX service** that **enhances** the existing Whisper (vLLM) deployment: keep the current stack as the baseline ASR, and add an **optional second stage** without replacing `vllm_transcribe` or changing the default ingestion contract unless explicitly enabled.

## Goals

1. **Preserve the existing Whisper deploy** — `INGEST_TRANSCRIBE_PROVIDER=openai` with `vllm_transcribe` remains the **primary** transcription path for media ingestion.
2. **Add a local WhisperX HTTP service** (same Docker/Compose network as the app) that **optionally post-processes** the same upload: receives the **original audio (or demuxed audio)** and the **baseline transcript from vLLM** (or runs WhisperX’s own ASR where required by the pipeline), then returns an **enhanced** transcript and optional structured fields.
3. **Operator control** — Enhancement is **off by default**; a single clear flag (or equivalent) enables the second stage so dev/staging can validate GPU and latency.
4. **Clear deployment contract** — Dedicated container with health checks, GPU expectations, optional Hugging Face tokens for diarization, timeouts, and upload size limits.
5. **Observability** — Predictable errors if WhisperX fails: policy for **fail closed** (surface error) vs **fail open** (use vLLM text only) must be explicit (see below).

## Non-goals

- **Replacing** `vllm_transcribe` as the default ASR for AquiLLM (WhisperX may run its own ASR internally for alignment/diarization; the **product** baseline remains vLLM-first unless operators choose otherwise later).
- **Guaranteed** lower latency; enhancement adds work and possibly a second GPU-bound step.
- Full **UI** for per-segment editing in v1; optional segments can live in `ExtractedTextPayload.metadata` when enabled.

## Current architecture (baseline)

| Piece | Role |
|--------|------|
| `aquillm/aquillm/ingestion/media.py` | `transcribe_media_bytes()` — OpenAI client, `audio.transcriptions.create`, returns plain text. |
| `INGEST_TRANSCRIBE_PROVIDER` | `openai` for the vLLM-backed path; base URL often `http://vllm_transcribe:8000/v1`. |
| `deploy/compose/*.yml` | `vllm_transcribe` serves Whisper via vLLM. |
| `ExtractedTextPayload` | `full_text` plus optional `metadata` for structured extras. |

## Proposed behavior: enhance, not replace

### Pipeline

```text
[ media bytes ]
      │
      ▼
┌─────────────────────┐
│ vLLM Whisper (v1)   │  ← existing deploy; unchanged default
│ audio.transcriptions│
└──────────┬──────────┘
           │ baseline text
           ▼
     (optional)
┌─────────────────────┐
│ Local WhisperX      │  ← new service; only if enhancement enabled
│ HTTP service        │
└──────────┬──────────┘
           │ enhanced text (+ optional segments / speakers)
           ▼
[ ingestion full_text / metadata ]
```

The WhisperX service always receives enough context to do its job: at minimum **the original file bytes** (or audio extracted server-side). It may also receive **`baseline_transcript`** from vLLM so the implementation can prefer **alignment + diarization** anchored to that text where WhisperX supports it; if the library requires its own ASR segments for alignment, the implementation plan may run an internal transcribe step while still treating **vLLM as the operator-visible primary** and merging policy in code (documented in the plan).

### Configuration (illustrative — final names in implementation)

| Variable | Purpose |
|----------|---------|
| `INGEST_TRANSCRIBE_PROVIDER` | Stays **`openai`** for the standard path. |
| `INGEST_TRANSCRIBE_WHISPERX_ENHANCE` | `0` / `1` (or `true` / `false`) — enable the second stage. |
| `INGEST_TRANSCRIBE_WHISPERX_BASE_URL` | Base URL of the **local** WhisperX service (e.g. `http://whisperx:8080`). |
| `INGEST_TRANSCRIBE_WHISPERX_*` | Optional: timeout, max bytes, `language`, diarization on/off, **fail-open** behavior. |

When enhancement is **disabled**, behavior matches **today** (no WhisperX call).

### Service contract (high level)

The local WhisperX service exposes at least:

- **Health** — for Compose/Kubernetes probes.
- **Enhance** (or **transcribe**) — accepts multipart upload: **file** + optional **baseline_transcript** + optional **language** / **diarize**; returns JSON:
  - **`text`** (required) — string used as ingestion `full_text` when enhancement succeeds.
  - **`segments`** (optional) — stable schema for timestamps/speakers; may be copied into `ExtractedTextPayload.metadata` (e.g. `whisperx_segments`).

Exact paths, field names, and error codes belong in the implementation plan and the service README.

### Failure policy

- **Recommended default:** If enhancement is enabled but the WhisperX request fails (timeout, 5xx, invalid JSON), **fall back to vLLM-only text** and log a warning, unless `INGEST_TRANSCRIBE_WHISPERX_STRICT=1` (or similar) is set to **fail the ingest** for operator visibility. Document the default in README.

### Resource and licensing notes

- **GPU:** WhisperX is typically GPU-heavy. It may run on a **second GPU**, a **second machine**, or the **same** GPU as vLLM only if VRAM and scheduling are explicitly sized (not recommended for small GPUs). Compose should use a profile (e.g. `vllm`) so operators enable GPU services together.
- **Hugging Face / pyannote:** Diarization may require HF **token** and **model license acceptance**. Document env-based injection only.
- **Network:** Bind on the internal Docker network; no public exposure in default templates.

## Alternatives considered

| Approach | Upside | Downside |
|----------|--------|----------|
| **A. Local WhisperX HTTP service as second stage** (chosen) | Keeps vLLM deploy; adds quality/structure; clear boundary. | Extra container; two steps latency. |
| **B. `INGEST_TRANSCRIBE_PROVIDER=whisperx` only** | Single HTTP hop | Replaces rather than *enhances* the existing Whisper story. |
| **C. WhisperX inside Django** | One fewer service | GPU contention; heavy deps in app image. |

## Security and privacy

- Treat uploads as sensitive; avoid logging bodies or paths at info level in production.
- Secrets only via env/secrets stores.

## Success criteria

- [ ] With enhancement **off**, ingestion behavior matches pre-change semantics (tests).
- [ ] With enhancement **on** and the local service healthy, `full_text` reflects WhisperX output (or merged policy) within timeouts.
- [ ] Documentation describes the **vLLM + optional WhisperX** topology, GPU notes, and HF token for diarization.
- [ ] Tests cover enhancement off, enhancement on (mocked HTTP), and failure/fallback behavior if implemented.

## Open questions (resolve during implementation)

- Exact WhisperX pipeline: **align vLLM text to audio** vs **full internal ASR** with vLLM as baseline for comparison/fallback — pick based on library support and quality checks.
- Default **failure policy**: fail-open vs strict (see above).
- Whether **word-level JSON** in `metadata` is default-on or flag-gated for storage size.
