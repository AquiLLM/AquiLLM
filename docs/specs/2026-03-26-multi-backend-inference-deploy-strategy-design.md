# Multi-Backend Inference and Deploy Strategy Design

**Date:** 2026-03-26

## Goal

Enable AquiLLM to run chat, embedding, rerank, OCR, and transcription workloads across multiple backend engines (not just vLLM), using only two deployment methods: Docker and Kubernetes.

## Decision Summary

Adopt a capability-based backend architecture with phased rollout:

1. **Phase 1:** Introduce a backend registry + adapter interface and migrate chat selection from `LLM_CHOICE` to explicit backend/model config.
2. **Phase 2:** Extend the same backend abstraction to embeddings/rerank/OCR/transcribe with per-capability backend selection.
3. **Phase 3:** Add deploy-strategy profiles for Docker and Kubernetes so local/dev/prod can choose the optimal engine mix per capability.

## Problem Statement

Current runtime and docs are heavily vLLM-centric for local model serving:

- Compose and startup scripts center around `--profile vllm` and serial vLLM sidecars.
- `LLM_CHOICE` encodes model presets rather than backend strategy.
- Local users with constrained hardware/tooling face high setup friction.
- The app already has partial provider abstraction, but deployment and environment contracts do not yet provide a first-class multi-backend strategy.

Result: teams cannot easily choose the most optimal engine per model class (for example, `llama.cpp` for quantized CPU/GPU local chat, `vLLM` for high-throughput GPU serving, or `Ollama`/`LM Studio` for zero-config local onboarding).

## Scope

- Unified backend abstraction for all inference capabilities:
  - `chat`
  - `embedding`
  - `rerank`
  - `ocr`
  - `transcribe`
- Deployment strategy abstraction across:
  - Docker
  - Kubernetes
- Backward-compatible migration path from existing env and compose contracts.

## Non-Goals

- Full rewrite of provider business logic.
- Immediate implementation of every backend for every capability in the first release.
- Auto-benchmarking and dynamic runtime model routing in this first spec wave.

## Current State (Repository-Aligned)

- Chat provider selection is partly centralized in `aquillm/aquillm/apps.py` and `lib/llm/providers/*`, but local model choices are wired through vLLM-oriented env defaults.
- Embedding/OCR/transcribe already use partial endpoint-based config, but default env keys and docs still assume vLLM service names.
- Compose profiles and startup scripts (`deploy/compose/*`, `deploy/scripts/start_dev.sh`) primarily orchestrate vLLM sidecars.

## Architecture Options

### Option A: Keep OpenAI-Compatible API Only

Treat all local backends as OpenAI-compatible endpoints and avoid backend-specific adapters.

- Pros: smallest implementation effort.
- Cons: leaks backend differences into config/runtime behavior; poor support for non-OpenAI-compatible features; weak observability and error normalization.

### Option B: Backend-Specific Branching in Existing Modules

Add `if backend == ...` branches directly in current chat/embed/ocr/transcribe modules.

- Pros: fast initial delivery.
- Cons: high code entropy, duplicated fallback logic, hard testing surface, painful long-term maintenance.

### Option C (Recommended): Capability + Backend Adapter Registry

Introduce explicit backend adapters behind a stable capability contract and bind deployment strategy separately.

- Pros: clean boundaries, reusable error handling, per-capability backend choice, incremental migration.
- Cons: moderate upfront refactor.

## Recommended Design

### 1. Backend and Capability Contracts

Define two orthogonal axes:

- **Capability:** `chat`, `embedding`, `rerank`, `ocr`, `transcribe`
- **Backend engine:** `openai`, `claude`, `gemini`, `vllm`, `llama_cpp`, `llama_cpp_ik`, `ollama`, `lmstudio`

Introduce interface packages:

- `aquillm/lib/inference/backends/base.py`
- `aquillm/lib/inference/backends/<backend>.py`
- `aquillm/lib/inference/registry.py`
- `aquillm/lib/inference/types.py`

Each backend adapter declares:

- supported capabilities
- transport shape (OpenAI-compatible REST, SDK-native, process wrapper)
- config schema + validation
- normalized error mapping

### 2. Configuration Contract (Env)

Replace model-only selection with explicit capability backend selection:

- `APP_CHAT_BACKEND`
- `APP_CHAT_MODEL`
- `APP_EMBED_BACKEND`
- `APP_EMBED_MODEL`
- `APP_RERANK_BACKEND`
- `APP_RERANK_MODEL`
- `APP_OCR_BACKEND`
- `APP_OCR_MODEL`
- `APP_TRANSCRIBE_BACKEND`
- `APP_TRANSCRIBE_MODEL`

Per-backend endpoint/auth keys (examples):

- `BACKEND_VLLM_BASE_URL`, `BACKEND_VLLM_API_KEY`
- `BACKEND_OLLAMA_BASE_URL`
- `BACKEND_LMSTUDIO_BASE_URL`
- `BACKEND_LLAMA_CPP_BASE_URL`
- `BACKEND_LLAMA_CPP_IK_BASE_URL`

Backward compatibility:

- keep `LLM_CHOICE` and existing `VLLM_*` keys in Phase 1
- map legacy keys into new config with deprecation warnings
- remove legacy keys in a later major cleanup window

### 3. Deployment Strategy Abstraction

Add deploy strategy profile (separate from backend):

- `APP_DEPLOY_STRATEGY=docker|k8s`

Behavior:

- `docker`: launch selected sidecars via compose profiles (vLLM, Ollama, or mixed), and/or connect web/worker containers to configured backend endpoints.
- `k8s`: use Kustomize overlays to deploy support services and wire backend endpoints through env/config maps and secrets.

### 4. Model-to-Backend Mapping Policy

Support deterministic mapping with optional fallback chain:

- `APP_CHAT_BACKEND=vllm`
- `APP_CHAT_MODEL=qwen3.5:27b`
- `APP_CHAT_BACKEND_FALLBACKS=ollama,lmstudio`

If primary backend is unavailable, runtime can fail fast or fall back (flag controlled):

- `APP_BACKEND_FALLBACK_ENABLED=0|1`

### 5. Runtime Integration Points

Primary modules to migrate incrementally:

- `aquillm/aquillm/apps.py` (startup selection and app-level wiring)
- `aquillm/lib/llm/providers/*` (chat capability adapters)
- `aquillm/lib/embeddings/*` (embedding backend binding)
- `aquillm/lib/ocr/*` (OCR backend binding)
- `aquillm/aquillm/ingestion/media.py` (transcribe backend binding)
- `deploy/compose/*` and `deploy/scripts/start_dev.sh` (strategy/profile orchestration)
- `.env.example` and `README.md` (operator contract)

## Backend Coverage Targets by Phase

- **Phase 1 (chat):** `vllm`, `openai`, `claude`, `gemini`, `ollama`, `lmstudio`
- **Phase 2 (chat + embed/rerank):** add `llama_cpp`, `llama_cpp_ik` and normalized local OpenAI-compatible adapters
- **Phase 3 (full multimodal parity):** extend OCR/transcribe path to backend registry where capability exists; keep vLLM default where no parity backend is production-ready

## Rollout Plan

1. Introduce registry + config parser with no behavior change by default.
2. Migrate chat path to `APP_CHAT_BACKEND` + legacy key shim.
3. Add Docker quickstart profiles:
   - `compose --profile ollama`
   - optional Docker profile for llama.cpp variants when packaged
4. Migrate embed/rerank/ocr/transcribe to capability-specific backend keys.
5. Add Kubernetes overlay env contract updates for new backend keys.
6. Update docs and mark vLLM-only setup as one strategy, not the default universal path.

## Verification

- Unit tests for config parsing, legacy mapping, and backend resolution.
- Contract tests for each backend adapter (request/response normalization).
- Integration tests for fallback behavior and startup selection.
- Compose smoke tests for:
  - vLLM strategy
  - Ollama strategy
  - mixed strategy (for example chat on Ollama, rerank on vLLM)
- Kubernetes smoke tests for:
  - portable overlay with vLLM services
  - production overlay with externalized backend endpoints

## Risks and Mitigations

- **Risk:** API surface drift across backends.
  - **Mitigation:** strict normalized response contract + adapter contract tests.
- **Risk:** config explosion and operator confusion.
  - **Mitigation:** capability-prefixed env naming + profile presets + migration warnings.
- **Risk:** partial feature mismatch (tool-calling, multimodal, rerank).
  - **Mitigation:** backend capability declaration and fail-fast validation at startup.

## Assumptions

- `llama.cpp-ik` is treated as a distinct backend flavor requiring separate endpoint/config keys.
- Not every backend will support every capability initially; parity is phased.
- Existing vLLM production deploys must remain supported without disruption during migration.
