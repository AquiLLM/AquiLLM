# RAG Token Efficiency Enhancements (LM-Lingua2 + LMCache) Implementation Plan

> **Status (2026-03-23):** Superseded by `docs/roadmap/plans/active/2026-03-23-caching-rag-token-efficiency-optimization-refresh.md`, which aligns with the current provider code and existing OpenAI token-overflow safeguards.

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve end-to-end RAG token efficiency and context utilization by adding controlled prompt compression (LM-Lingua2) and optional vLLM KV reuse/offloading plumbing (LMCache) across the full pipeline after the refactor/remediation baseline is stable.

**Architecture:** Add a provider-agnostic token-efficiency policy layer in `lib/llm`, then integrate LM-Lingua2 in all active RAG answer-provider paths (OpenAI-compatible/vLLM, Claude, Gemini) using the same guardrails and fail-open behavior. For LMCache, add optional deployment wiring for RAG-serving vLLM services (chat, embed, rerank) via env/compose and `vllm_start.sh`, with immediate rollback by flag disable.

**Tech Stack:** Django 5.1, Python, vLLM OpenAI-compatible endpoints, Redis, Docker Compose, pytest.

**Roadmap Positioning:** This work **depends on the refactor/remediation track** and should start only after remediation baseline commits are merged and verified.

**Concept Source (for LMCache overlay):**
- `arXiv:2511.01815v2` â€” *KV Cache Transform Coding for Compact Storage in LLM Inference* (submitted Nov 3, 2025; revised Mar 11, 2026).
- Core concept to apply here: KV cache transform coding via decorrelation + adaptive quantization + entropy coding, layered on top of reusable/offloaded KV cache workflows.

**Scope Coverage (full RAG pipeline):**
- LM-Lingua2 compression for RAG answer-generation provider paths (`openai`, `claude`, `gemini`) while preserving tool payload integrity.
- Shared policy/flags for memory-augmented and retrieval-heavy prompts in chat turns.
- LMCache runtime wiring for vLLM RAG services: chat generation (`vllm`), embeddings (`vllm_embed`), and rerank (`vllm_rerank`).
- KVTC-style compression-on-offload layer on top of LMCache for compact KV storage and faster restore under memory pressure.

**Depends on:**
- `docs/roadmap/plans/pending/2026-03-21-architecture-boundary-and-structural-remediation.md`
- `docs/roadmap/plans/pending/2026-03-21-architecture-remediation-commit-plan.md`
- `docs/roadmap/plans/superseded/2026-03-22-multimodal-rag-caching-latency-optimization.md`

**Required baseline before execution (minimum):**
- Commit 3 (`apps.documents` chunking extraction)
- Commit 6 (`lib.tools` search/doc tool extraction)
- Commit 11 (ingestion API decomposition)
- Commit 12 (`apps.documents` chunk services split)

---

## Chunk 1: Token-Efficiency Foundation and Feature Flags

### Task 1: Add global token-efficiency settings and env contracts

**Files:**
- Modify: `aquillm/aquillm/settings.py`
- Modify: `.env.example`
- Test: `aquillm/tests/integration/test_settings_security_flags.py`

- [ ] **Step 1: Write failing settings tests**

```python
def test_token_efficiency_flags_exist(settings):
    assert hasattr(settings, "TOKEN_EFFICIENCY_ENABLED")
    assert hasattr(settings, "LM_LINGUA2_ENABLED")
    assert hasattr(settings, "LMCACHE_ENABLED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd aquillm && pytest tests/integration/test_settings_security_flags.py::test_token_efficiency_flags_exist -q`  
Expected: FAIL.

- [ ] **Step 3: Add settings and safe defaults**

Add env-driven settings:
- `TOKEN_EFFICIENCY_ENABLED` (default `False`)
- `LM_LINGUA2_ENABLED` (default `False`)
- `LM_LINGUA2_TARGET_RATIO` (default `0.65`)
- `LM_LINGUA2_MIN_PROMPT_TOKENS` (default `1800`)
- `LM_LINGUA2_MAX_PROMPT_TOKENS` (default `24000`)
- `LM_LINGUA2_SKIP_WITH_TOOLS` (default `True`)
- `LMCACHE_ENABLED` (default `False`)
- `LMCACHE_PROFILE` (default `"local"`)

- [ ] **Step 4: Add `.env.example` entries**

Add matching commented variables with rollout-safe defaults (`0` / conservative values).

- [ ] **Step 5: Re-run tests**

Run: `cd aquillm && pytest tests/integration/test_settings_security_flags.py::test_token_efficiency_flags_exist -q`  
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add aquillm/aquillm/settings.py .env.example aquillm/tests/integration/test_settings_security_flags.py
git commit -m "feat(token-efficiency): add LM-Lingua2 and LMCache feature-flag settings"
```

### Task 2: Introduce shared token-efficiency policy helpers

**Files:**
- Create: `aquillm/lib/llm/utils/token_efficiency.py`
- Create: `aquillm/lib/llm/tests/test_token_efficiency_policy.py`

- [ ] **Step 1: Write failing policy tests**

```python
def test_should_compress_requires_feature_flags():
    assert should_compress_prompt(enabled=False, prompt_tokens=3000, has_tools=False) is False
```

```python
def test_should_compress_respects_min_prompt_tokens():
    assert should_compress_prompt(enabled=True, prompt_tokens=400, has_tools=False) is False
```

- [ ] **Step 2: Run tests to verify fail**

Run: `cd aquillm && pytest lib/llm/tests/test_token_efficiency_policy.py -q`  
Expected: FAIL.

- [ ] **Step 3: Implement minimal policy module**

Add pure helpers:
- `should_compress_prompt(...)`
- `normalized_target_ratio(...)`
- `safe_prompt_hash(...)` (for cache/metrics keys)

- [ ] **Step 4: Re-run tests**

Run: `cd aquillm && pytest lib/llm/tests/test_token_efficiency_policy.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aquillm/lib/llm/utils/token_efficiency.py aquillm/lib/llm/tests/test_token_efficiency_policy.py
git commit -m "feat(llm): add shared token-efficiency policy helpers"
```

---

## Chunk 2: LM-Lingua2 Prompt Compression Integration

### Task 3: Add LM-Lingua2 adapter with fail-open behavior

**Files:**
- Create: `aquillm/lib/llm/optimizations/lm_lingua2_adapter.py`
- Modify: `requirements.txt`
- Create: `aquillm/lib/llm/tests/test_lm_lingua2_adapter.py`

- [ ] **Step 1: Write failing adapter tests**

```python
def test_compress_prompt_returns_original_on_disabled():
    assert compress_prompt_if_enabled("abc", enabled=False)[0] == "abc"
```

```python
def test_compress_prompt_fail_open_on_adapter_error(monkeypatch):
    # simulated compression exception should return original text and status=error
```

- [ ] **Step 2: Run tests to verify fail**

Run: `cd aquillm && pytest lib/llm/tests/test_lm_lingua2_adapter.py -q`  
Expected: FAIL.

- [ ] **Step 3: Implement adapter**

Implementation requirements:
- Lazy import LM-Lingua2 dependency.
- Fail-open on import/runtime errors.
- Return structured metadata: `applied`, `status`, `before_tokens`, `after_tokens`, `ratio`.
- Do not mutate input message order.

- [ ] **Step 4: Add dependency**

Add LM-Lingua2 package pin in `requirements.txt` with conservative version constraint.

- [ ] **Step 5: Re-run tests**

Run: `cd aquillm && pytest lib/llm/tests/test_lm_lingua2_adapter.py -q`  
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add aquillm/lib/llm/optimizations/lm_lingua2_adapter.py aquillm/lib/llm/tests/test_lm_lingua2_adapter.py requirements.txt
git commit -m "feat(llm): add LM-Lingua2 adapter with fail-open safeguards"
```

### Task 4: Wire compression into full RAG answer-provider paths

**Files:**
- Modify: `aquillm/lib/llm/providers/base.py`
- Modify: `aquillm/lib/llm/providers/openai.py`
- Modify: `aquillm/lib/llm/providers/claude.py`
- Modify: `aquillm/lib/llm/providers/gemini.py`
- Test: `aquillm/apps/chat/tests/test_messages.py`
- Create: `aquillm/lib/llm/tests/test_openai_prompt_compression.py`
- Create: `aquillm/lib/llm/tests/test_claude_prompt_compression.py`
- Create: `aquillm/lib/llm/tests/test_gemini_prompt_compression.py`

- [ ] **Step 1: Write failing integration tests**

```python
def test_openai_provider_compresses_large_prompt_when_enabled(...):
    ...
```

```python
def test_claude_provider_compresses_large_prompt_when_enabled(...):
    ...
```

```python
def test_gemini_provider_compresses_large_prompt_when_enabled(...):
    ...
```

- [ ] **Step 2: Run tests to verify fail**

Run: `cd aquillm && pytest lib/llm/tests/test_openai_prompt_compression.py lib/llm/tests/test_claude_prompt_compression.py lib/llm/tests/test_gemini_prompt_compression.py apps/chat/tests/test_messages.py -q`  
Expected: FAIL.

- [ ] **Step 3: Implement minimal wiring**

Rules:
- Apply only when `TOKEN_EFFICIENCY_ENABLED=1` and `LM_LINGUA2_ENABLED=1`.
- Respect policy gates from `token_efficiency.py` for all providers.
- Keep system prompt intact; compress only user/assistant history payload text.
- Preserve tool JSON payloads and message roles.
- On compression failure, send original prompt unchanged (fail-open).
- Keep provider-specific request/response mappings unchanged except prompt text input.

- [ ] **Step 4: Re-run tests**

Run: `cd aquillm && pytest lib/llm/tests/test_openai_prompt_compression.py lib/llm/tests/test_claude_prompt_compression.py lib/llm/tests/test_gemini_prompt_compression.py apps/chat/tests/test_messages.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aquillm/lib/llm/providers/base.py aquillm/lib/llm/providers/openai.py aquillm/lib/llm/providers/claude.py aquillm/lib/llm/providers/gemini.py aquillm/lib/llm/tests/test_openai_prompt_compression.py aquillm/lib/llm/tests/test_claude_prompt_compression.py aquillm/lib/llm/tests/test_gemini_prompt_compression.py aquillm/apps/chat/tests/test_messages.py
git commit -m "feat(llm): integrate LM-Lingua2 compression across RAG answer providers"
```

---

## Chunk 3: LMCache Runtime Integration + KVTC Overlay (Optional, Deployment-First)

### Task 5: Add LMCache env contracts and vLLM startup plumbing for RAG services

**Files:**
- Modify: `deploy/scripts/vllm_start.sh`
- Modify: `.env.example`
- Create: `aquillm/tests/integration/test_vllm_lmcache_args.py`

- [ ] **Step 1: Write failing startup parser tests**

```python
def test_lmcache_disabled_adds_no_kv_connector_args():
    ...
```

```python
def test_lmcache_enabled_adds_required_kv_args_for_chat_embed_rerank():
    ...
```

- [ ] **Step 2: Run tests to verify fail**

Run: `cd aquillm && pytest tests/integration/test_vllm_lmcache_args.py -q`  
Expected: FAIL.

- [ ] **Step 3: Implement startup wiring**

Implementation requirements:
- Add env toggles:
  - `LMCACHE_ENABLED`
  - `LMCACHE_HOST`
  - `LMCACHE_PORT`
  - `LMCACHE_CONFIG_JSON`
- Add service-scoped toggles:
  - `LMCACHE_CHAT_ENABLED`
  - `LMCACHE_EMBED_ENABLED`
  - `LMCACHE_RERANK_ENABLED`
- Add codec toggles:
  - `LMCACHE_CODEC_ENABLED`
  - `LMCACHE_CODEC` (`none|kvtc`)
  - `LMCACHE_CODEC_PROFILE` (`balanced|max_compression`)
- If enabled, append KV connector/offloading args through existing `cmd` assembly for the targeted service role.
- Reuse current auto-add for `--disable-hybrid-kv-cache-manager` when KV offloading args are present.
- Keep behavior unchanged when disabled.

- [ ] **Step 4: Re-run tests**

Run: `cd aquillm && pytest tests/integration/test_vllm_lmcache_args.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add deploy/scripts/vllm_start.sh .env.example aquillm/tests/integration/test_vllm_lmcache_args.py
git commit -m "feat(vllm): add optional LMCache startup args with env-driven controls"
```

### Task 6: Add compose-level LMCache service profile and vLLM dependency wiring (chat/embed/rerank)

**Files:**
- Modify: `deploy/compose/base.yml`
- Modify: `deploy/compose/development.yml`
- Modify: `deploy/compose/production.yml`
- Test: `aquillm/tests/integration/test_compose_multimodal_services.py`

- [ ] **Step 1: Write failing compose-contract tests**

```python
def test_compose_lmcache_profile_exists():
    ...
```

```python
def test_vllm_chat_embed_rerank_can_wire_lmcache_when_enabled():
    ...
```

- [ ] **Step 2: Run tests to verify fail**

Run: `cd aquillm && pytest tests/integration/test_compose_multimodal_services.py -q`  
Expected: FAIL.

- [ ] **Step 3: Implement compose wiring**

Add optional `lmcache` service and profile:
- conservative memory limits
- healthcheck
- no default hard dependency for non-LMCache deployments
- env wiring into `vllm`, `vllm_embed`, and `vllm_rerank` through `.env` flags

- [ ] **Step 4: Re-run tests**

Run: `cd aquillm && pytest tests/integration/test_compose_multimodal_services.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add deploy/compose/base.yml deploy/compose/development.yml deploy/compose/production.yml aquillm/tests/integration/test_compose_multimodal_services.py
git commit -m "feat(deploy): add optional LMCache compose profile and vLLM wiring"
```

---

### Task 6B: Implement KVTC-style codec layer on top of LMCache

**Files:**
- Create: `aquillm/lib/llm/cache/kvtc_codec.py`
- Create: `aquillm/lib/llm/cache/kvtc_calibration.py`
- Create: `aquillm/lib/llm/cache/types.py`
- Modify: `deploy/scripts/vllm_start.sh`
- Modify: `.env.example`
- Create: `aquillm/lib/llm/tests/test_kvtc_codec.py`
- Create: `aquillm/tests/integration/test_lmcache_codec_flags.py`

- [ ] **Step 1: Write failing codec tests**

```python
def test_kvtc_roundtrip_preserves_shape_and_dtype():
    ...
```

```python
def test_kvtc_reconstruction_error_bounded_under_profile():
    ...
```

```python
def test_codec_fail_open_falls_back_to_uncompressed_path():
    ...
```

- [ ] **Step 2: Run tests to verify fail**

Run: `cd aquillm && pytest lib/llm/tests/test_kvtc_codec.py tests/integration/test_lmcache_codec_flags.py -q`  
Expected: FAIL.

- [ ] **Step 3: Implement codec + calibration scaffolding**

Implementation requirements:
- Implement a KVTC-style pipeline for LMCache payloads:
  - transform/decorrelation stage
  - adaptive quantization stage
  - entropy-coding stage
- Add calibration artifact generation per service/model profile (chat/embed/rerank).
- Keep schema versioned so artifacts are invalidated safely on profile/model change.
- Fail-open to standard LMCache path on codec errors.

- [ ] **Step 4: Wire codec flags into runtime startup**

Implementation requirements:
- `vllm_start.sh` exposes codec flags through connector args/env passthrough.
- Disabled by default.
- Service-scoped enablement supported (`chat/embed/rerank`).

- [ ] **Step 5: Re-run tests**

Run: `cd aquillm && pytest lib/llm/tests/test_kvtc_codec.py tests/integration/test_lmcache_codec_flags.py -q`  
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add aquillm/lib/llm/cache/kvtc_codec.py aquillm/lib/llm/cache/kvtc_calibration.py aquillm/lib/llm/cache/types.py deploy/scripts/vllm_start.sh .env.example aquillm/lib/llm/tests/test_kvtc_codec.py aquillm/tests/integration/test_lmcache_codec_flags.py
git commit -m "feat(lmcache): add KVTC-style codec overlay with fail-open fallback"
```

---

## Chunk 4: Observability, Rollout Guardrails, and Verification

### Task 7: Add token-efficiency metrics/logging and rollout toggles

**Files:**
- Modify: `aquillm/lib/llm/providers/base.py`
- Modify: `aquillm/lib/llm/providers/openai.py`
- Modify: `aquillm/lib/llm/providers/claude.py`
- Modify: `aquillm/lib/llm/providers/gemini.py`
- Modify: `aquillm/lib/llm/cache/kvtc_codec.py`
- Modify: `aquillm/lib/llm/optimizations/lm_lingua2_adapter.py`
- Modify: `.env.example`
- Create: `aquillm/tests/integration/test_token_efficiency_flags.py`

- [ ] **Step 1: Write failing observability tests**

```python
def test_token_efficiency_disabled_bypasses_compression(monkeypatch):
    ...
```

- [ ] **Step 2: Run tests to verify fail**

Run: `cd aquillm && pytest tests/integration/test_token_efficiency_flags.py -q`  
Expected: FAIL.

- [ ] **Step 3: Implement instrumentation**

Add structured logs:
- `token_efficiency.compression.applied`
- `token_efficiency.compression.skipped`
- `token_efficiency.compression.error`
- `token_efficiency.compression.ratio`
- `token_efficiency.provider` (`openai|claude|gemini`)
- `token_efficiency.pipeline_stage` (`chat_answer|post_tool_synthesis`)
- `lmcache.codec.encode_ratio`
- `lmcache.codec.encode_latency_ms`
- `lmcache.codec.decode_latency_ms`
- `lmcache.codec.fallback_count`

Rules:
- Do not log raw prompt content.
- Only log token counts, ratios, and reason codes.

- [ ] **Step 4: Re-run tests**

Run: `cd aquillm && pytest tests/integration/test_token_efficiency_flags.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aquillm/lib/llm/providers/base.py aquillm/lib/llm/providers/openai.py aquillm/lib/llm/providers/claude.py aquillm/lib/llm/providers/gemini.py aquillm/lib/llm/cache/kvtc_codec.py aquillm/lib/llm/optimizations/lm_lingua2_adapter.py .env.example aquillm/tests/integration/test_token_efficiency_flags.py
git commit -m "chore(observability): add cross-provider token-efficiency and LMCache codec instrumentation"
```

### Task 8: Verification and rollout notes

**Files:**
- Create: `docs/roadmap/plans/superseded/2026-03-22-rag-token-efficiency-enhancements-rollout-notes.md`
- Modify: `README.md`

- [ ] **Step 1: Run targeted test suites**

Run:
- `cd aquillm && pytest lib/llm/tests/test_token_efficiency_policy.py -q`
- `cd aquillm && pytest lib/llm/tests/test_lm_lingua2_adapter.py -q`
- `cd aquillm && pytest lib/llm/tests/test_openai_prompt_compression.py -q`
- `cd aquillm && pytest lib/llm/tests/test_claude_prompt_compression.py -q`
- `cd aquillm && pytest lib/llm/tests/test_gemini_prompt_compression.py -q`
- `cd aquillm && pytest tests/integration/test_vllm_lmcache_args.py -q`
- `cd aquillm && pytest lib/llm/tests/test_kvtc_codec.py -q`
- `cd aquillm && pytest tests/integration/test_lmcache_codec_flags.py -q`
- `cd aquillm && pytest tests/integration/test_token_efficiency_flags.py -q`

Expected: PASS.

- [ ] **Step 2: Run broader regression smoke**

Run: `cd aquillm && pytest apps/chat/tests lib/llm/tests tests/integration -q --tb=short`  
Expected: PASS.

- [ ] **Step 3: Write rollout notes**

Document:
- recommended flag rollout order
- kill-switch instructions (`TOKEN_EFFICIENCY_ENABLED=0`, `LM_LINGUA2_ENABLED=0`, `LMCACHE_ENABLED=0`, `LMCACHE_CODEC_ENABLED=0`)
- success metrics and failure signals
- rollback procedure with zero schema changes

- [ ] **Step 4: Commit**

```bash
git add docs/roadmap/plans/superseded/2026-03-22-rag-token-efficiency-enhancements-rollout-notes.md README.md
git commit -m "docs(token-efficiency): add RAG token-efficiency rollout guide"
```

---

## Suggested Rollout Defaults

- `TOKEN_EFFICIENCY_ENABLED=0` (enable after validation)
- `LM_LINGUA2_ENABLED=0` (enable for canary users first)
- `LM_LINGUA2_TARGET_RATIO=0.65`
- `LM_LINGUA2_MIN_PROMPT_TOKENS=1800`
- `LM_LINGUA2_SKIP_WITH_TOOLS=1`
- `LMCACHE_ENABLED=0` (enable only with validated vLLM connector args)
- `LMCACHE_CHAT_ENABLED=0`
- `LMCACHE_EMBED_ENABLED=0`
- `LMCACHE_RERANK_ENABLED=0`
- `LMCACHE_CODEC_ENABLED=0`
- `LMCACHE_CODEC=kvtc`
- `LMCACHE_CODEC_PROFILE=balanced`

---

## Exit Gate

- [ ] No regression in chat/tool-call correctness tests.
- [ ] Compression policy is validated across OpenAI, Claude, and Gemini RAG answer paths.
- [ ] Compression path is fail-open and fully disableable by env.
- [ ] LMCache integration is optional, deployment-safe, and validated for `vllm`, `vllm_embed`, and `vllm_rerank`.
- [ ] KVTC-on-LMCache codec path is fail-open, versioned, and bounded by quality/error thresholds.
- [ ] Logs/metrics show token savings and no spike in failed completions.
- [ ] Roadmap and rollout docs are updated to reflect dependency sequencing.

---

**Plan complete and saved to `docs/roadmap/plans/superseded/2026-03-22-rag-token-efficiency-enhancements.md`.**


