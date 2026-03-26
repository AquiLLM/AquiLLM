# Caching and RAG Token-Efficiency Optimization Refresh Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce end-to-end RAG latency and token spend by adding safe retrieval caches plus provider-aware prompt budget controls, while preserving current fallback behavior.

**Architecture:** Keep the existing retrieval/rerank/prompt orchestration intact, then wrap repeated expensive operations with short-lived cache helpers and explicit feature flags. Build on current OpenAI-compatible context-overflow handling (`openai_tokens.py`, `openai_overflow.py`) and extend budget controls across provider paths. Keep all optimizations fail-open and reversible by env flags.

**Tech Stack:** Django 5.1, Redis, pgvector/Postgres, vLLM OpenAI-compatible endpoints, pytest.

---

## Current Codebase Baseline (2026-03-23)

- OpenAI-compatible prompt budgeting already exists (`aquillm/lib/llm/providers/openai_tokens.py`, `openai_overflow.py`, `openai.py`), including preflight trim, overflow retries, and image stripping.
- Tool payload token controls already exist:
  - tool text truncation (`aquillm/apps/chat/consumers/utils.py`)
  - base64 redaction for tool text (`aquillm/lib/llm/providers/image_context.py`)
  - compact fallback synthesis (`aquillm/lib/llm/providers/summary.py`, `complete_turn.py`)
- Shared Django cache config and RAG cache helpers are **not** implemented yet:
  - no `CACHES` block in `aquillm/aquillm/settings.py`
  - no retrieval cache helper module
  - no query-embedding/result cache layer in `chunk_search` / `chunk_rerank_local_vllm`
- LM-Lingua2 and LMCache wiring are not yet present in code or env contracts.

---

## Chunk 1: Cache Foundation and Safety Rails

### Task 1: Add explicit cache backend + feature flags

**Files:**
- Modify: `aquillm/aquillm/settings.py`
- Modify: `.env.example`
- Create: `aquillm/tests/integration/test_cache_settings_flags.py`

- [ ] **Step 1: Add failing tests for cache contracts**
- [ ] **Step 2: Implement `CACHES` and env-backed flags**

Add settings with conservative defaults:
- `RAG_CACHE_ENABLED` (default false)
- `RAG_QUERY_EMBED_TTL_SECONDS` (default 300)
- `RAG_DOC_ACCESS_TTL_SECONDS` (default 60)
- `RAG_IMAGE_DATA_URL_TTL_SECONDS` (default 120)
- `RAG_RERANK_RESULT_TTL_SECONDS` (default 45)
- `RAG_RERANK_CAPABILITY_TTL_SECONDS` (default 900)

Use Redis-backed cache when available; keep local/test fallback deterministic.

- [ ] **Step 3: Re-run tests and commit**

Run: `cd aquillm && pytest tests/integration/test_cache_settings_flags.py -q`

### Task 2: Introduce reusable RAG cache helper module

**Files:**
- Create: `aquillm/apps/documents/services/rag_cache.py`
- Create: `aquillm/apps/documents/tests/test_rag_cache.py`

- [ ] **Step 1: Add failing tests for cache key normalization and fail-open behavior**
- [ ] **Step 2: Implement helper functions**

Implement:
- normalized cache-key builder (stable hash for long keys)
- safe `cache_get/cache_set` wrappers (swallow backend failures)
- helpers for query-embedding, rerank capability, rerank result signatures

- [ ] **Step 3: Re-run tests and commit**

Run: `cd aquillm && pytest apps/documents/tests/test_rag_cache.py -q`

---

## Chunk 2: Retrieval Path Caching

### Task 3: Cache repeated query embeddings in hybrid search

**Files:**
- Modify: `aquillm/apps/documents/services/chunk_search.py`
- Modify: `aquillm/apps/documents/services/rag_cache.py`
- Create: `aquillm/apps/documents/tests/test_chunk_search_query_cache.py`

- [ ] **Step 1: Add failing tests (same query/model/input_type should embed once per TTL)**
- [ ] **Step 2: Wrap `get_embedding(query)` with cache helper**
- [ ] **Step 3: Verify no behavior change on cache miss/error**
- [ ] **Step 4: Re-run tests and commit**

Run: `cd aquillm && pytest apps/documents/tests/test_chunk_search_query_cache.py -q`

### Task 4: Cache user collection document resolution in chat tools

**Files:**
- Modify: `aquillm/apps/chat/services/tool_wiring/documents.py`
- Modify: `aquillm/apps/collections/models/collection.py`
- Create: `aquillm/apps/chat/tests/test_tool_wiring_doc_access_cache.py`

- [ ] **Step 1: Add failing tests for repeated `vector_search` and `document_ids` lookups**
- [ ] **Step 2: Add cache wrapper keyed by `(user_id, collection_ids, permission)`**
- [ ] **Step 3: Ensure permission changes naturally expire via short TTL**
- [ ] **Step 4: Re-run tests and commit**

Run: `cd aquillm && pytest apps/chat/tests/test_tool_wiring_doc_access_cache.py -q`

### Task 5: Cache document lookup and image data-url generation

**Files:**
- Modify: `aquillm/apps/documents/models/document.py`
- Modify: `aquillm/apps/documents/models/chunks.py`
- Modify: `aquillm/apps/documents/services/image_payloads.py`
- Modify: `aquillm/apps/documents/services/chunk_embeddings.py`
- Create: `aquillm/apps/documents/tests/test_document_lookup_and_image_cache.py`

- [ ] **Step 1: Add failing tests for repeated `Document.get_by_id` scans and repeated storage reads**
- [ ] **Step 2: Add lookup cache by `doc_id` and image payload cache by `(doc_id, image_file_name)`**
- [ ] **Step 3: Keep existing fallback behavior for missing files/storage errors**
- [ ] **Step 4: Re-run tests and commit**

Run: `cd aquillm && pytest apps/documents/tests/test_document_lookup_and_image_cache.py -q`

### Task 6: Cache rerank capability negotiation and short-lived rerank outputs

**Files:**
- Modify: `aquillm/apps/documents/services/chunk_rerank_local_vllm.py`
- Modify: `aquillm/apps/documents/services/chunk_rerank.py`
- Modify: `aquillm/apps/documents/services/rag_cache.py`
- Create: `aquillm/apps/documents/tests/test_rerank_http_cache.py`

- [ ] **Step 1: Add failing tests for repeated probe/request paths**
- [ ] **Step 2: Cache endpoint/payload capability probe outcome per `(base_url, model)`**
- [ ] **Step 3: Cache ranked IDs per `(query_signature, candidate_ids, top_k, model)`**
- [ ] **Step 4: Re-run tests and commit**

Run: `cd aquillm && pytest apps/documents/tests/test_rerank_http_cache.py -q`

---

## Chunk 3: Cross-Provider Token-Efficiency Controls

### Task 7: Add shared prompt budget policy module (provider-agnostic)

**Files:**
- Create: `aquillm/lib/llm/utils/prompt_budget.py`
- Create: `aquillm/lib/llm/tests/test_prompt_budget.py`
- Modify: `.env.example`

- [ ] **Step 1: Add failing tests for budget policies and feature flags**
- [ ] **Step 2: Implement policy helpers**

Implement helpers for:
- `TOKEN_EFFICIENCY_ENABLED`
- min/max thresholds for compaction/compression
- safe budget computation for history trimming before provider calls

- [ ] **Step 3: Re-run tests and commit**

Run: `cd aquillm && pytest lib/llm/tests/test_prompt_budget.py -q`

### Task 8: Extend token-budget controls to Claude and Gemini paths

**Files:**
- Modify: `aquillm/lib/llm/providers/claude.py`
- Modify: `aquillm/lib/llm/providers/gemini.py`
- Modify: `aquillm/lib/llm/providers/base.py`
- Create: `aquillm/lib/llm/tests/test_claude_prompt_budget.py`
- Create: `aquillm/lib/llm/tests/test_gemini_prompt_budget.py`

- [ ] **Step 1: Add failing tests for oversize history handling**
- [ ] **Step 2: Apply shared policy before provider API invocation**
- [ ] **Step 3: Preserve tool-call semantics and message role ordering**
- [ ] **Step 4: Re-run tests and commit**

Run: `cd aquillm && pytest lib/llm/tests/test_claude_prompt_budget.py lib/llm/tests/test_gemini_prompt_budget.py -q`

### Task 9: Optional LM-Lingua2 integration behind hard flags

**Files:**
- Create: `aquillm/lib/llm/optimizations/lm_lingua2_adapter.py`
- Modify: `aquillm/lib/llm/providers/openai.py`
- Modify: `aquillm/lib/llm/providers/claude.py`
- Modify: `aquillm/lib/llm/providers/gemini.py`
- Modify: `requirements.txt`
- Create: `aquillm/lib/llm/tests/test_lm_lingua2_adapter.py`

- [ ] **Step 1: Keep default disabled (`LM_LINGUA2_ENABLED=0`)**
- [ ] **Step 2: Compress only long plain-text history sections**
- [ ] **Step 3: Fail open (original prompt on any adapter failure)**
- [ ] **Step 4: Re-run tests and commit**

Run: `cd aquillm && pytest lib/llm/tests/test_lm_lingua2_adapter.py -q`

---

## Chunk 4: Deployment Wiring, Metrics, and Rollout

### Task 10: LMCache env/plumbing for vLLM startup and compose

**Files:**
- Modify: `deploy/scripts/vllm_start.sh`
- Modify: `deploy/compose/base.yml`
- Modify: `deploy/compose/development.yml`
- Modify: `deploy/compose/production.yml`
- Modify: `.env.example`
- Create: `aquillm/tests/integration/test_vllm_lmcache_plumbing.py`

- [ ] **Step 1: Add env contracts (`LMCACHE_ENABLED`, `LMCACHE_*`)**
- [ ] **Step 2: Conditionally append KV connector args in startup wrapper**
- [ ] **Step 3: Keep existing `--disable-hybrid-kv-cache-manager` safety behavior**
- [ ] **Step 4: Re-run tests and commit**

Run: `cd aquillm && pytest tests/integration/test_vllm_lmcache_plumbing.py -q`

### Task 11: Add metrics and rollout guide

**Files:**
- Modify: `aquillm/apps/documents/services/chunk_search.py`
- Modify: `aquillm/apps/documents/services/chunk_rerank_local_vllm.py`
- Modify: `aquillm/lib/llm/providers/openai.py`
- Modify: `aquillm/lib/llm/providers/claude.py`
- Modify: `aquillm/lib/llm/providers/gemini.py`
- Create: `docs/roadmap/plans/active/2026-03-23-caching-rag-token-efficiency-rollout-notes.md`
- Modify: `README.md`

- [ ] **Step 1: Log cache hit/miss counters and provider budget-trim decisions**
- [ ] **Step 2: Ensure logs never include raw prompt bodies or base64 payloads**
- [ ] **Step 3: Document staged rollout + rollback**
- [ ] **Step 4: Run targeted regression suite and commit**

Run:
- `cd aquillm && pytest apps/chat/tests/test_multimodal_messages.py -q`
- `cd aquillm && pytest apps/documents/tests -q`
- `cd aquillm && pytest tests/integration/test_cache_settings_flags.py tests/integration/test_vllm_lmcache_plumbing.py -q`

---

## Execution Order

1. Chunk 1 (settings/helper)  
2. Chunk 2 (retrieval caches)  
3. Chunk 3 (cross-provider token controls)  
4. Chunk 4 (deployment + observability + rollout docs)

---

## Definition of Done

- [ ] Shared cache backend is configured and controlled by explicit env flags.
- [ ] Query embedding, doc access, image payload, and rerank probes/results have bounded-TTL caches.
- [ ] OpenAI token handling remains stable; Claude/Gemini gain equivalent budget guardrails.
- [ ] LM-Lingua2 and LMCache remain optional and fail-open with one-flag rollback.
- [ ] Tests cover cache correctness, fallback behavior, and no-regression chat behavior.
- [ ] Rollout notes document metrics, enablement sequence, and rollback steps.

---

**This refresh supersedes:**
- `docs/roadmap/plans/superseded/2026-03-22-multimodal-rag-caching-latency-optimization.md`
- `docs/roadmap/plans/superseded/2026-03-22-rag-token-efficiency-enhancements.md`

**Plan complete and saved to `docs/roadmap/plans/active/2026-03-23-caching-rag-token-efficiency-optimization-refresh.md`.**


