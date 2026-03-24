# RAG cache and token-efficiency rollout notes

This note accompanies the implementation in `2026-03-23-caching-rag-token-efficiency-optimization-refresh.md`.

## Observability (safe logging)

- RAG cache: `apps.documents.services.rag_cache` logs **hits** at INFO (`rag_cache.* hit`) and **misses** at DEBUG (`rag_cache.* miss`). No query text, embeddings, or base64 payloads are logged.
- Prompt budget: `lib.llm.utils.prompt_budget` logs a single INFO line when preflight trim runs: estimated input tokens before/after and effective `max_tokens` — no message bodies.
- LM-Lingua2: `lib.llm.optimizations.lm_lingua2_adapter` logs only role and character counts when compression succeeds; failures log a short warning.

## Suggested enablement order

1. **Django cache backend**: Set `DJANGO_CACHE_REDIS_URL` in multi-worker deployments so all processes share the same cache. Leave unset for LocMem (tests and single-process dev).
2. **RAG caches**: Set `RAG_CACHE_ENABLED=1`. Tune TTLs (`RAG_*_TTL_SECONDS`) if permission or content freshness requirements are stricter than defaults.
3. **Cross-provider prompt budget**: Set `TOKEN_EFFICIENCY_ENABLED=1` and `PROMPT_BUDGET_CONTEXT_LIMIT` (or rely on `OPENAI_CONTEXT_LIMIT` / `VLLM_MAX_MODEL_LEN`). Claude and Gemini use the same preflight trim logic as OpenAI-shaped estimates.
4. **LM-Lingua2**: Install `llmlingua`, set `LM_LINGUA2_ENABLED=1` only after validating quality on representative chats. Default remains off.
5. **LMCache / vLLM**: Set `LMCACHE_ENABLED=1` and supply vLLM-compatible flags via `LMCACHE_EXTRA_ARGS` (parsed like `VLLM_EXTRA_ARGS`). Existing logic still appends `--disable-hybrid-kv-cache-manager` when KV offloading args are present.

## Rollback

- **RAG**: `RAG_CACHE_ENABLED=0` — code paths become no-ops; no cache reads or writes.
- **Prompt budget**: `TOKEN_EFFICIENCY_ENABLED=0` — Claude/Gemini skip preflight trim.
- **LM-Lingua2**: `LM_LINGUA2_ENABLED=0` — OpenAI path skips compression; any runtime error already fails open to the original text.
- **LMCache**: `LMCACHE_ENABLED=0` and remove `LMCACHE_EXTRA_ARGS`; restart vLLM.

## Regression tests (targeted)

From the repository root (with `SECRET_KEY`, API keys, and OAuth placeholders as in README):

```bash
pytest aquillm/tests/integration/test_cache_settings_flags.py \
  aquillm/tests/integration/test_vllm_lmcache_plumbing.py \
  aquillm/apps/documents/tests/test_rag_cache.py \
  aquillm/apps/documents/tests/test_chunk_search_query_cache.py \
  aquillm/apps/chat/tests/test_tool_wiring_doc_access_cache.py \
  aquillm/apps/documents/tests/test_document_lookup_and_image_cache.py \
  aquillm/apps/documents/tests/test_rerank_http_cache.py \
  aquillm/lib/llm/tests/test_prompt_budget.py \
  aquillm/lib/llm/tests/test_claude_prompt_budget.py \
  aquillm/lib/llm/tests/test_gemini_prompt_budget.py \
  aquillm/lib/llm/tests/test_lm_lingua2_adapter.py \
  aquillm/apps/chat/tests/test_multimodal_messages.py -q
```

Full `aquillm/apps/documents/tests` requires PostgreSQL (pgvector) as today.
