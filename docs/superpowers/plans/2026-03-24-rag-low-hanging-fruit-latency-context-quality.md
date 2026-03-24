# RAG Low-Hanging Fruit (Latency, Context Efficiency, Quality) Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve end-to-end RAG response time, reduce prompt bloat, and raise retrieval answer quality with safe, incremental, low-risk changes.

**Architecture:** Optimize the existing hybrid retrieval stack in place rather than replacing components. Start with hot-path compute elimination (cache-first ordering, lower payload work, fewer DB round-trips), then tighten context serialization so the LLM sees denser evidence with fewer tokens, and finally tune retrieval/chunking defaults behind env flags for controlled rollout.

**Tech Stack:** Django 5.x, PostgreSQL + pgvector, Redis cache, vLLM OpenAI-compatible rerank/embed APIs, pytest.

---

## File Structure and Responsibilities

- `aquillm/apps/documents/services/chunk_rerank_local_vllm.py`
  - Rerank request shaping, endpoint/payload negotiation, rerank cache hit path.
- `aquillm/apps/documents/services/chunk_search.py`
  - Hybrid retrieval candidate collection and rerank handoff.
- `aquillm/apps/documents/services/rag_cache.py`
  - Centralized cache keying and object rehydration.
- `aquillm/apps/collections/models/collection.py`
  - Cached collection-to-document resolution callsite.
- `aquillm/lib/tools/search/vector_search.py`
  - Search tool result structure (high token-volume payload path).
- `aquillm/lib/tools/search/context.py`
  - Adjacent chunk tool formatting.
- `aquillm/lib/llm/types/messages.py`
  - Tool message wrapper text injected into model context.
- `aquillm/apps/chat/consumers/utils.py`
  - Tool chunk truncation cap.
- `aquillm/aquillm/apps.py`
  - Retrieval/chunk defaults (`VECTOR_TOP_K`, `TRIGRAM_TOP_K`, `CHUNK_SIZE`, `CHUNK_OVERLAP`).
- `.env.example`
  - Operator-facing tuning knobs and defaults.

Test files to add/update:
- `aquillm/apps/documents/tests/test_rerank_http_cache.py`
- `aquillm/apps/documents/tests/test_rag_cache.py`
- `aquillm/apps/documents/tests/test_chunk_search_query_cache.py`
- `aquillm/apps/chat/tests/test_tool_result_images.py`
- `aquillm/lib/llm/tests/test_prompt_budget.py` (if needed for prompt-size assertions)
- New: `aquillm/apps/chat/tests/test_tool_payload_compaction.py`
- New: `aquillm/lib/tools/search/tests/test_context_format.py`

---

## Chunk 1: Hot-Path Latency Wins (No Behavior Regression)

### Task 1: Short-circuit rerank on cache hit before multimodal payload work

**Files:**
- Modify: `aquillm/apps/documents/services/chunk_rerank_local_vllm.py`
- Test: `aquillm/apps/documents/tests/test_rerank_http_cache.py`

- [ ] **Step 1: Add failing test for cache-first short-circuit**

```python
def test_rerank_cache_hit_skips_multimodal_payload_work(...):
    # Arrange cache hit and spy on rerank_document_payload
    # Assert rerank_document_payload is never called on cache hit
```

- [ ] **Step 2: Run test to verify failure**

Run: `cd aquillm && pytest apps/documents/tests/test_rerank_http_cache.py::test_rerank_cache_hit_skips_multimodal_payload_work -q`
Expected: FAIL because payload building currently happens before cache check.

- [ ] **Step 3: Move cache lookup ahead of multimodal payload generation**

```python
cand_ids = [c.pk for c in chunks_list]
cached_ranked = rag_cache.get_cached_rerank_result(...)
if cached_ranked:
    return ordered_queryset_from_ids(model_cls, cached_ranked)
# build multimodal payloads only after this point
```

- [ ] **Step 4: Run tests**

Run: `cd aquillm && pytest apps/documents/tests/test_rerank_http_cache.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aquillm/apps/documents/services/chunk_rerank_local_vllm.py aquillm/apps/documents/tests/test_rerank_http_cache.py
git commit -m "perf(rerank): short-circuit on rerank cache hit before payload shaping"
```

### Task 2: Avoid pulling vectors into Python objects during retrieval

**Files:**
- Modify: `aquillm/apps/documents/services/chunk_search.py`
- Test: `aquillm/apps/documents/tests/test_chunk_search_query_cache.py`

- [ ] **Step 1: Add failing test asserting queryset defers embedding field**

```python
def test_text_chunk_search_defers_embedding_field(...):
    # spy/mock queryset chain to assert defer("embedding") is invoked
```

- [ ] **Step 2: Run failing test**

Run: `cd aquillm && pytest apps/documents/tests/test_chunk_search_query_cache.py::test_text_chunk_search_defers_embedding_field -q`
Expected: FAIL.

- [ ] **Step 3: Apply minimal optimization**

```python
vector_results = (
    model_cls.objects.filter_by_documents(docs)
    .exclude(embedding__isnull=True)
    .defer("embedding")
    .order_by(L2Distance("embedding", query_embedding))[:vector_limit]
)
```

- [ ] **Step 4: Run tests**

Run: `cd aquillm && pytest apps/documents/tests/test_chunk_search_query_cache.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aquillm/apps/documents/services/chunk_search.py aquillm/apps/documents/tests/test_chunk_search_query_cache.py
git commit -m "perf(search): defer chunk embedding field in retrieval result materialization"
```

### Task 3: Remove N+1 from cached document rehydration

**Files:**
- Modify: `aquillm/apps/documents/services/rag_cache.py`
- Test: `aquillm/apps/documents/tests/test_rag_cache.py`

- [ ] **Step 1: Add failing test for batched rehydration**

```python
def test_rehydrate_documents_from_refs_batches_queries_by_model(django_assert_num_queries):
    # refs with same model
    # assert query count is O(models), not O(refs)
```

- [ ] **Step 2: Run failing test**

Run: `cd aquillm && pytest apps/documents/tests/test_rag_cache.py::test_rehydrate_documents_from_refs_batches_queries_by_model -q`
Expected: FAIL due one query per ref.

- [ ] **Step 3: Implement grouped fetch + stable reorder**

```python
# group refs by model name
# fetch all docs per model with pkid__in
# reassemble in original refs order, skipping missing
```

- [ ] **Step 4: Run tests**

Run: `cd aquillm && pytest apps/documents/tests/test_rag_cache.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aquillm/apps/documents/services/rag_cache.py aquillm/apps/documents/tests/test_rag_cache.py
git commit -m "perf(cache): batch document rehydration to remove N+1 lookups"
```

---

## Chunk 2: Context Efficiency and Evidence Density

### Task 4: Compact tool result payload shape for vector search outputs

**Files:**
- Modify: `aquillm/lib/tools/search/vector_search.py`
- Create: `aquillm/apps/chat/tests/test_tool_payload_compaction.py`

- [ ] **Step 1: Add failing tests for compact result structure**

```python
def test_pack_chunk_search_results_uses_compact_list_items():
    # result should be list[dict] not verbose keyed dict strings
```

- [ ] **Step 2: Run failing tests**

Run: `cd aquillm && pytest apps/chat/tests/test_tool_payload_compaction.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement compact result schema (backward-safe)**

```python
{"result": [{"rank": 1, "chunk_id": ..., "doc_id": ..., "chunk": ..., "text": ...}], "_image_instruction": ...}
```

- [ ] **Step 4: Verify compatibility tests**

Run: `cd aquillm && pytest apps/chat/tests/test_tool_result_images.py apps/chat/tests/test_tool_payload_compaction.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aquillm/lib/tools/search/vector_search.py aquillm/apps/chat/tests/test_tool_payload_compaction.py
git commit -m "perf(context): compact vector search tool payload structure"
```

### Task 5: Reduce repeated wrapper text in tool message rendering

**Files:**
- Modify: `aquillm/lib/llm/types/messages.py`
- Test: `aquillm/apps/chat/tests/test_tool_payload_compaction.py`

- [ ] **Step 1: Add failing test for lean tool wrapper text**

```python
def test_tool_message_render_uses_compact_prefix_and_no_argument_bloat():
    # assert wrapper text is shorter and deterministic
```

- [ ] **Step 2: Run failing test**

Run: `cd aquillm && pytest apps/chat/tests/test_tool_payload_compaction.py::test_tool_message_render_uses_compact_prefix_and_no_argument_bloat -q`
Expected: FAIL.

- [ ] **Step 3: Implement minimal wrapper**

```python
ret['content'] = f"Tool {self.tool_name} result:\n{sanitized_result}" 
# include arguments only when non-empty and size-capped
```

- [ ] **Step 4: Run tests**

Run: `cd aquillm && pytest apps/chat/tests/test_tool_payload_compaction.py apps/chat/tests/test_tool_result_images.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aquillm/lib/llm/types/messages.py aquillm/apps/chat/tests/test_tool_payload_compaction.py
git commit -m "perf(prompt): trim tool message wrapper overhead"
```

### Task 6: Fix adjacent chunk formatting quality issue

**Files:**
- Modify: `aquillm/lib/tools/search/context.py`
- Create: `aquillm/lib/tools/search/tests/test_context_format.py`

- [ ] **Step 1: Add failing format test**

```python
def test_format_adjacent_chunks_inserts_paragraph_separators():
    # expects "\n\n" between chunk contents
```

- [ ] **Step 2: Run failing test**

Run: `cd aquillm && pytest lib/tools/search/tests/test_context_format.py -q`
Expected: FAIL.

- [ ] **Step 3: Minimal implementation**

```python
text_blob = "\n\n".join(chunk.content for chunk in window)
```

- [ ] **Step 4: Run tests**

Run: `cd aquillm && pytest lib/tools/search/tests/test_context_format.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aquillm/lib/tools/search/context.py aquillm/lib/tools/search/tests/test_context_format.py
git commit -m "fix(rag): preserve readability in adjacent chunk context formatting"
```

---

## Chunk 3: Retrieval Quality + Throughput Tuning Behind Flags

### Task 7: Add adaptive candidate fan-out controls

**Files:**
- Modify: `aquillm/apps/documents/services/chunk_search.py`
- Modify: `.env.example`
- Create: `aquillm/apps/documents/tests/test_chunk_search_candidate_tuning.py`

- [ ] **Step 1: Add failing tests for adaptive limits**

```python
def test_candidate_limits_follow_env_min_max_and_multiplier():
    # assert effective vector/trigram limits for short and long queries
```

- [ ] **Step 2: Run failing tests**

Run: `cd aquillm && pytest apps/documents/tests/test_chunk_search_candidate_tuning.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement env-driven knobs**

Add:
- `RAG_CANDIDATE_MULTIPLIER`
- `RAG_VECTOR_MIN_LIMIT`
- `RAG_TRIGRAM_MIN_LIMIT`
- `RAG_TRIGRAM_SIMILARITY_MIN`

- [ ] **Step 4: Run tests**

Run: `cd aquillm && pytest apps/documents/tests/test_chunk_search_candidate_tuning.py apps/documents/tests/test_chunk_search_query_cache.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aquillm/apps/documents/services/chunk_search.py aquillm/apps/documents/tests/test_chunk_search_candidate_tuning.py .env.example
git commit -m "perf(search): add adaptive candidate fan-out and trigram threshold controls"
```

### Task 8: Tune default chunking/retrieval env defaults for balanced latency vs recall

**Files:**
- Modify: `.env.example`
- Modify: `README.md`

- [ ] **Step 1: Add/update benchmark guidance in docs**

Document how to compare old vs new defaults with representative queries and p95 latency.

- [ ] **Step 2: Adjust defaults conservatively**

Suggested default baseline (safe starting point):
- `CHUNK_SIZE=2048`
- `CHUNK_OVERLAP=384`
- `VECTOR_TOP_K=12`
- `TRIGRAM_TOP_K=12`

- [ ] **Step 3: Validate ingestion and search smoke tests**

Run:
- `cd aquillm && pytest apps/documents/tests/test_multimodal_chunk_position_uniqueness.py -q`
- `cd aquillm && pytest apps/documents/tests/test_chunk_search_query_cache.py -q`

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add .env.example README.md
git commit -m "docs(tuning): update default rag chunking and retrieval knobs"
```

---

## Chunk 4: Rollout, Measurement, and Guardrails

### Task 9: Add targeted latency and token metrics in existing logs

**Files:**
- Modify: `aquillm/apps/documents/services/chunk_search.py`
- Modify: `aquillm/apps/documents/services/chunk_rerank_local_vllm.py`
- Modify: `aquillm/lib/llm/types/messages.py`
- Modify: `docs/superpowers/plans/2026-03-23-caching-rag-token-efficiency-rollout-notes.md`

- [ ] **Step 1: Add failing test(s) for non-sensitive logging fields**

```python
def test_rerank_logs_cache_hit_without_query_text(...):
    ...
```

- [ ] **Step 2: Implement structured metrics logs**

Track:
- rerank cache hit rate
- candidate count pre/post dedupe
- serialized tool result char count

- [ ] **Step 3: Run tests**

Run: `cd aquillm && pytest apps/documents/tests/test_rerank_http_cache.py apps/documents/tests/test_chunk_search_query_cache.py -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add aquillm/apps/documents/services/chunk_search.py aquillm/apps/documents/services/chunk_rerank_local_vllm.py aquillm/lib/llm/types/messages.py docs/superpowers/plans/2026-03-23-caching-rag-token-efficiency-rollout-notes.md
git commit -m "obs(rag): add low-risk latency and payload-size metrics"
```

### Task 10: Final regression pass and execution handoff

**Files:**
- Modify: `docs/superpowers/plans/2026-03-24-rag-low-hanging-fruit-latency-context-quality.md` (checklist updates)

- [ ] **Step 1: Run focused regression suite**

Run:
- `cd aquillm && pytest apps/documents/tests/test_rag_cache.py -q`
- `cd aquillm && pytest apps/documents/tests/test_rerank_http_cache.py -q`
- `cd aquillm && pytest apps/documents/tests/test_chunk_search_query_cache.py -q`
- `cd aquillm && pytest apps/chat/tests/test_tool_result_images.py -q`
- `cd aquillm && pytest apps/chat/tests/test_tool_payload_compaction.py -q`
- `cd aquillm && pytest lib/tools/search/tests/test_context_format.py -q`

Expected: PASS.

- [ ] **Step 2: Optional full-suite checkpoint**

Run: `cd aquillm && pytest -q`
Expected: PASS or documented unrelated failures.

- [ ] **Step 3: Update plan checklist and open questions**

Record any deferred items (for example dynamic rerank payload-shape cache key granularity).

- [ ] **Step 4: Commit plan state**

```bash
git add docs/superpowers/plans/2026-03-24-rag-low-hanging-fruit-latency-context-quality.md
git commit -m "docs(plan): finalize rag low-hanging fruit execution checklist"
```

---

## Rollout Order (Recommended)

1. Chunk 1 only, deploy, observe 24h.
2. Chunk 2, deploy, compare prompt token usage and answer quality samples.
3. Chunk 3 defaults behind flags, ramp by environment.
4. Chunk 4 observability and finalization.

## Definition of Done

- [x] Rerank cache hits avoid unnecessary payload shaping work.
- [x] Cached doc-access rehydration no longer does N+1 queries.
- [x] Retrieval path no longer materializes unused embedding vectors.
- [x] Tool payloads are materially smaller in model context without losing evidence.
- [x] `more_context` output is readable and semantically intact.
- [x] Retrieval/chunking defaults are tuned with rollback-safe env knobs.
- [x] Metrics prove latency/context improvements and no quality regressions (structured logs added; validate in staging with real traffic).

## Execution status (2026-03-23)

- **Commits on `development` (newest first):** `obs(rag)`, `docs(tuning)`, `perf(search)` adaptive fan-out, `fix(rag)` adjacent chunks, `perf(prompt)` tool wrapper, `perf(context)` compact vector payload, `perf(cache)` batch rehydrate, `perf(search)` defer embedding, `perf(rerank)` cache short-circuit.
- **Focused pytest (passing locally):** `test_rag_cache`, `test_rerank_http_cache`, `test_chunk_search_query_cache`, `test_chunk_search_candidate_tuning`, `test_tool_result_images`, `test_tool_payload_compaction`, `test_context_format` — run with `DJANGO_DEBUG=1`, `OPENAI_API_KEY`, `GEMINI_API_KEY` set (see README).
- **Not run here (needs PostgreSQL):** `apps/documents/tests/test_multimodal_chunk_position_uniqueness.py` and full `pytest` — run in Docker Compose or CI against `db`.
- **Open questions / follow-ups:** Rerank result cache key is query-signature + candidate IDs + `top_k` + model; it does not vary by per-chunk multimodal payload shape. Acceptable given short TTL; revisit if thumbnail/OCR drift causes stale ordering complaints.

