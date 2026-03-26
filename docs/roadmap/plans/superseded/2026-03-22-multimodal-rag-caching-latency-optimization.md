# Multimodal RAG Caching and Latency Optimization Implementation Plan

> **Status (2026-03-23):** Superseded by `docs/roadmap/plans/active/2026-03-23-caching-rag-token-efficiency-optimization-refresh.md`, which reflects the current codebase baseline and updated execution order.

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce chat-time multimodal RAG latency by adding safe, low-risk caching at the highest-cost stages (query embedding, document access resolution, image payload generation, rerank path negotiation, and short-lived rerank outputs).

**Architecture:** Introduce a shared Redis-backed Django cache layer and small, explicit cache helpers in the documents/chat/memory paths. Keep existing retrieval logic and model choices intact, but add cache lookups around repeated work with short TTLs and feature flags for safe rollout. Preserve correctness by scoping cache keys to user/query/model/candidate-set signatures and by retaining fallback behavior on cache miss or cache failure.

**Tech Stack:** Django 5.1, Redis (existing service), Channels/Celery stack, pgvector/Postgres, vLLM rerank/embed services, pytest.

**Alignment Dependencies:** This plan is aligned to the architecture remediation baseline from:
- `docs/roadmap/plans/pending/2026-03-21-architecture-boundary-and-structural-remediation.md`
- `docs/roadmap/plans/pending/2026-03-21-architecture-remediation-commit-plan.md`

**Required remediation baseline before execution (recommended):**
- Commit 3 (`apps.documents.tasks.chunking`, `apps.documents.services.image_payloads`)
- Commit 6 (`lib.tools.search/*`, `lib.tools/documents/*`, `apps.chat.services.tool_wiring`)
- Commit 11 (ingestion API split under `apps/ingestion/views/api/*`)
- Commit 12 (`apps.documents.services.chunk_search/chunk_rerank/chunk_embeddings`)

**Compatibility fallback rule:** If a listed remediation commit is not yet landed, apply the cache change in current legacy files first, then move it during remediation commit adoption.

---

## Chunk 1: Cache Foundation and Query Embedding

### Task 1: Add shared Django cache configuration for retrieval caches

**Files:**
- Modify: `aquillm/aquillm/settings.py`
- Modify: `.env.example`
- Test: `aquillm/tests/integration/test_settings_security_flags.py`

- [ ] **Step 1: Write the failing test for cache settings defaults**

```python
def test_cache_defaults_present(settings):
    assert "default" in settings.CACHES
    assert settings.CACHES["default"]["BACKEND"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd aquillm && pytest tests/integration/test_settings_security_flags.py::test_cache_defaults_present -q`  
Expected: FAIL because `CACHES` is not configured.

- [ ] **Step 3: Add cache settings and env-driven knobs**

Add to settings:
- `CACHES` using Redis by default (`redis://redis:6379/1`) with `LocMemCache` fallback for local test/dev.
- `RAG_CACHE_ENABLED` and default TTL envs:
  - `RAG_EMBED_QUERY_TTL_SECONDS`
  - `RAG_DOC_ACCESS_TTL_SECONDS`
  - `RAG_IMAGE_DATA_URL_TTL_SECONDS`
  - `RAG_RERANK_RESULT_TTL_SECONDS`
  - `RAG_RERANK_CAPABILITY_TTL_SECONDS`

- [ ] **Step 4: Run test to verify it passes**

Run: `cd aquillm && pytest tests/integration/test_settings_security_flags.py::test_cache_defaults_present -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aquillm/aquillm/settings.py .env.example aquillm/tests/integration/test_settings_security_flags.py
git commit -m "feat(cache): add shared Django cache config and RAG TTL settings"
```

### Task 2: Add reusable query embedding cache helper and integrate search path

**Files:**
- Create: `aquillm/apps/documents/services/rag_cache.py`
- Modify (preferred, post-Commit 12): `aquillm/apps/documents/services/chunk_search.py`
- Modify (fallback, pre-Commit 12): `aquillm/apps/documents/models/chunks.py`
- Test: `aquillm/apps/documents/tests/test_query_embedding_cache.py`

- [ ] **Step 1: Write failing tests for query embedding cache**

```python
def test_query_embedding_cache_hits_on_repeat(monkeypatch):
    calls = {"n": 0}
    def fake_embed(query, input_type="search_query"):
        calls["n"] += 1
        return [0.1, 0.2]
    # invoke same query twice through cached helper
    assert calls["n"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd aquillm && pytest apps/documents/tests/test_query_embedding_cache.py -q`  
Expected: FAIL because helper does not exist.

- [ ] **Step 3: Implement cache helper and wire `text_chunk_search`**

Implementation details:
- `rag_cache.py`:
  - key normalizer (trim/lower/collapse whitespace)
  - `get_cached_query_embedding(query, input_type, model_id, compute_fn)` wrapper
  - robust `cache.get`/`cache.set` with exception-safe fallback
- Replace direct query embedding call in search orchestration with cached helper.

- [ ] **Step 4: Run tests to verify pass**

Run: `cd aquillm && pytest apps/documents/tests/test_query_embedding_cache.py apps/documents/tests -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aquillm/apps/documents/services/rag_cache.py aquillm/apps/documents/services/chunk_search.py aquillm/apps/documents/models/chunks.py aquillm/apps/documents/tests/test_query_embedding_cache.py
git commit -m "feat(rag): cache repeated query embeddings for search path"
```

---

## Chunk 2: Retrieval Path Caching (Docs, Images, Rerank)

### Task 3: Cache per-user collection document access map in chat tools

**Files:**
- Modify (preferred, post-Commit 6): `aquillm/lib/tools/search/vector_search.py`
- Modify (preferred, post-Commit 6): `aquillm/lib/tools/documents/list_ids.py`
- Modify (fallback, pre-Commit 6): `aquillm/apps/chat/consumers/chat.py`
- Modify: `aquillm/apps/collections/models/collection.py`
- Test: `aquillm/apps/chat/tests/test_vector_search_document_access_cache.py`

- [ ] **Step 1: Write failing tests for repeated access lookups**

```python
def test_vector_search_reuses_cached_accessible_docs(monkeypatch):
    # call vector_search twice with same user + selected collections
    # assert Collection.get_user_accessible_documents called once
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd aquillm && pytest apps/chat/tests/test_vector_search_document_access_cache.py -q`  
Expected: FAIL due repeated calls.

- [ ] **Step 3: Implement cache on `(user_id, collection_ids, perm)`**

Implementation details:
- Add short-TTL cache helper in chat consumer for:
  - `vector_search`
  - `document_ids`
- Cache value should include only serialized metadata needed by tools or doc IDs that can be re-hydrated.
- Keep permission correctness by keying on user and exact selected collection IDs.

- [ ] **Step 4: Run tests to verify pass**

Run: `cd aquillm && pytest apps/chat/tests/test_vector_search_document_access_cache.py apps/chat/tests -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aquillm/lib/tools/search/vector_search.py aquillm/lib/tools/documents/list_ids.py aquillm/apps/chat/consumers/chat.py aquillm/apps/collections/models/collection.py aquillm/apps/chat/tests/test_vector_search_document_access_cache.py
git commit -m "feat(chat): cache accessible document sets for repeated tool calls"
```

### Task 4: Cache document lookup and image data-url generation for multimodal chunks

**Files:**
- Modify: `aquillm/apps/documents/models/document.py`
- Modify (preferred, post-Commit 3): `aquillm/apps/documents/services/image_payloads.py`
- Modify (preferred, post-Commit 12): `aquillm/apps/documents/services/chunk_rerank.py`
- Modify (fallback): `aquillm/apps/documents/models/chunks.py`
- Modify (fallback only): `aquillm/aquillm/models.py`
- Test: `aquillm/apps/documents/tests/test_image_data_url_cache.py`
- Test: `aquillm/apps/documents/tests/test_document_lookup_cache.py`

- [ ] **Step 1: Write failing tests for repeated lookup/encoding**

```python
def test_doc_lookup_cache_avoids_multi_model_scan(monkeypatch):
    # same doc_id resolved repeatedly
    # assert underlying model filter loop runs once
    ...

def test_image_data_url_cache_avoids_repeat_storage_reads(monkeypatch):
    # _doc_image_data_url called twice for same doc
    # assert storage.open/read called once
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd aquillm && pytest apps/documents/tests/test_document_lookup_cache.py apps/documents/tests/test_image_data_url_cache.py -q`  
Expected: FAIL.

- [ ] **Step 3: Implement doc lookup + image payload caches**

Implementation details:
- Add cache wrapper in `Document.get_by_id`.
- Update chunk/document resolution to leverage cached lookup path.
- Cache image data URL generation in `apps.documents.services.image_payloads` by `doc_id` and image file identity (name + modified timestamp when available).
- Keep behavior identical on cache miss/error.

- [ ] **Step 4: Run tests to verify pass**

Run: `cd aquillm && pytest apps/documents/tests/test_document_lookup_cache.py apps/documents/tests/test_image_data_url_cache.py apps/documents/tests -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aquillm/apps/documents/models/document.py aquillm/apps/documents/services/image_payloads.py aquillm/apps/documents/services/chunk_rerank.py aquillm/apps/documents/models/chunks.py aquillm/aquillm/models.py aquillm/apps/documents/tests/test_document_lookup_cache.py aquillm/apps/documents/tests/test_image_data_url_cache.py
git commit -m "feat(multimodal): cache doc lookup and image data-url generation"
```

### Task 5: Cache rerank capability negotiation and short-lived rerank outputs

**Files:**
- Modify (preferred, post-Commit 12): `aquillm/apps/documents/services/chunk_rerank.py`
- Modify (fallback): `aquillm/apps/documents/models/chunks.py`
- Create: `aquillm/apps/documents/tests/test_rerank_cache.py`

- [ ] **Step 1: Write failing tests for rerank request count**

```python
def test_rerank_capability_cache_skips_probe_retries(monkeypatch):
    # simulate one successful payload shape, rerun same model/query
    # assert probe sequence does not repeat
    ...

def test_rerank_result_cache_hits_for_identical_candidate_signature(monkeypatch):
    # same query + same ordered chunk ids + same top_k
    # second call should not invoke requests.post
    ...
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd aquillm && pytest apps/documents/tests/test_rerank_cache.py -q`  
Expected: FAIL.

- [ ] **Step 3: Implement rerank caches**

Implementation details:
- Capability cache:
  - key: `(rerank_model, base_url)` -> preferred endpoint/payload variant
  - bypass repeated endpoint/payload probing.
- Result cache:
  - key: hash of `(query_normalized, top_k, rerank_model, candidate_chunk_ids)`
  - value: ranked chunk IDs
  - short TTL (30-120s) to avoid stale ranking risk.
- Keep fallback order behavior if cache data invalid.

- [ ] **Step 4: Run tests to verify pass**

Run: `cd aquillm && pytest apps/documents/tests/test_rerank_cache.py apps/documents/tests -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aquillm/apps/documents/services/chunk_rerank.py aquillm/apps/documents/models/chunks.py aquillm/apps/documents/tests/test_rerank_cache.py
git commit -m "feat(rerank): cache capability negotiation and repeated rerank outputs"
```

---

## Chunk 3: Memory Path, Observability, and Rollout Safety

### Task 6: Reuse embedding cache in episodic memory retrieval

**Files:**
- Modify: `aquillm/aquillm/memory.py`
- Modify: `aquillm/apps/documents/services/rag_cache.py`
- Test: `aquillm/apps/memory/tests/test_memory_query_embedding_cache.py`

- [ ] **Step 1: Write failing memory cache test**

```python
def test_memory_retrieval_reuses_query_embedding_cache(monkeypatch, django_user_model):
    # call get_episodic_memories twice with same query
    # assert embed provider called once
    ...
```

- [ ] **Step 2: Run test to verify failure**

Run: `cd aquillm && pytest apps/memory/tests/test_memory_query_embedding_cache.py -q`  
Expected: FAIL.

- [ ] **Step 3: Integrate shared cached query embedding helper**

Implementation details:
- Use same normalization/key strategy as retrieval query embedding cache.
- Preserve existing mem0-first behavior and local fallback.

- [ ] **Step 4: Run tests to verify pass**

Run: `cd aquillm && pytest apps/memory/tests/test_memory_query_embedding_cache.py apps/memory/tests -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aquillm/aquillm/memory.py aquillm/apps/documents/services/rag_cache.py aquillm/apps/memory/tests/test_memory_query_embedding_cache.py
git commit -m "feat(memory): reuse query embedding cache for episodic retrieval"
```

### Task 7: Add cache observability and explicit feature flags

**Files:**
- Modify (preferred, post-Commit 12): `aquillm/apps/documents/services/chunk_search.py`
- Modify (preferred, post-Commit 12): `aquillm/apps/documents/services/chunk_rerank.py`
- Modify (fallback): `aquillm/apps/documents/models/chunks.py`
- Modify (preferred, post-Commit 6): `aquillm/lib/tools/search/vector_search.py`
- Modify (fallback): `aquillm/apps/chat/consumers/chat.py`
- Modify: `aquillm/aquillm/memory.py`
- Modify: `.env.example`
- Test: `aquillm/tests/integration/test_rag_cache_flags.py`

- [ ] **Step 1: Write failing tests for feature flags**

```python
def test_rag_cache_disable_flag_bypasses_cache(monkeypatch, settings):
    settings.RAG_CACHE_ENABLED = False
    # assert compute path always executes
```

- [ ] **Step 2: Run test to verify failure**

Run: `cd aquillm && pytest tests/integration/test_rag_cache_flags.py -q`  
Expected: FAIL.

- [ ] **Step 3: Implement flags and logging**

Implementation details:
- Add structured log lines for cache hit/miss:
  - embedding cache
  - doc access cache
  - image payload cache
  - rerank cache
- Add env toggles in `.env.example`.
- Ensure logs avoid sensitive content.

- [ ] **Step 4: Run tests to verify pass**

Run: `cd aquillm && pytest tests/integration/test_rag_cache_flags.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aquillm/apps/documents/services/chunk_search.py aquillm/apps/documents/services/chunk_rerank.py aquillm/apps/documents/models/chunks.py aquillm/lib/tools/search/vector_search.py aquillm/apps/chat/consumers/chat.py aquillm/aquillm/memory.py .env.example aquillm/tests/integration/test_rag_cache_flags.py
git commit -m "chore(observability): add RAG cache flags and hit-miss instrumentation"
```

### Task 8: Verification suite and rollout checklist

**Files:**
- Modify: `docs/documents/architecture/aquillm-current-architecture-mermaid.md`
- Create: `docs/roadmap/plans/superseded/2026-03-22-multimodal-rag-caching-rollout-notes.md`

- [ ] **Step 1: Run targeted backend tests**

Run:
- `cd aquillm && pytest apps/documents/tests/test_query_embedding_cache.py -q`
- `cd aquillm && pytest apps/documents/tests/test_rerank_cache.py -q`
- `cd aquillm && pytest apps/documents/tests/test_image_data_url_cache.py -q`
- `cd aquillm && pytest apps/chat/tests/test_vector_search_document_access_cache.py -q`
- `cd aquillm && pytest apps/memory/tests/test_memory_query_embedding_cache.py -q`
- `cd aquillm && pytest tests/integration/test_rag_cache_flags.py -q`

Expected: PASS.

- [ ] **Step 2: Run regression smoke tests**

Run: `cd aquillm && pytest apps/chat/tests apps/documents/tests apps/memory/tests tests/integration -q --tb=short`  
Expected: PASS.

- [ ] **Step 3: Record rollout notes and defaults**

Document:
- safe initial TTLs
- flag strategy (`RAG_CACHE_ENABLED=0/1`)
- rollback procedure (disable cache flags only, no schema changes)
- what metrics/logs to watch in first deploy window.

- [ ] **Step 4: Commit**

```bash
git add docs/documents/architecture/aquillm-current-architecture-mermaid.md docs/roadmap/plans/superseded/2026-03-22-multimodal-rag-caching-rollout-notes.md
git commit -m "docs(rag): add caching rollout and rollback notes"
```

---

## Chunk 4: Phase 2 Extended Latency Optimizations

### Task 9: Remove synchronous conversation auto-title from hot chat path

**Files:**
- Modify: `aquillm/apps/chat/models/conversation.py`
- Modify: `aquillm/apps/chat/consumers/chat.py`
- Create: `aquillm/apps/chat/tests/test_conversation_title_async.py`

- [ ] **Step 1: Write failing test for non-blocking title generation**

```python
def test_set_name_not_called_inline_on_chat_save(monkeypatch):
    # verify save path does not synchronously invoke LLM titling
    ...
```

- [ ] **Step 2: Run test to verify failure**

Run: `cd aquillm && pytest apps/chat/tests/test_conversation_title_async.py -q`  
Expected: FAIL.

- [ ] **Step 3: Implement deferred/asynchronous title generation**

Implementation details:
- Move title generation to async task trigger or delayed path.
- Keep existing fallback title logic unchanged.

- [ ] **Step 4: Run tests to verify pass**

Run: `cd aquillm && pytest apps/chat/tests/test_conversation_title_async.py apps/chat/tests -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aquillm/apps/chat/models/conversation.py aquillm/apps/chat/consumers/chat.py aquillm/apps/chat/tests/test_conversation_title_async.py
git commit -m "perf(chat): defer conversation auto-title off request critical path"
```

### Task 10: Cache ingestion monitor response per user (short TTL)

**Files:**
- Modify (preferred, post-Commit 11): `aquillm/apps/ingestion/views/api/uploads.py`
- Modify (fallback): `aquillm/apps/ingestion/views/api.py`
- Create: `aquillm/apps/ingestion/tests/test_ingestion_monitor_cache.py`

- [ ] **Step 1: Write failing test for repeated monitor calls**

```python
def test_ingestion_monitor_uses_short_ttl_cache(client, django_user_model, monkeypatch):
    # repeated calls should reuse cached payload and avoid repeated model fan-out
    ...
```

- [ ] **Step 2: Run test to verify failure**

Run: `cd aquillm && pytest apps/ingestion/tests/test_ingestion_monitor_cache.py -q`  
Expected: FAIL.

- [ ] **Step 3: Implement user-scoped monitor cache**

Implementation details:
- Cache key by `(user_id)`.
- TTL 5-15s to balance freshness with UI polling overhead.

- [ ] **Step 4: Run tests to verify pass**

Run: `cd aquillm && pytest apps/ingestion/tests/test_ingestion_monitor_cache.py apps/ingestion/tests -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aquillm/apps/ingestion/views/api/uploads.py aquillm/apps/ingestion/views/api.py aquillm/apps/ingestion/tests/test_ingestion_monitor_cache.py
git commit -m "perf(ingestion): cache ingestion monitor payload for polling clients"
```

### Task 11: Add HTTP cache validators for document image endpoint

**Files:**
- Modify: `aquillm/apps/documents/views/pages.py`
- Create: `aquillm/apps/documents/tests/test_document_image_http_cache.py`

- [ ] **Step 1: Write failing test for ETag/Last-Modified behavior**

```python
def test_document_image_sets_cache_headers(client, image_doc):
    resp = client.get(f"/aquillm/document_image/{image_doc.id}/")
    assert "ETag" in resp
```

- [ ] **Step 2: Run test to verify failure**

Run: `cd aquillm && pytest apps/documents/tests/test_document_image_http_cache.py -q`  
Expected: FAIL.

- [ ] **Step 3: Implement cache headers and conditional GET support**

Implementation details:
- Add `ETag`, `Last-Modified`, `Cache-Control`.
- Support `If-None-Match` / `If-Modified-Since` returning `304`.

- [ ] **Step 4: Run tests to verify pass**

Run: `cd aquillm && pytest apps/documents/tests/test_document_image_http_cache.py apps/documents/tests -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aquillm/apps/documents/views/pages.py aquillm/apps/documents/tests/test_document_image_http_cache.py
git commit -m "perf(documents): add HTTP caching semantics for document_image endpoint"
```

### Task 12: Cache Mem0 OSS supported search call signature

**Files:**
- Modify: `aquillm/lib/memory/mem0/operations.py`
- Create: `aquillm/lib/memory/tests/test_mem0_search_signature_cache.py`

- [ ] **Step 1: Write failing test for repeated signature probing**

```python
def test_mem0_search_reuses_cached_call_shape(monkeypatch):
    # first call discovers valid signature, second call skips failed variants
    ...
```

- [ ] **Step 2: Run test to verify failure**

Run: `cd aquillm && pytest lib/memory/tests/test_mem0_search_signature_cache.py -q`  
Expected: FAIL.

- [ ] **Step 3: Implement signature-shape cache**

Implementation details:
- Cache successful arg/kwarg pattern in-memory per process.
- Keep fallback probing if cached signature fails later.

- [ ] **Step 4: Run tests to verify pass**

Run: `cd aquillm && pytest lib/memory/tests/test_mem0_search_signature_cache.py lib/memory/tests -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aquillm/lib/memory/mem0/operations.py aquillm/lib/memory/tests/test_mem0_search_signature_cache.py
git commit -m "perf(mem0): cache discovered OSS search signature to avoid probe overhead"
```

### Task 13: Add ingestion extraction dedupe cache by file hash

**Files:**
- Modify: `aquillm/aquillm/tasks.py`
- Modify: `aquillm/aquillm/ingestion/parsers.py`
- Create: `aquillm/apps/ingestion/tests/test_extraction_dedupe_cache.py`

- [ ] **Step 1: Write failing test for duplicate file ingestion**

```python
def test_duplicate_payload_reuses_extraction_outputs(monkeypatch):
    # same content hash should skip repeated heavy extraction path
    ...
```

- [ ] **Step 2: Run test to verify failure**

Run: `cd aquillm && pytest apps/ingestion/tests/test_extraction_dedupe_cache.py -q`  
Expected: FAIL.

- [ ] **Step 3: Implement content-hash extraction cache**

Implementation details:
- Hash raw bytes + content type + parser version stamp.
- Cache normalized extraction outputs with short/medium TTL.
- Preserve current safety/error handling.

- [ ] **Step 4: Run tests to verify pass**

Run: `cd aquillm && pytest apps/ingestion/tests/test_extraction_dedupe_cache.py apps/ingestion/tests -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aquillm/aquillm/tasks.py aquillm/aquillm/ingestion/parsers.py aquillm/apps/ingestion/tests/test_extraction_dedupe_cache.py
git commit -m "perf(ingestion): dedupe parser extraction for repeated identical uploads"
```

### Task 14: Add UUID-to-document-type resolver cache for cross-model scans

**Files:**
- Modify: `aquillm/apps/documents/models/document.py`
- Modify: `aquillm/apps/documents/views/pages.py`
- Create: `aquillm/apps/documents/tests/test_doc_type_resolver_cache.py`

- [ ] **Step 1: Write failing test for repeated subtype scans**

```python
def test_get_doc_reuses_doc_type_resolver_cache(monkeypatch):
    # repeated lookups for same UUID should not scan all DESCENDED_FROM_DOCUMENT each time
    ...
```

- [ ] **Step 2: Run test to verify failure**

Run: `cd aquillm && pytest apps/documents/tests/test_doc_type_resolver_cache.py -q`  
Expected: FAIL.

- [ ] **Step 3: Implement resolver cache**

Implementation details:
- Cache `(doc_id -> model_label)` mapping with TTL and safe invalidation on miss.
- Reuse in `Document.get_by_id` and `get_doc`.

- [ ] **Step 4: Run tests to verify pass**

Run: `cd aquillm && pytest apps/documents/tests/test_doc_type_resolver_cache.py apps/documents/tests -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aquillm/apps/documents/models/document.py aquillm/apps/documents/views/pages.py aquillm/apps/documents/tests/test_doc_type_resolver_cache.py
git commit -m "perf(documents): cache doc-id model resolution to avoid repeated subtype scans"
```

### Task 15: Extend websocket lifecycle caching for repeated permission/access checks

**Files:**
- Modify (preferred, post-Commit 6): `aquillm/apps/chat/services/tool_wiring.py`
- Modify (fallback): `aquillm/apps/chat/consumers/chat.py`
- Create: `aquillm/apps/chat/tests/test_websocket_access_cache.py`

- [ ] **Step 1: Write failing test for repeated permission checks within socket lifecycle**

```python
def test_chat_socket_reuses_permission_resolution(monkeypatch):
    # same user/collections in one socket session should not re-run identical access checks
    ...
```

- [ ] **Step 2: Run test to verify failure**

Run: `cd aquillm && pytest apps/chat/tests/test_websocket_access_cache.py -q`  
Expected: FAIL.

- [ ] **Step 3: Implement per-socket access cache**

Implementation details:
- Add in-memory cache on consumer instance for connection lifetime.
- Keep shared Redis cache as outer layer; socket cache as fastest inner layer.

- [ ] **Step 4: Run tests to verify pass**

Run: `cd aquillm && pytest apps/chat/tests/test_websocket_access_cache.py apps/chat/tests -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aquillm/apps/chat/services/tool_wiring.py aquillm/apps/chat/consumers/chat.py aquillm/apps/chat/tests/test_websocket_access_cache.py
git commit -m "perf(chat): add per-websocket access cache for repeated permission lookups"
```

---

## Phase 2 Optional Flags (Recommended)

- `RAG_PHASE2_ENABLE_ASYNC_TITLE=1`
- `RAG_INGEST_MONITOR_CACHE_TTL_SECONDS=10`
- `RAG_DOC_IMAGE_HTTP_CACHE_MAX_AGE_SECONDS=300`
- `RAG_MEM0_SIGNATURE_CACHE_ENABLED=1`
- `RAG_INGEST_EXTRACTION_CACHE_TTL_SECONDS=900`
- `RAG_DOC_TYPE_RESOLVER_TTL_SECONDS=300`
- `RAG_SOCKET_ACCESS_CACHE_ENABLED=1`

---

## Suggested Initial Runtime Defaults

- `RAG_CACHE_ENABLED=1`
- `RAG_EMBED_QUERY_TTL_SECONDS=300`
- `RAG_DOC_ACCESS_TTL_SECONDS=60`
- `RAG_IMAGE_DATA_URL_TTL_SECONDS=120`
- `RAG_RERANK_CAPABILITY_TTL_SECONDS=600`
- `RAG_RERANK_RESULT_TTL_SECONDS=45`

These values are intentionally conservative to prioritize freshness and safety.

---

## Execution Notes

- Use `@superpowers/test-driven-development` while implementing each task.
- Use `@superpowers/verification-before-completion` before claiming each chunk is complete.
- Keep commits scoped to one task to simplify rollback/bisect.
- Do not change retrieval model IDs or database schema in this plan.

---

**Plan complete and saved to `docs/roadmap/plans/superseded/2026-03-22-multimodal-rag-caching-latency-optimization.md`. Ready to execute?**



