# Fast Academic RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make selected-collection academic answers faster while preserving reranking, graph-assisted evidence, citations, and thinking for final synthesis.

**Architecture:** Route obvious collection questions through the existing deterministic direct-RAG seam. Build up to three non-LLM query variants, execute the existing reranker-backed search concurrently, reciprocal-rank-fuse cited chunks, and make one thinking-enabled synthesis call. Fallback tool routing disables thinking per request, while graph diagnostics remain private and citations retain their current backend enforcement.

**Tech Stack:** Python 3, Django, Channels/ASGI, pytest, OpenAI-compatible vLLM, React/TypeScript, Vitest.

**Spec:** `docs/superpowers/specs/2026-08-25-fast-academic-rag.md`

## Global Constraints

- Keep final academic synthesis thinking-enabled.
- Keep vector, trigram, exact-match, optional knowledge-graph, and reranker stages authoritative for every retrieval query.
- Keep the configured 131,072-token model context limit and chunk citation format unchanged.
- Never commit or print the remote `.env`.
- Deploy only in this order: local commit, push `origin/development`, remote pull, deploy.

---

### Task 1: Per-request thinking control

**Files:**
- Modify: `aquillm/lib/llm/providers/complete_turn.py`
- Modify: `aquillm/lib/llm/providers/openai.py`
- Test: `aquillm/apps/chat/tests/test_llm_complete_retry.py`
- Test: `aquillm/lib/llm/tests/test_openai_memory_prompt.py`

**Interfaces:**
- Consumes: existing provider keyword `thinking_budget` and `UserMessage.tools`.
- Produces: OpenAI-compatible `chat_template_kwargs.enable_thinking=False` only when `thinking_budget == 0`; final synthesis continues to use the global default.

- [ ] **Step 1: Write a failing provider test**

```python
def test_local_vllm_thinking_budget_zero_overrides_enabled_default(self):
    # Call get_message(thinking_budget=0) with OPENAI_COMPAT_ENABLE_THINKING=1.
    # Assert extra_body.chat_template_kwargs.enable_thinking is False.
```

- [ ] **Step 2: Run the provider test and verify it fails because the override is ignored**

Run: `pytest -q aquillm/lib/llm/tests/test_openai_memory_prompt.py -k thinking_budget_zero`
Expected: FAIL with `enable_thinking` equal to `True`.

- [ ] **Step 3: Write failing orchestration tests**

```python
async def test_required_tool_routing_disables_thinking():
    # Complete an initial UserMessage carrying tools and assert the fake provider
    # receives thinking_budget=0.

async def test_post_tool_synthesis_keeps_configured_thinking():
    # Complete a ToolMessage evidence turn and assert thinking_budget is absent.
```

- [ ] **Step 4: Run the orchestration tests and verify the initial tool assertion fails**

Run: `pytest -q aquillm/apps/chat/tests/test_llm_complete_retry.py -k thinking`
Expected: FAIL because `complete_conversation_turn` does not pass `thinking_budget`.

- [ ] **Step 5: Implement the minimal override**

```python
# complete_turn.py: add only for initial UserMessage tool routing
sdk_args["thinking_budget"] = 0

# openai.py
thinking_budget = kwargs.pop("thinking_budget", None)
enable_thinking = global_default if thinking_budget is None else thinking_budget != 0
```

Apply the same zero-budget keyword to hidden/forced tool-call retries, but not to
post-tool synthesis, citation repair, continuation, or direct final answers.

- [ ] **Step 6: Run both targeted suites**

Run: `pytest -q aquillm/lib/llm/tests/test_openai_memory_prompt.py aquillm/apps/chat/tests/test_llm_complete_retry.py`
Expected: PASS.

### Task 2: Deterministic multi-query retrieval and rank fusion

**Files:**
- Modify: `aquillm/apps/chat/services/rag_config.py`
- Modify: `aquillm/apps/chat/services/rag_query.py`
- Create: `aquillm/apps/chat/services/rag_retrieval.py`
- Modify: `aquillm/apps/chat/services/rag_pipeline.py`
- Test: `aquillm/apps/chat/tests/test_rag_query.py`
- Create: `aquillm/apps/chat/tests/test_rag_retrieval.py`
- Modify: `aquillm/apps/chat/tests/test_direct_rag_pipeline.py`

**Interfaces:**
- Produces: `build_retrieval_queries(conversation, latest_user_text, max_queries) -> list[str]`.
- Produces: `merge_ranked_tool_results(results, limit) -> dict[str, Any]`.
- Consumes: the existing synchronous `_run_vector_search(consumer, query, top_k)` so every query retains the current reranker and graph overlay.

- [ ] **Step 1: Write failing query-variant tests**

```python
def test_multi_part_question_keeps_full_query_and_question_clauses():
    queries = build_retrieval_queries(convo, "Explain each paper? What overlaps?", 3)
    assert queries[0] == "Explain each paper? What overlaps?"
    assert queries[1:] == ["Explain each paper", "What overlaps"]

def test_simple_question_uses_one_query():
    assert build_retrieval_queries(convo, "Explain dark matter", 3) == ["Explain dark matter"]
```

- [ ] **Step 2: Run and verify missing API failure**

Run: `pytest -q aquillm/apps/chat/tests/test_rag_query.py`
Expected: FAIL because `build_retrieval_queries` does not exist.

- [ ] **Step 3: Implement minimal deterministic decomposition**

Split only on question marks, semicolons, and newlines; normalize whitespace,
deduplicate case-insensitively, retain the full rewritten query first, require a
meaningful clause length, and cap with `RAG_DIRECT_MAX_QUERIES` (default `3`).

- [ ] **Step 4: Write failing rank-fusion tests**

```python
def test_merge_deduplicates_citations_and_preserves_cross_query_evidence():
    merged = merge_ranked_tool_results([query_a, query_b], limit=3)
    assert [row["citation"] for row in merged["result"]] == [ref_a, ref_shared, ref_b]
    assert merged["retrieved_count"] == 3
```

Also assert private `_retrieval_diagnostics` never appears in
`serialize_tool_result_for_llm(merged)`.

- [ ] **Step 5: Run and verify missing module/API failure**

Run: `pytest -q aquillm/apps/chat/tests/test_rag_retrieval.py`
Expected: FAIL because `rag_retrieval` does not exist.

- [ ] **Step 6: Implement stable reciprocal-rank fusion**

Use citation/ref as the primary identity, fall back to chunk ID, sum
`1 / (60 + rank)` per list, use first-seen order as a stable tie-breaker, retain
document titles and image instructions, and cap the merged rows to
`RAG_DIRECT_TOP_K`.

- [ ] **Step 7: Write a failing direct-pipeline test**

```python
async def test_multi_part_direct_rag_searches_variants_before_one_synthesis():
    # Assert distinct queries are searched, cited rows are merged, and synthesis
    # is called once with the merged EvidencePacket.
```

- [ ] **Step 8: Run and verify the pipeline still performs one search**

Run: `pytest -q aquillm/apps/chat/tests/test_direct_rag_pipeline.py -k multi_part`
Expected: FAIL because only the original query is searched.

- [ ] **Step 9: Execute searches concurrently through Channels database wrappers**

Use `database_sync_to_async(_run_vector_search, thread_sensitive=False)` plus
`asyncio.gather`. Allocate each query enough candidates for reranking, merge to
the configured final limit, then keep the existing evidence builder and single
synthesis call.

- [ ] **Step 10: Run direct-RAG suites**

Run: `pytest -q aquillm/apps/chat/tests/test_rag_query.py aquillm/apps/chat/tests/test_rag_retrieval.py aquillm/apps/chat/tests/test_direct_rag_pipeline.py aquillm/apps/chat/tests/test_rag_evidence.py`
Expected: PASS.

### Task 3: Safe knowledge-graph contribution observability

**Files:**
- Modify: `aquillm/lib/tools/search/vector_search.py`
- Modify: `aquillm/apps/chat/services/rag_retrieval.py`
- Modify: `aquillm/apps/chat/services/rag_pipeline.py`
- Modify: `aquillm/apps/chat/tests/test_direct_rag_pipeline.py`
- Modify: `aquillm/lib/tools/search/tests/test_vector_search_pack.py`

**Interfaces:**
- Produces: private `_retrieval_diagnostics` on packed tool results.
- Consumes: existing validated `log_direct_rag_turn` optional graph fields.

- [ ] **Step 1: Write failing private-diagnostics tests**

```python
def test_pack_keeps_graph_diagnostics_private_for_backend_metrics():
    packed = pack_chunk_search_results(..., retrieval_diagnostics=graph_diag)
    assert packed["_retrieval_diagnostics"]["graph_status"] == "hit"
    assert "graph_status" not in serialize_tool_result_for_llm(packed)
```

- [ ] **Step 2: Run and verify private diagnostics are absent**

Run: `pytest -q aquillm/lib/tools/search/tests/test_vector_search_pack.py -k diagnostics`
Expected: FAIL because result-bearing payloads discard diagnostics.

- [ ] **Step 3: Retain diagnostics under a private key**

Copy diagnostics to `_retrieval_diagnostics` for both result and no-result
payloads. Do not alter the public `retrieval_diagnostics` no-result contract.

- [ ] **Step 4: Write a failing pipeline metrics test**

Assert a merged graph `hit` passes only safe timing/count/status/signature fields
to `log_direct_rag_turn`, while private labels and paths never enter serialized
tool content.

- [ ] **Step 5: Wire merged private diagnostics to the existing validated logger**

Aggregate multiple query statuses deterministically (`hit` wins, then `timeout`,
`error`, `miss`), sum bounded counts/timing, and let `rag_metrics` perform its
existing final validation/redaction.

- [ ] **Step 6: Run retrieval and graph tests**

Run: `pytest -q aquillm/lib/tools/search/tests/test_vector_search_pack.py aquillm/apps/chat/tests/test_direct_rag_pipeline.py aquillm/apps/documents/tests/test_chunk_search_graph_overlay.py aquillm/apps/documents/tests/test_hybrid_graph_reranker_authority.py`
Expected: PASS.

### Task 4: Citation source CSRF

**Files:**
- Modify: `react/src/features/chat/components/MessageSources.tsx`
- Test: `react/src/features/chat/components/MessageSources.test.tsx`

**Interfaces:**
- Consumes: existing `getCsrfCookie()` utility.
- Produces: `X-CSRFToken` on `/api/citations/sources/` POST requests.

- [ ] **Step 1: Write a failing component test**

Mock the CSRF utility and `fetch`; expand sources and assert the POST includes
`X-CSRFToken` with the mocked value.

- [ ] **Step 2: Run and verify the missing-header failure**

Run: `npm test -- --run MessageSources.test.tsx`
Expected: FAIL because only `Content-Type` is sent.

- [ ] **Step 3: Add the existing CSRF helper and header**

Match the working citation-narrowing request in `MessageBubble.tsx`; do not add a
new cookie parser.

- [ ] **Step 4: Run the component test and React typecheck**

Run: `npm test -- --run MessageSources.test.tsx`
Expected: PASS.

Run: `npm run typecheck`
Expected: PASS.

### Task 5: Verification, commit, and deployment

**Files:**
- Modify remotely only: `~/AquiLLM/.env` after inspecting current non-secret flag values.
- Do not add: local `.env`, remote `.env`, `.codex-ssh/`.

**Interfaces:**
- Consumes: completed local code and tests.
- Produces: synced `development` deployment with measured RAG and graph behavior.

- [ ] **Step 1: Run focused Python tests and the offline RAG eval**

Run: `pytest -q` on all files changed above.
Expected: PASS.

Run: `python -m apps.chat.evals.run_rag_eval` from `aquillm/`.
Expected: all cases PASS.

- [ ] **Step 2: Run repository verification proportional to the changes**

Run the project Python lint/type commands and React test/typecheck commands found
in package configuration. Expected: PASS with no new warnings.

- [ ] **Step 3: Inspect the staged diff for secrets and scope**

Run: `git diff --check`, `git status --short`, and a staged scan proving `.env`
and `.codex-ssh/` are absent.

- [ ] **Step 4: Commit and push**

Commit message: `fix(rag): accelerate evidence-first academic search`

Push: `git push origin development`

- [ ] **Step 5: Restore authenticated remote access without exposing secrets**

If the ephemeral key is not authorized, ask the user to add its public key. Do
not print private keys or server environment values.

- [ ] **Step 6: Inspect live graph readiness before changing gates**

Check container health, projection/gateway/Memgraph readiness, collection
projection status, and only the names plus boolean/non-secret values of relevant
flags. Do not dump the environment.

- [ ] **Step 7: Update remote-only flags and deploy**

Keep final thinking enabled. Enable direct RAG with up to three queries and a
larger evidence budget. Enable graph traversal/direct/extended gates only when
their live readiness checks pass. Pull `development`, rebuild/restart affected
services, and leave the remote `.env` untracked.

- [ ] **Step 8: Monitor one UI request**

Record end-to-end duration, number of LLM generations, per-query retrieval time,
reranker use/cache status, graph status/candidate count, cited chunk diversity,
final token usage, and citation source/chunk HTTP status. Acceptance: one final
thinking generation for direct RAG, useful evidence from selected collections,
working citations, and no 5xx/CSRF errors.
