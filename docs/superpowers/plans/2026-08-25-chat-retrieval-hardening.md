# Chat Retrieval Reliability and Latency Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make selected-collection chat reliably evidence-first and materially faster without weakening academic synthesis, reranking, citations, thinking, authorization, or the 131,072-token context contract.

**Architecture:** Resolve selected-collection intent deterministically before model routing, keep the existing parallel hybrid retrieval and single final synthesis, and bound the compatibility tool loop to a small non-thinking generation. Make WebSocket delivery best-effort after persistence, make local reranking capability-aware and concurrent, and record safe per-stage latency/token metadata.

**Tech Stack:** Django, Channels/WebSockets, asyncio, Pydantic, OpenAI-compatible chat completions, requests, tiktoken, pytest/Django tests, React/Vitest, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-08-25-chat-retrieval-hardening.md`

## Global Constraints

- Selected collections are an evidence contract for substantive knowledge questions.
- Final synthesis keeps thinking enabled and keeps its configured answer budget.
- Mechanical initial/retry tool selection sets `thinking_budget=0`, `chat_template_kwargs.enable_thinking=false`, and uses at most 256 completion tokens, including when an environment cap is `0`.
- Retrieval authorization and existing vector, trigram, exact-term, optional-ready-graph, reciprocal-rank fusion, reranking, evidence packing, and citations remain authoritative.
- Tool text recovery may execute only a tool supplied to that exact model request and must never expose raw tool syntax.
- Logs contain only opaque correlation IDs, stages, counts, durations, safe status values, and token counts; never prompts, tool arguments, document IDs, graph paths, source text, credentials, or environment values.
- Graph retrieval remains disabled until the graph build/projection is complete and benchmarked.
- Preserve the UI/server 131,072-token context contract.
- The remote `.env` remains untracked and must never be added, committed, printed, or pushed.
- Deployment order is local commit, push `origin/development`, remote pull, then targeted `--no-deps` deployment.
- Do not recreate or start `vllm_transcribe`.

---

## File Structure

- `aquillm/apps/chat/services/rag_intent.py`: deterministic precedence and selected-collection knowledge-question policy.
- `aquillm/apps/chat/services/rag_pipeline.py`: direct-RAG execution, retry reuse, failure classification, and stage correlation.
- `aquillm/apps/chat/consumers/chat_transport.py`: recognized-disconnect detection and best-effort WebSocket delivery.
- `aquillm/apps/chat/consumers/chat.py`, `chat_receive.py`, `chat_delta.py`: use the transport boundary and the same direct-RAG-first completion path on receive/reconnect.
- `aquillm/lib/llm/providers/complete_turn.py`: hard mechanical generation cap and deterministic required-tool fallback.
- `aquillm/lib/llm/providers/openai_tool_text.py`, `fallback_heuristics.py`, `visibility.py`: authorized same-line textual tool recovery and raw-markup suppression.
- `aquillm/apps/documents/services/chunk_rerank_budget.py`: token-based pair trimming with a deterministic safety reserve.
- `aquillm/apps/documents/services/chunk_rerank_local_vllm.py`: cached payload capability and supported batch/bounded concurrent single scoring.
- `aquillm/lib/llm/types/response.py`, `aquillm/lib/llm/providers/openai.py`, `openai_streaming.py`: optional reasoning-token metadata and safe provider timing logs.
- Existing focused test modules receive incident-specific regressions; no schema migration is required.

---

### Task 1: Selected-Collection Evidence Contract

**Files:**
- Modify: `aquillm/apps/chat/services/rag_intent.py`
- Modify: `aquillm/apps/chat/services/rag_pipeline.py`
- Modify: `aquillm/apps/chat/tests/test_rag_intent.py`
- Modify: `aquillm/apps/chat/tests/test_document_search_intent.py`
- Modify: `aquillm/apps/chat/tests/test_direct_rag_pipeline.py`
- Modify: `aquillm/apps/chat/tests/test_rag_query.py`

**Interfaces:**
- Consumes: `classify_chat_message(text, selected_collection_ids, prior_tools=None, prior_tool_choice=None) -> ChatIntent` and `build_retrieval_queries(conversation, latest_user_text, max_queries) -> list[str]`.
- Produces: selected-collection substantive questions return `ChatIntent(requires_rag=True, reason="collection_backed_question")`; retries with prior retrieval reuse the last persisted vector-search query and enter direct RAG.

- [ ] **Step 1: Write failing intent tests for selected-collection questions and exclusions**

```python
@pytest.mark.parametrize("text", [
    "what is attensity",
    "attensity",
    "explain calibration drift",
    "compare the reported outcomes",
])
def test_selected_collection_substantive_question_requires_rag(text):
    result = classify_chat_message(text, selected_collection_ids=[203])
    assert result.requires_rag is True
    assert result.reason == "collection_backed_question"

@pytest.mark.parametrize("text", [
    "hi",
    "thanks",
    "open collection settings",
    "what did we discuss in our previous chat?",
    "run point source detection on this FITS file",
])
def test_selected_collection_non_document_work_stays_out_of_rag(text):
    result = classify_chat_message(text, selected_collection_ids=[203])
    assert result.requires_rag is False

def test_attensity_without_selected_collection_remains_general_chat():
    result = classify_chat_message("what is attensity", selected_collection_ids=[])
    assert result.requires_rag is False
```

- [ ] **Step 2: Run the intent regressions and verify the selected-collection cases fail**

Run: `python -m pytest aquillm/apps/chat/tests/test_rag_intent.py aquillm/apps/chat/tests/test_document_search_intent.py -q`

Expected: the new `attensity`/terse-term cases fail because `_collection_backed_document_question` still requires a literal document cue; existing explicit-search, local-tool, retry, and no-selection cases pass.

- [ ] **Step 3: Implement explicit precedence and conservative substantive-question detection**

Add anchored regular expressions for small talk, UI/collection management, and cross-chat history. Preserve retry and local-tool precedence. After exclusions, treat non-empty selected-collection text containing an alphanumeric knowledge term as collection-backed; do not require `paper`, `document`, or `collection`.

```python
def _collection_backed_document_question(text: str, collection_ids: list) -> bool:
    normalized = " ".join((text or "").split()).strip()
    if not collection_ids or not normalized:
        return False
    if _SMALL_TALK_RE.fullmatch(normalized):
        return False
    if _UI_MANAGEMENT_RE.search(normalized) or _CHAT_HISTORY_RE.search(normalized):
        return False
    if _SELECTED_COLLECTION_CLARIFICATION_RE.match(normalized):
        return True
    return bool(re.search(r"[A-Za-z0-9]", normalized))
```

- [ ] **Step 4: Write failing direct-pipeline tests for zero tool routing and retry reuse**

```python
async def test_selected_collection_definition_uses_direct_rag_without_tool_spin(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    searched_queries: list[str] = []

    def fake_search(_consumer, query, _top_k):
        searched_queries.append(query)
        return _results_payload()

    async def fake_synth(_llm_if, working_convo, _packet, *, stream_func=None):
        return working_convo + [
            AssistantMessage(content="Attensity answer [doc:doc-a chunk:1].", stop_reason="end_turn")
        ]

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", fake_search)
    monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", fake_synth)
    convo = Conversation(system="test", messages=[UserMessage(content="what is attensity")])
    consumer = _consumer(convo, [203])
    llm_if = SimpleNamespace(get_message=AsyncMock())
    outcome = await run_direct_rag_turn(consumer, llm_if, convo)
    assert outcome == "handled"
    assert searched_queries == ["what is attensity"]
    llm_if.get_message.assert_not_called()

async def test_retry_reuses_last_direct_vector_query(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    searched_queries: list[str] = []

    def fake_search(_consumer, query, _top_k):
        searched_queries.append(query)
        return _results_payload()

    async def fake_synth(_llm_if, working_convo, _packet, *, stream_func=None):
        return working_convo + [
            AssistantMessage(content="Retried answer [doc:doc-a chunk:1].", stop_reason="end_turn")
        ]

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", fake_search)
    monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", fake_synth)
    convo = Conversation(
        system="test",
        messages=[
            UserMessage(content="what is attensity"),
            AssistantMessage(
                content="",
                stop_reason="tool_use",
                tool_call_id="call-1",
                tool_call_name="vector_search",
                tool_call_input={"search_string": "attensity", "top_k": 10},
            ),
            ToolMessage(
                tool_name="vector_search",
                for_whom="assistant",
                content="evidence",
                arguments={"search_string": "attensity", "top_k": 10},
                result_dict=_results_payload(),
            ),
            AssistantMessage(content="Earlier answer.", stop_reason="end_turn"),
            UserMessage(content="retry"),
        ],
    )
    consumer = _consumer(convo, [203])
    llm_if = SimpleNamespace(get_message=AsyncMock())
    outcome = await run_direct_rag_turn(consumer, llm_if, convo)
    assert outcome == "handled"
    assert searched_queries == ["attensity"]
```

- [ ] **Step 5: Run the direct-RAG regressions and verify they fail for current classification/retry behavior**

Run: `python -m pytest aquillm/apps/chat/tests/test_direct_rag_pipeline.py aquillm/apps/chat/tests/test_rag_query.py -q`

Expected: the selected definition is skipped under the old cue rule and the retry is skipped because `run_direct_rag_turn` rejects `intent.is_retry`.

- [ ] **Step 6: Route selected knowledge and prior-retrieval retry turns through direct RAG**

Pass the latest prior user tool intent into classification or resolve retry eligibility from the last persisted `vector_search` tool call. Continue to use `build_retrieval_queries`, which already reuses `_last_vector_search_query` for retry text. Keep no-selection explicit document requests on the transparent selection notice path.

- [ ] **Step 7: Run focused intent/direct-RAG tests**

Run: `python -m pytest aquillm/apps/chat/tests/test_rag_intent.py aquillm/apps/chat/tests/test_document_search_intent.py aquillm/apps/chat/tests/test_direct_rag_pipeline.py aquillm/apps/chat/tests/test_rag_query.py -q`

Expected: PASS; a simple selected-collection definition performs retrieval and exactly one final synthesis, with no tool-selection generation.

- [ ] **Step 8: Commit the routing slice**

```bash
git add aquillm/apps/chat/services/rag_intent.py aquillm/apps/chat/services/rag_pipeline.py aquillm/apps/chat/tests/test_rag_intent.py aquillm/apps/chat/tests/test_document_search_intent.py aquillm/apps/chat/tests/test_direct_rag_pipeline.py aquillm/apps/chat/tests/test_rag_query.py
git commit -m "fix(rag): enforce selected collection evidence routing"
```

---

### Task 2: Bounded Mechanical Tool Calls and Authorized Text Recovery

**Files:**
- Modify: `aquillm/lib/llm/providers/complete_turn.py`
- Modify: `aquillm/lib/llm/providers/openai_tool_text.py`
- Modify: `aquillm/lib/llm/providers/openai.py`
- Modify: `aquillm/lib/llm/providers/openai_streaming.py`
- Modify: `aquillm/lib/llm/providers/fallback_heuristics.py`
- Modify: `aquillm/lib/llm/providers/visibility.py`
- Modify: `aquillm/apps/chat/tests/test_llm_complete_retry.py`
- Modify: `aquillm/apps/chat/tests/test_llm_interim_visibility.py`
- Modify: `aquillm/lib/llm/tests/test_openai_streaming.py`
- Create: `aquillm/lib/llm/tests/test_openai_tool_text.py`

**Interfaces:**
- Consumes: per-turn `UserMessage.tools`, `ToolChoice`, `LLMInterface.get_message`, and `extract_tool_call_from_text(text, raw_tools)`.
- Produces: `_resolve_tool_step_max_tokens(max_tokens, tool_choice_type) -> int` never exceeds 256; exact `Tool:<authorized_name> {json-object}` returns a normal tool-call payload; raw syntax is hidden.

- [ ] **Step 1: Write failing cap tests for zero and oversized environment values**

```python
@override_settings()
@patch.dict(os.environ, {
    "LLM_TOOL_STEP_MAX_TOKENS": "0",
    "LLM_TOOL_CALL_RETRY_MAX_TOKENS": "0",
})
def test_zero_tool_caps_resolve_to_safe_256_not_full_answer_budget(self):
    self.assertEqual(complete_turn._resolve_tool_step_max_tokens(12_288, "any"), 256)

@patch.dict(os.environ, {"LLM_TOOL_STEP_MAX_TOKENS": "4096"})
def test_oversized_tool_cap_is_clamped_to_256(self):
    self.assertEqual(complete_turn._resolve_tool_step_max_tokens(12_288, "auto"), 256)
```

- [ ] **Step 2: Run cap tests and verify the zero-cap case currently receives 12,288 tokens**

Run: `python -m pytest aquillm/apps/chat/tests/test_llm_complete_retry.py -q`

Expected: new assertions fail because `_env_optional_cap("LLM_TOOL_STEP_MAX_TOKENS", 2048, minimum=128)=0` currently falls back to the caller’s full answer budget and retry uses the larger cap.

- [ ] **Step 3: Implement a hard 256-token mechanical ceiling**

Define `_MECHANICAL_TOOL_MAX_TOKENS = 256`. Resolve both initial and hidden retry generations with `min(max_tokens, configured_positive_or_256, 256)`. A value at or below zero selects 256, not the caller budget. Hidden retry must use the resolved cap rather than `max(current_max_tokens, retry_cap)`. Preserve final/post-tool synthesis calculations.

- [ ] **Step 4: Write failing same-line tool parsing and visibility tests**

```python
def test_parses_authorized_same_line_tool_call():
    parsed = extract_tool_call_from_text(
        'Tool:vector_search {"search_string":"attensity","top_k":10}',
        [{"name": "vector_search"}],
    )
    assert parsed["tool_call_name"] == "vector_search"
    assert parsed["tool_call_input"] == {"search_string": "attensity", "top_k": 10}

@pytest.mark.parametrize("text", [
    'Tool:unknown {"x":1}',
    'Tool:vector_search {bad json}',
])
def test_rejects_unauthorized_or_malformed_same_line_tool_call(text):
    assert extract_tool_call_from_text(text, [{"name": "vector_search"}]) is None

def test_same_line_tool_transcript_is_never_visible():
    text = 'Tool:vector_search {"search_string":"attensity","top_k":10}'
    assert looks_like_raw_tool_transcript(text)
    assert visible_stream_content(text, raw_tools=[{"name": "vector_search"}], done=False) == ""
```

- [ ] **Step 5: Run parser/stream tests and verify the same-line form fails**

Run: `python -m pytest aquillm/lib/llm/tests/test_openai_tool_text.py aquillm/lib/llm/tests/test_openai_streaming.py aquillm/apps/chat/tests/test_llm_interim_visibility.py -q`

Expected: parser and raw-transcript detection fail for same-line JSON arguments.

- [ ] **Step 6: Implement anchored, authorization-bound textual tool recovery**

Match only the complete response form below, decode the JSON object, and use the existing allowed-tool set. Do not execute unknown tools or accept prose before/after the call.

```python
_SAME_LINE_TOOL_RE = re.compile(
    r"^\s*Tool\s*:\s*([A-Za-z_][\w.-]*)\s+(\{[\s\S]*\})\s*$",
    flags=re.IGNORECASE,
)
```

When parsing succeeds, set response text to `None` in both streaming and non-streaming OpenAI paths. Extend raw-transcript detection to recognize the same-line prefix so partial/final markup cannot render.

- [ ] **Step 7: Prevent cutoff routing from entering answer continuation or citation repair**

Add a regression where a required tool-selection response has `stop_reason="length"`, no visible text, and no structured call. Assert that `_deterministic_required_tool_call` returns the authorized `vector_search` once and the fake LLM records no unconstrained second generation. Malformed optional-tool routing returns the existing concise clean failure.

- [ ] **Step 8: Run the complete tool-provider test slice**

Run: `python -m pytest aquillm/apps/chat/tests/test_llm_complete_retry.py aquillm/apps/chat/tests/test_llm_interim_visibility.py aquillm/lib/llm/tests/test_openai_tool_text.py aquillm/lib/llm/tests/test_openai_streaming.py aquillm/lib/llm/tests/test_visibility.py -q`

Expected: PASS; initial/retry mechanical calls are non-thinking and at most 256 tokens; authorized textual recovery is structured and invisible.

- [ ] **Step 9: Commit the tool hardening slice**

```bash
git add aquillm/lib/llm/providers/complete_turn.py aquillm/lib/llm/providers/openai_tool_text.py aquillm/lib/llm/providers/openai.py aquillm/lib/llm/providers/openai_streaming.py aquillm/lib/llm/providers/fallback_heuristics.py aquillm/lib/llm/providers/visibility.py aquillm/apps/chat/tests/test_llm_complete_retry.py aquillm/apps/chat/tests/test_llm_interim_visibility.py aquillm/lib/llm/tests/test_openai_tool_text.py aquillm/lib/llm/tests/test_openai_streaming.py aquillm/lib/llm/tests/test_visibility.py
git commit -m "fix(llm): bound and recover mechanical tool calls"
```

---

### Task 3: Disconnect-Safe Direct RAG and Reconnect Resume

**Files:**
- Create: `aquillm/apps/chat/consumers/chat_transport.py`
- Modify: `aquillm/apps/chat/consumers/chat.py`
- Modify: `aquillm/apps/chat/consumers/chat_receive.py`
- Modify: `aquillm/apps/chat/consumers/chat_delta.py`
- Modify: `aquillm/apps/chat/services/rag_pipeline.py`
- Create: `aquillm/apps/chat/tests/test_chat_transport.py`
- Modify: `aquillm/apps/chat/tests/test_direct_rag_pipeline.py`
- Modify: `aquillm/apps/chat/tests/test_direct_rag_websocket_smoke.py`

**Interfaces:**
- Consumes: `consumer.send`, `run_direct_rag_turn`, `run_llm_spin`, and persisted `WSConversation` state.
- Produces: `is_client_disconnect(exc) -> bool`, `best_effort_send(consumer, *, text_data) -> bool`, and a shared direct-first pending-turn path used by both receive and reconnect.

- [ ] **Step 1: Write failing recognized-disconnect tests**

```python
@pytest.mark.asyncio
async def test_best_effort_send_swallows_recognized_disconnect_only():
    consumer = FakeConsumer(RuntimeError("Unexpected ASGI message 'websocket.send', after sending 'websocket.close'"))
    assert await best_effort_send(consumer, text_data="{}") is False
    assert consumer.transport_connected is False

@pytest.mark.asyncio
async def test_best_effort_send_reraises_application_failure():
    consumer = FakeConsumer(ValueError("serialization bug"))
    with pytest.raises(ValueError, match="serialization bug"):
        await best_effort_send(consumer, text_data="{}")
```

- [ ] **Step 2: Run the new transport tests and verify the helper is absent**

Run: `python -m pytest aquillm/apps/chat/tests/test_chat_transport.py -q`

Expected: FAIL on import because the transport boundary does not exist.

- [ ] **Step 3: Implement recognized-disconnect best-effort delivery**

Recognize `ConnectionResetError`, `BrokenPipeError`, Channels client-disconnect exceptions when importable, and the narrow ASGI-after-close runtime message. Mark `consumer.transport_connected=False` and return `False`; re-raise all other exceptions. Initialize/reset the flag on consumer construction/connect and set it false on `disconnect`.

- [ ] **Step 4: Write a failing regression for disconnect during synthesis**

```python
async def test_stream_disconnect_does_not_fall_back_to_tool_loop_and_answer_persists(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    convo = _user_convo("what is attensity")
    consumer = _consumer(convo, [203])
    consumer.transport_connected = True

    async def raw_send(*, text_data):
        raise RuntimeError(
            "Unexpected ASGI message 'websocket.send', after sending 'websocket.close'"
        )

    consumer.send = raw_send

    async def fake_synth(_llm_if, working_convo, _packet, *, stream_func=None):
        await stream_func({"content": "partial"})
        return working_convo + [
            AssistantMessage(
                content="Grounded answer [doc:doc-a chunk:1]", stop_reason="end_turn"
            )
        ]

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", lambda *_args: _results_payload())
    monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", fake_synth)
    stream = lambda payload: best_effort_send(
        consumer, text_data=json.dumps({"stream": payload})
    )
    outcome = await run_direct_rag_turn(
        consumer, SimpleNamespace(), convo, stream_func=stream
    )
    assert outcome == "handled"
    assert consumer.convo[-1].content == "Grounded answer [doc:doc-a chunk:1]"
    assert consumer.transport_connected is False
```

Also assert `send_conversation_delta` saves before its best-effort send and advances `last_sent_sequence` only after a successful send.

- [ ] **Step 5: Run direct-RAG WebSocket tests and verify the disconnect currently returns `skipped`**

Run: `python -m pytest aquillm/apps/chat/tests/test_direct_rag_pipeline.py aquillm/apps/chat/tests/test_direct_rag_websocket_smoke.py aquillm/apps/chat/tests/test_chat_transport.py -q`

Expected: the disconnect propagates through synthesis, `run_direct_rag_turn` catches it as a model/retrieval error, and the test observes fallback eligibility.

- [ ] **Step 6: Route all stream/delta sends through the transport helper**

Use `best_effort_send` in `_send_stream_payload` and the final delta path. Because transport exceptions no longer enter `run_direct_rag_turn`, successful synthesis sets `consumer.convo` and returns `handled`. Persistence happens before best-effort notification.

- [ ] **Step 7: Resume pending reconnect turns through the same direct-first path**

On `connect`, after loading/rebinding the conversation, detect a trailing user message with no assistant result. Invoke `run_direct_rag_turn` first; invoke `run_llm_spin` only when it returns `skipped`. Add a smoke test proving a persisted selected-collection `what is attensity` turn performs direct retrieval once and does not enter tool spin.

- [ ] **Step 8: Run WebSocket/direct-RAG tests**

Run: `python -m pytest aquillm/apps/chat/tests/test_chat_transport.py aquillm/apps/chat/tests/test_direct_rag_pipeline.py aquillm/apps/chat/tests/test_direct_rag_websocket_smoke.py aquillm/apps/chat/tests/test_chat_consumer_append.py -q`

Expected: PASS; transport loss cannot cause direct RAG and the tool loop to run for the same turn.

- [ ] **Step 9: Commit the transport slice**

```bash
git add aquillm/apps/chat/consumers/chat_transport.py aquillm/apps/chat/consumers/chat.py aquillm/apps/chat/consumers/chat_receive.py aquillm/apps/chat/consumers/chat_delta.py aquillm/apps/chat/services/rag_pipeline.py aquillm/apps/chat/tests/test_chat_transport.py aquillm/apps/chat/tests/test_direct_rag_pipeline.py aquillm/apps/chat/tests/test_direct_rag_websocket_smoke.py
git commit -m "fix(chat): persist direct rag across disconnects"
```

---

### Task 4: Token-Safe, Capability-Aware Concurrent Reranking

**Files:**
- Create: `aquillm/apps/documents/services/chunk_rerank_budget.py`
- Modify: `aquillm/apps/documents/services/chunk_rerank_config.py`
- Modify: `aquillm/apps/documents/services/chunk_rerank_local_vllm.py`
- Modify: `aquillm/apps/documents/services/rag_cache.py`
- Create: `aquillm/apps/documents/tests/test_chunk_rerank_budget.py`
- Modify: `aquillm/apps/documents/tests/test_chunk_search_graph_overlay.py`
- Modify: `aquillm/apps/documents/tests/test_chunk_rerank_cache.py`

**Interfaces:**
- Consumes: query, candidate text/multimodal payloads, configured reranker base URL/model, existing rerank cache, and `requests.post`.
- Produces: `trim_rerank_pair(query, document, max_pair_tokens, reserve_tokens) -> tuple[str, str]`; successful capability record keyed by base URL/model; complete scores via supported batch or bounded single-pair concurrency.

- [ ] **Step 1: Write failing token-budget tests**

```python
def test_trim_rerank_pair_respects_token_budget_and_preserves_query():
    query = "attensity calibration"
    document = "evidence " * 4000
    trimmed_query, trimmed_document = trim_rerank_pair(
        query, document, max_pair_tokens=900, reserve_tokens=64
    )
    assert trimmed_query == query
    assert count_rerank_tokens(trimmed_query, trimmed_document) <= 836
    assert trimmed_document

def test_trim_is_deterministic_for_unicode_academic_text():
    pair_a = trim_rerank_pair("β calibration", "λ evidence " * 2000, 900, 64)
    pair_b = trim_rerank_pair("β calibration", "λ evidence " * 2000, 900, 64)
    assert pair_a == pair_b
```

- [ ] **Step 2: Run token-budget tests and verify the module is absent**

Run: `python -m pytest aquillm/apps/documents/tests/test_chunk_rerank_budget.py -q`

Expected: FAIL on import.

- [ ] **Step 3: Implement token-based trimming with the existing tiktoken dependency**

Use a module-level cached encoding, count query and document tokens rather than characters, reserve configured template overhead, and truncate the document token list deterministically. Keep `RAG_RERANK_DOC_CHAR_LIMIT` only as an earlier memory/serialization guard; the token budget is authoritative before requests.

- [ ] **Step 4: Write failing capability and concurrency tests**

```python
def test_cached_single_score_capability_skips_known_invalid_batch_probe(monkeypatch):
    capability = {
        "endpoint": "http://local/score",
        "shape": "score_single_text_pair",
    }
    monkeypatch.setattr(
        rag_cache, "get_cached_rerank_capability", lambda _base, _model: capability
    )
    monkeypatch.setattr(
        local_vllm,
        "_score_documents_concurrently",
        lambda **_kwargs: [(0, 0.9), (1, 0.2)],
    )
    monkeypatch.setattr(
        local_vllm.requests,
        "post",
        lambda *_args, **_kwargs: pytest.fail("cached single capability must skip probes"),
    )
    ranked = local_vllm.rerank_via_local_vllm(TextChunk, "attensity", rows, 2)
    assert [row.pk for row in ranked] == [rows[0].pk, rows[1].pk]

def test_single_pair_fallback_uses_bounded_concurrency(monkeypatch):
    lock = threading.Lock()
    active = 0
    peak = 0

    def score_one(*, index, document, **_kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return index, float(len(document))

    monkeypatch.setattr(local_vllm, "_score_one_document", score_one)
    scores = local_vllm._score_documents_concurrently(
        endpoint="http://local/score",
        query="attensity",
        documents=[f"document-{index}" for index in range(12)],
        headers={},
        timeout=5.0,
        max_workers=4,
        payload_shape="score_single_text_pair",
    )
    assert 1 < peak <= 4
    assert sorted(index for index, _score in scores) == list(range(12))

def test_incomplete_or_nonfinite_scores_do_not_populate_cache(monkeypatch):
    stored_rankings: list[list[int]] = []
    monkeypatch.setattr(
        rag_cache,
        "set_cached_rerank_result",
        lambda _query, _candidates, _top_k, _model, ranked: stored_rankings.append(ranked),
    )
    monkeypatch.setattr(
        local_vllm,
        "_score_documents_concurrently",
        lambda **_kwargs: [(0, 0.9), (1, float("nan"))],
    )
    result = local_vllm.rerank_via_local_vllm(TextChunk, "attensity", rows, 2)
    assert list(result) == []
    assert stored_rankings == []
```

- [ ] **Step 5: Run reranker tests and verify sequential/probing assertions fail**

Run: `python -m pytest aquillm/apps/documents/tests/test_chunk_rerank_budget.py aquillm/apps/documents/tests/test_chunk_search_graph_overlay.py aquillm/apps/documents/tests/test_chunk_rerank_cache.py -q`

Expected: current code sends batch shapes before cached single scoring and loops through candidates sequentially.

- [ ] **Step 6: Cache payload shape and score single pairs with bounded concurrency**

Store a structured successful capability with exact keys `{"endpoint": "<resolved URL>", "shape": "rerank_documents"}`, where `shape` is one of `rerank_documents`, `score_batch_text_pairs`, or `score_single_text_pair`, keyed by base URL/model. Try a cached capability first and invalidate it only after a relevant endpoint/validation failure. For single-pair scoring, use `ThreadPoolExecutor(max_workers=RAG_RERANK_SCORE_CONCURRENCY)` with a default of 6 and clamp 1..16. Preserve candidate indices, require a complete finite score set for strict evaluation, and cache only a complete successful ranking.

- [ ] **Step 7: Preserve fail-open production and fail-closed evaluation semantics**

Run existing graph-overlay strict-evaluation cases for empty, missing, NaN, and infinite scores. Production returns the deterministic fallback order when the local reranker returns no valid ranking; strict evaluation raises `StrictRerankUnavailable` exactly as before.

- [ ] **Step 8: Run all document retrieval/reranker tests**

Run: `python -m pytest aquillm/apps/documents/tests/test_chunk_rerank_budget.py aquillm/apps/documents/tests/test_chunk_rerank_cache.py aquillm/apps/documents/tests/test_chunk_search_graph_overlay.py aquillm/apps/documents/tests/test_chunk_search_candidate_tuning.py aquillm/apps/documents/tests/test_chunk_search_diagnostics.py aquillm/apps/documents/tests/test_chunk_search_fusion.py aquillm/apps/documents/tests/test_chunk_search_query_cache.py aquillm/apps/documents/tests/test_hybrid_graph_authorization.py aquillm/apps/documents/tests/test_hybrid_graph_failures.py -q`

Expected: PASS; a warmed capability emits no known-invalid batch probe and single scoring is bounded-concurrent.

- [ ] **Step 9: Commit the reranker slice**

```bash
git add aquillm/apps/documents/services/chunk_rerank_budget.py aquillm/apps/documents/services/chunk_rerank_config.py aquillm/apps/documents/services/chunk_rerank_local_vllm.py aquillm/apps/documents/services/rag_cache.py aquillm/apps/documents/tests/test_chunk_rerank_budget.py aquillm/apps/documents/tests/test_chunk_search_graph_overlay.py aquillm/apps/documents/tests/test_chunk_rerank_cache.py
git commit -m "perf(rag): parallelize capability aware reranking"
```

---

### Task 5: Safe Token and Stage-Latency Telemetry

**Files:**
- Modify: `aquillm/lib/llm/types/response.py`
- Create: `aquillm/lib/llm/providers/request_observability.py`
- Modify: `aquillm/lib/llm/providers/openai.py`
- Modify: `aquillm/lib/llm/providers/openai_streaming.py`
- Modify: `aquillm/lib/llm/providers/complete_turn.py`
- Modify: `aquillm/apps/chat/services/rag_metrics.py`
- Modify: `aquillm/apps/chat/services/rag_pipeline.py`
- Create: `aquillm/lib/llm/tests/test_request_observability.py`
- Modify: `aquillm/lib/llm/tests/test_openai_streaming.py`
- Modify: `aquillm/apps/chat/tests/test_direct_rag_pipeline.py`

**Interfaces:**
- Consumes: OpenAI-compatible `usage`, `completion_tokens_details.reasoning_tokens`, finish reason, configured/effective max tokens, thinking flag, and monotonic timestamps.
- Produces: optional `LLMResponse.reasoning_usage: int | None`; safe structured `obs.llm.request_completed` and direct-RAG stage logs keyed by an opaque correlation ID.

- [ ] **Step 1: Write failing usage extraction tests**

```python
def test_extract_usage_reports_reasoning_when_provider_supplies_it():
    usage = SimpleNamespace(
        prompt_tokens=120,
        completion_tokens=40,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=31),
    )
    assert extract_usage(usage) == UsageCounts(120, 40, 31)

def test_extract_usage_marks_reasoning_unavailable_instead_of_inventing_zero():
    usage = SimpleNamespace(prompt_tokens=120, completion_tokens=40)
    assert extract_usage(usage).reasoning_tokens is None
```

- [ ] **Step 2: Run observability tests and verify the helper/type field is absent**

Run: `python -m pytest aquillm/lib/llm/tests/test_request_observability.py aquillm/lib/llm/tests/test_openai_streaming.py -q`

Expected: FAIL on import/field access.

- [ ] **Step 3: Implement safe usage and timing helpers**

Add immutable `UsageCounts(prompt_tokens, completion_tokens, reasoning_tokens)` and extraction using attribute access only. Add optional `reasoning_usage` to `LLMResponse` without changing aggregate `AssistantMessage.usage`. Generate an opaque UUID correlation ID per turn and pass a safe stage label (`tool_selection`, `tool_retry`, `direct_synthesis`, `post_tool_synthesis`, `general_answer`) through provider kwargs.

- [ ] **Step 4: Capture streaming/non-streaming provider metrics**

Record monotonic request start, first received stream chunk carrying content/reasoning/tool data, and completion. Log configured/effective max, thinking requested, prompt/completion counts, `reasoning_tokens` as an integer or the literal status `unavailable`, finish reason, TTFT, and total duration. Do not include message content, arguments, tool schemas, model prompts, or environment values.

- [ ] **Step 5: Extend direct-RAG stage timing without private identifiers**

Reuse the correlation ID for intent, query construction, retrieval, evidence packing, synthesis, and persistence. Log retrieved/retained counts and safe graph status only. Add a test that serializes captured log kwargs and asserts query text, document text, collection/document IDs, and tool arguments are absent.

- [ ] **Step 6: Run provider and RAG telemetry tests**

Run: `python -m pytest aquillm/lib/llm/tests/test_request_observability.py aquillm/lib/llm/tests/test_openai_streaming.py aquillm/apps/chat/tests/test_direct_rag_pipeline.py -q`

Expected: PASS; missing provider reasoning counts remain explicitly unavailable.

- [ ] **Step 7: Commit the observability slice**

```bash
git add aquillm/lib/llm/types/response.py aquillm/lib/llm/providers/request_observability.py aquillm/lib/llm/providers/openai.py aquillm/lib/llm/providers/openai_streaming.py aquillm/lib/llm/providers/complete_turn.py aquillm/apps/chat/services/rag_metrics.py aquillm/apps/chat/services/rag_pipeline.py aquillm/lib/llm/tests/test_request_observability.py aquillm/lib/llm/tests/test_openai_streaming.py aquillm/apps/chat/tests/test_direct_rag_pipeline.py
git commit -m "feat(obs): trace safe rag token and latency stages"
```

---

### Task 6: Integrated Quality, Performance, Deployment, and Live Verification

**Files:**
- Modify only if regression output requires a scoped fix: files already listed in Tasks 1-5.
- Do not modify or stage: `.env`, `.codex-ssh/`, remote environment files, transcription compose definitions.

**Interfaces:**
- Consumes: all completed implementation slices and existing evaluation/build commands.
- Produces: pushed `origin/development`, remotely pulled commit, targeted healthy web/worker deployment, and measured live evidence for routing/retrieval/reranker/synthesis behavior.

- [ ] **Step 1: Run focused Python regression suites**

Run:

```bash
python -m pytest aquillm/apps/chat/tests/test_rag_intent.py aquillm/apps/chat/tests/test_document_search_intent.py aquillm/apps/chat/tests/test_direct_rag_pipeline.py aquillm/apps/chat/tests/test_direct_rag_websocket_smoke.py aquillm/apps/chat/tests/test_chat_transport.py aquillm/apps/chat/tests/test_llm_complete_retry.py aquillm/apps/chat/tests/test_llm_interim_visibility.py aquillm/lib/llm/tests/test_openai_tool_text.py aquillm/lib/llm/tests/test_openai_streaming.py aquillm/lib/llm/tests/test_visibility.py aquillm/lib/llm/tests/test_request_observability.py aquillm/apps/documents/tests/test_chunk_rerank_budget.py aquillm/apps/documents/tests/test_chunk_rerank_cache.py aquillm/apps/documents/tests/test_chunk_search_graph_overlay.py -q
```

Expected: PASS with no unexpected deselections or xfails.

- [ ] **Step 2: Run wider chat/document regression suites and lint**

Run: `python -m pytest aquillm/apps/chat/tests aquillm/apps/documents/tests -q`

Run: `ruff check aquillm/apps/chat aquillm/apps/documents aquillm/lib/llm`

Expected: PASS. If unrelated pre-existing failures occur, record exact failing tests separately and do not claim they were introduced or fixed without evidence.

- [ ] **Step 3: Run offline RAG evaluation**

Run: `python aquillm/apps/chat/evals/run_rag_eval.py`

Expected: 6/6 existing scenarios pass; selected-collection answer cases retain evidence/citation requirements.

- [ ] **Step 4: Run citation UI tests and production build**

Run the repository’s existing package-manager commands for the citation-source/chunk-expansion Vitest slice and production React build from the frontend package directory identified in `package.json`.

Expected: citation tests and build PASS. Typecheck failures may only be classified as pre-existing after comparing exact error locations to the known baseline.

- [ ] **Step 5: Review diff and prove protected files are absent**

Run:

```bash
git status --short
git diff --check
git diff --stat origin/development...HEAD
git diff --name-only origin/development...HEAD
git ls-files | grep -E '(^|/)(\.env|authorized_keys|aquillm-dev2-deploy)' && exit 1 || true
```

Expected: only approved source/tests/docs are changed; `.codex-ssh/` remains untracked and unstaged; no environment or SSH material appears.

- [ ] **Step 6: Perform final local review and commit any verified integration adjustment**

If integration required a scoped adjustment, repeat its failing/passing test cycle, stage only explicit source/test paths, and commit with `fix(rag): harden retrieval latency integration`. Otherwise retain the task commits as the complete local history.

- [ ] **Step 6a: Verify ASGI vector-index prewarming**

If the deployed ASGI process logs `obs.rag.hnsw_prewarm_failed` while synchronous management commands succeed, keep the ORM call off the running event loop by scheduling one named daemon prewarm thread. Cover synchronous-command and asynchronous-server startup paths with focused tests; retrieval output must remain unchanged.

- [ ] **Step 7: Push local development commits**

Run: `git push origin development`

Expected: remote `development` advances to the verified local `HEAD`.

- [ ] **Step 8: Verify the remote identity and back up the unprinted environment**

Connect with the separate ephemeral deploy key, verify hostname `aquillm-dev2` and ED25519 fingerprint `SHA256:4cm7E/DfZY14GZL6OsC1IH7gGRupBXnlCKnFQ5ayDXo`, confirm the repo branch is `development`, and copy the remote environment to a timestamped path under `~/.config/aquillm/env-backups/` without printing it.

- [ ] **Step 9: Pull and deploy affected services only**

On `~/AquiLLM`, run a fast-forward-only pull of `origin/development`. Update only exact non-secret remote flags needed for the 256-token tool caps and reranker concurrency. Recreate/build only `web` and `worker` with the existing development/local compose files and `--no-deps`. Do not include or start `vllm_transcribe`; do not recreate vLLM model services unless a measured dependency requires it.

- [ ] **Step 10: Verify remote service and edge health**

Confirm web/worker status, web health endpoint, nginx status, and HTTPS 200. Confirm `compose-vllm_transcribe-1` remains exited. Inspect only filtered safe logs for startup exceptions and known-invalid reranker 400 responses.

- [ ] **Step 11: Monitor one live selected-collection request**

Use `what is attensity` or an equivalent selected-collection UI request. Verify safe logs show:

- routing below 10 ms and direct RAG selected;
- zero LLM tool-selection calls;
- concurrent query retrieval with successful fail-soft fusion;
- no known-invalid reranker 400 and no sequential per-candidate waterfall;
- retrieval plus reranking warm p95 target below 5 seconds after repeated warm trials;
- first visible synthesis token target below 10 seconds;
- exactly one final synthesis;
- persisted answer with real `[doc:... chunk:...]` citations and working chunk expansion;
- no duplicate generation if the browser reconnects.

If the live trace shows collection prompt-skill loading materializing every document,
replace it with candidate-only queries while preserving marked skills and skill-pack
semantics. If a Qwen reranker pair still crosses the 1,024-token limit, raise the
template reserve and retry only the overflowing pair with a tighter deterministic
budget. Keep thinking enabled for direct synthesis, but bound total completion to
4,096 tokens so reasoning cannot consume an unbounded latency tail before the cited
answer.

- [ ] **Step 12: Record rollout/rollback facts**

Report local and remote commit SHA, commands/tests and their observed results, service health, measured stage timings, citation verification, transcription-stopped status, and the remote environment backup path. If acceptance fails, restore the prior commit and backed-up exact flags, recreate only web/worker with `--no-deps`, and report the failing stage without claiming completion.
