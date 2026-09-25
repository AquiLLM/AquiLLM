"""Tests for the direct RAG pipeline orchestrator (Tasks 4 and 5)."""

from __future__ import annotations

from json import dumps
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from apps.chat.consumers.chat_transport import best_effort_send
from apps.chat.services import rag_metrics, rag_pipeline
from apps.chat.services.rag_pipeline import run_direct_rag_turn
from apps.chat.tests.chat_message_test_support import _FakeLLMInterface
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage
from lib.llm.types.response import LLMResponse

from .direct_rag_test_support import (
    _consumer as _consumer,
)
from .direct_rag_test_support import (
    _results_payload as _results_payload,
)
from .direct_rag_test_support import (
    _user_convo as _user_convo,
)


async def test_skipped_when_flag_off(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "0")
    convo = _user_convo("search the selected documents for calibration")
    consumer = _consumer(convo, [1])
    llm_if = SimpleNamespace(get_message=AsyncMock())

    outcome = await run_direct_rag_turn(consumer, llm_if, convo, stream_func=None)

    assert outcome == "skipped"
    llm_if.get_message.assert_not_called()


async def test_skipped_when_intent_not_rag(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    convo = _user_convo("hello there, how are you")
    consumer = _consumer(convo, [1])
    llm_if = SimpleNamespace(get_message=AsyncMock())

    outcome = await run_direct_rag_turn(consumer, llm_if, convo, stream_func=None)

    assert outcome == "skipped"
    llm_if.get_message.assert_not_called()


async def test_explicit_collection_synthesis_without_selection_is_handled(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    convo = _user_convo(
        "Hi aquillm can you tell me about the documents in this collection and synthesize things?"
    )
    consumer = _consumer(convo, [])
    llm_if = SimpleNamespace(get_message=AsyncMock())

    outcome = await run_direct_rag_turn(consumer, llm_if, convo, stream_func=None)

    assert outcome == "handled"
    llm_if.get_message.assert_not_called()
    assert "no collections are selected" in consumer.convo[-1].content.lower()


async def test_skipped_when_last_message_not_user(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    convo = Conversation(
        system="sys",
        messages=[
            UserMessage(content="hi"),
            AssistantMessage(content="hello", stop_reason="end_turn"),
        ],
    )
    consumer = _consumer(convo, [1])
    llm_if = SimpleNamespace(get_message=AsyncMock())

    outcome = await run_direct_rag_turn(consumer, llm_if, convo, stream_func=None)

    assert outcome == "skipped"


async def test_handled_retrieves_before_llm(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    order: list[str] = []
    raw = _results_payload()

    def fake_search(consumer, query, top_k):
        order.append("retrieval")
        return raw

    async def fake_synth(llm_if, convo, packet, *, stream_func=None):
        order.append("synthesis")
        return convo + [
            AssistantMessage(
                content="Answer [doc:doc-a chunk:1].", stop_reason="end_turn"
            )
        ]

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", fake_search)
    monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", fake_synth)

    convo = _user_convo("search the selected documents for calibration")
    consumer = _consumer(convo, [1])
    llm_if = SimpleNamespace(get_message=AsyncMock())

    outcome = await run_direct_rag_turn(consumer, llm_if, convo, stream_func=None)

    assert outcome == "handled"
    assert order == ["retrieval", "synthesis"]
    llm_if.get_message.assert_not_called()
    assert "Answer" in consumer.convo[-1].content


async def test_candidate_pool_is_larger_than_final_packet(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    monkeypatch.setenv("RAG_DIRECT_TOP_K", "2")
    monkeypatch.setenv("RAG_DIRECT_MAX_QUERIES", "1")
    observed = []

    def search(_consumer, _query, top_k):
        observed.append(top_k)
        rows = [dict(_results_payload()["result"][0], chunk_id=i,
                     citation=f"[doc:doc-a chunk:{i}]") for i in range(1, 6)]
        rows.append(dict(rows[0], doc_id="doc-b", chunk_id=6,
                         title="Paper B", citation="[doc:doc-b chunk:6]"))
        return {"result": rows[:top_k]}

    async def synth(_llm_if, convo, packet, **kwargs):
        assert [row["doc_id"] for row in packet.chunks] == ["doc-a", "doc-b"]
        return convo + [AssistantMessage(content="Answer", stop_reason="end_turn")]

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", search)
    monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", synth)
    convo = _user_convo("compare the selected papers")
    consumer = _consumer(convo, [1])
    assert await run_direct_rag_turn(consumer, object(), convo) == "handled"
    assert observed == [6]


async def test_selected_collection_definition_uses_direct_rag(monkeypatch):
    """A terse selected-collection question must skip model tool selection."""
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    searched_queries: list[str] = []

    def fake_search(_consumer, query, _top_k):
        searched_queries.append(query)
        return _results_payload()

    async def fake_synth(_llm_if, working_convo, _packet, *, stream_func=None):
        return working_convo + [
            AssistantMessage(
                content="Attensity answer [doc:doc-a chunk:1].",
                stop_reason="end_turn",
            )
        ]

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", fake_search)
    monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", fake_synth)

    convo = _user_convo("what is attensity")
    consumer = _consumer(convo, [203])
    llm_if = SimpleNamespace(get_message=AsyncMock())

    outcome = await run_direct_rag_turn(consumer, llm_if, convo, stream_func=None)

    assert outcome == "handled"
    assert searched_queries == ["what is attensity"]
    llm_if.get_message.assert_not_called()


def test_direct_rag_metrics_accept_safe_optional_graph_fields(monkeypatch):
    captured: dict = {}

    def capture(event, **fields):
        captured["event"] = event
        captured.update(fields)

    monkeypatch.setattr(rag_metrics.logger, "info", capture)

    rag_metrics.log_direct_rag_turn(
        correlation_id="0f22db7309f04ab0a4676cdb5a76f962",
        intent_ms=1.1,
        query_ms=2.2,
        retrieval_ms=3.3,
        evidence_ms=4.4,
        synthesis_ms=5.5,
        persistence_ms=0.2,
        total_ms=16.5,
        retrieved_count=2,
        retained_count=2,
        retrieval_status="results_found",
        graph_ms=0.7,
        graph_seed_count=3,
        graph_candidate_count=1,
        graph_status="hit",
        graph_algorithm_signature="a" * 64,
        graph_version_signature="b" * 64,
    )

    assert captured["event"] == "rag_direct_turn"
    assert captured["correlation_id"] == "0f22db7309f04ab0a4676cdb5a76f962"
    assert captured["retained_count"] == 2
    assert captured["persistence_ms"] == 0.2
    assert captured["graph_status"] == "hit"
    assert captured["graph_seed_count"] == 3
    assert captured["graph_candidate_count"] == 1
    assert captured["graph_algorithm_signature"] == "a" * 64
    assert captured["graph_version_signature"] == "b" * 64


def test_direct_rag_metrics_omit_poisoned_graph_fields(monkeypatch):
    captured: dict = {}

    def capture(event, **fields):
        captured["event"] = event
        captured.update(fields)

    monkeypatch.setattr(rag_metrics.logger, "info", capture)

    rag_metrics.log_direct_rag_turn(
        intent_ms=1.0,
        query_ms=1.0,
        retrieval_ms=1.0,
        evidence_ms=1.0,
        synthesis_ms=1.0,
        total_ms=5.0,
        retrieved_count=1,
        retrieval_status="results_found",
        graph_ms=float("nan"),
        graph_seed_count=True,
        graph_candidate_count=21,
        graph_status="query=private text",
        graph_algorithm_signature="PRIVATE-LABEL" * 6,
        graph_version_signature="A" * 64,
    )

    assert captured["event"] == "rag_direct_turn"
    assert not any(key.startswith("graph_") for key in captured)


async def test_end_to_end_real_synthesis_single_llm_call(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    monkeypatch.setenv("RAG_ENFORCE_CHUNK_CITATIONS", "1")
    order: list[str] = []
    raw = _results_payload()

    def fake_search(consumer, query, top_k):
        order.append("retrieval")
        return raw

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", fake_search)

    answer = (
        "The paper describes a calibration method using flat fields and dark "
        "frames to remove instrument signatures [doc:doc-a chunk:1]."
    )
    llm_if = _FakeLLMInterface(
        [
            LLMResponse(
                text=answer,
                tool_call=None,
                stop_reason="end_turn",
                input_usage=1,
                output_usage=1,
            )
        ]
    )

    # Track that the first LLM call happens only after retrieval.
    original_get_message = llm_if.get_message

    async def tracked_get_message(*args, **kwargs):
        order.append("get_message")
        return await original_get_message(*args, **kwargs)

    llm_if.get_message = tracked_get_message

    convo = _user_convo("search the selected documents for the calibration method")
    consumer = _consumer(convo, [1])

    outcome = await run_direct_rag_turn(consumer, llm_if, convo, stream_func=None)

    assert outcome == "handled"
    assert order[0] == "retrieval"
    assert order.count("get_message") == 1
    assert "thinking_budget" not in llm_if.calls[0]
    assert "calibration" in consumer.convo[-1].content.lower()
    assert "[doc:doc-a chunk:1]" in consumer.convo[-1].content


async def test_direct_rag_no_results_returns_notice_without_llm(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    raw = {
        "result": [],
        "retrieval_status": "no_results",
        "retrieval_message": (
            'I searched the selected documents for "dark matter", '
            "but retrieval returned no relevant passages."
        ),
        "retrieval_diagnostics": {
            "doc_count": 1,
            "vector_error": None,
        },
    }
    monkeypatch.setattr(rag_pipeline, "_run_vector_search", lambda c, q, k: raw)

    convo = _user_convo("search the documents for dark matter")
    consumer = _consumer(convo, [1])
    # Empty response list: any LLM call would raise IndexError.
    llm_if = _FakeLLMInterface([])

    outcome = await run_direct_rag_turn(consumer, llm_if, convo, stream_func=None)

    assert outcome == "handled"
    assert llm_if.calls == []
    assert "no relevant passages" in consumer.convo[-1].content.lower()


async def test_direct_rag_figure_request_embeds_image(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    raw = _results_payload()
    raw["result"][0]["image_url"] = "/aquillm/document_image/doc-a/"
    monkeypatch.setattr(rag_pipeline, "_run_vector_search", lambda c, q, k: raw)

    answer = (
        "The figure shows calibration drift across magnitude bins for the survey "
        "sample [doc:doc-a chunk:1]."
    )
    llm_if = _FakeLLMInterface(
        [
            LLMResponse(
                text=answer,
                tool_call=None,
                stop_reason="end_turn",
                input_usage=1,
                output_usage=1,
            )
        ]
    )

    convo = _user_convo("show me the figure for the calibration method")
    consumer = _consumer(convo, [1])

    outcome = await run_direct_rag_turn(consumer, llm_if, convo, stream_func=None)

    assert outcome == "handled"
    assert "/aquillm/document_image/doc-a/" in consumer.convo[-1].content


async def test_handled_appends_synthetic_tool_messages(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    raw = _results_payload()

    captured: dict = {}

    def fake_search(consumer, query, top_k):
        return raw

    async def fake_synth(llm_if, convo, packet, *, stream_func=None):
        captured["convo"] = convo
        captured["packet"] = packet
        return convo + [AssistantMessage(content="done", stop_reason="end_turn")]

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", fake_search)
    monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", fake_synth)

    convo = _user_convo("search the documents for calibration")
    consumer = _consumer(convo, [1])
    llm_if = SimpleNamespace(get_message=AsyncMock())

    await run_direct_rag_turn(consumer, llm_if, convo, stream_func=None)

    synth_convo = captured["convo"]
    # user -> assistant(tool_call) -> tool(result)
    assert isinstance(synth_convo[-1], ToolMessage)
    assert synth_convo[-1].for_whom == "assistant"
    assert synth_convo[-1].tool_name == "vector_search"
    assert isinstance(synth_convo[-2], AssistantMessage)
    assert synth_convo[-2].tool_call_name == "vector_search"
    assert captured["packet"].chunks



async def test_history_tool_choice_is_not_preempted_by_selected_collection(monkeypatch):
    from apps.chat.refs import ChatRef
    from apps.chat.services.tool_wiring.memory import search_past_chats_tool
    from lib.llm.types.tools import ToolChoice

    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    for query in ("Remind me what we said about the papers before.", "retry"):
        convo = _user_convo(query)
        convo[-1].tools = [search_past_chats_tool(None, ChatRef(None))]
        convo[-1].tool_choice = ToolChoice(type="any")
        consumer = _consumer(convo, [1])
        search = Mock(return_value=_results_payload())
        monkeypatch.setattr(rag_pipeline, "_run_vector_search", search)
        assert await run_direct_rag_turn(consumer, object(), convo) == "skipped"
        search.assert_not_called()



async def test_multi_part_direct_rag_searches_variants_before_one_synthesis(
    monkeypatch,
):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    monkeypatch.setenv("RAG_DIRECT_MAX_QUERIES", "3")
    searched: list[str] = []
    synthesized: list = []

    def fake_search(consumer, query, top_k):
        searched.append(query)
        chunk_id = len(searched)
        return {
            "result": [
                {
                    "rank": 1,
                    "chunk_id": chunk_id,
                    "doc_id": f"doc-{chunk_id}",
                    "title": f"Paper {chunk_id}",
                    "text": f"Evidence for query {chunk_id}.",
                    "citation": f"[doc:doc-{chunk_id} chunk:{chunk_id}]",
                }
            ],
            "retrieval_status": "results_found",
            "retrieved_count": 1,
            "retrieved_documents": [f"Paper {chunk_id}"],
        }

    async def fake_synth(llm_if, convo, packet, *, stream_func=None):
        synthesized.append(packet)
        return convo + [
            AssistantMessage(content="Cited answer", stop_reason="end_turn")
        ]

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", fake_search)
    monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", fake_synth)

    convo = _user_convo("Explain what each paper is about? What overlaps between them?")
    consumer = _consumer(convo, [1, 2, 3])

    outcome = await run_direct_rag_turn(
        consumer,
        SimpleNamespace(get_message=AsyncMock()),
        convo,
        stream_func=None,
    )

    assert outcome == "handled"
    assert searched == [
        "Explain what each paper is about? What overlaps between them?",
        "Explain what each paper is about",
        "What overlaps between them",
    ]
    assert len(synthesized) == 1
    assert len(synthesized[0].citation_tokens) == 3



async def test_multi_query_retrieval_uses_successful_variants_when_one_fails(
    monkeypatch,
):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    monkeypatch.setenv("RAG_DIRECT_MAX_QUERIES", "3")
    synthesized: list = []

    def fake_search(consumer, query, top_k):
        if query == "What overlaps between them":
            raise RuntimeError("one query backend failure")
        return _results_payload()

    async def fake_synth(llm_if, convo, packet, *, stream_func=None):
        synthesized.append(packet)
        return convo + [
            AssistantMessage(content="Cited answer", stop_reason="end_turn")
        ]

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", fake_search)
    monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", fake_synth)

    convo = _user_convo("Explain what each paper is about? What overlaps between them?")
    consumer = _consumer(convo, [1, 2, 3])

    outcome = await run_direct_rag_turn(
        consumer,
        SimpleNamespace(get_message=AsyncMock()),
        convo,
        stream_func=None,
    )

    assert outcome == "handled"
    assert len(synthesized) == 1
    assert synthesized[0].citation_tokens == ["[doc:doc-a chunk:1]"]



async def test_no_collections_prompts_selection(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    called: list = []
    monkeypatch.setattr(
        rag_pipeline, "_run_vector_search", lambda *a, **k: called.append(1)
    )

    convo = _user_convo("search the documents for calibration")
    consumer = _consumer(convo, [])
    llm_if = SimpleNamespace(get_message=AsyncMock())

    outcome = await run_direct_rag_turn(consumer, llm_if, convo, stream_func=None)

    assert outcome == "handled"
    assert not called
    last = consumer.convo[-1]
    assert isinstance(last, AssistantMessage)
    assert "collection" in last.content.lower()



async def test_retrieval_failure_falls_back_to_skipped(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")

    def boom(consumer, query, top_k):
        raise RuntimeError("retrieval backend down")

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", boom)

    convo = _user_convo("search the documents for calibration")
    consumer = _consumer(convo, [1])
    llm_if = SimpleNamespace(get_message=AsyncMock())

    outcome = await run_direct_rag_turn(consumer, llm_if, convo, stream_func=None)

    assert outcome == "skipped"
    # consumer.convo must be untouched so the normal tool loop can run.
    assert consumer.convo is convo
    assert isinstance(consumer.convo[-1], UserMessage)



async def test_retry_reuses_last_direct_vector_query(monkeypatch):
    """Retrying a direct retrieval reuses its resolved query, not the word retry."""
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    searched_queries: list[str] = []

    def fake_search(_consumer, query, _top_k):
        searched_queries.append(query)
        return _results_payload()

    async def fake_synth(_llm_if, working_convo, _packet, *, stream_func=None):
        return working_convo + [
            AssistantMessage(
                content="Retried answer [doc:doc-a chunk:1].",
                stop_reason="end_turn",
            )
        ]

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", fake_search)
    monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", fake_synth)

    convo = Conversation(
        system="sys",
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

    outcome = await run_direct_rag_turn(
        consumer,
        SimpleNamespace(get_message=AsyncMock()),
        convo,
        stream_func=None,
    )

    assert outcome == "handled"
    assert searched_queries == ["attensity"]



async def test_stream_disconnect_does_not_turn_direct_rag_into_tool_fallback(
    monkeypatch,
):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    convo = _user_convo("what is attensity")
    consumer = _consumer(convo, [203])
    consumer.transport_connected = True

    async def raw_send(*, text_data):
        raise RuntimeError(
            "Unexpected ASGI message 'websocket.send', after sending 'websocket.close'"
        )

    consumer.send = raw_send

    async def safe_stream(payload):
        await best_effort_send(
            consumer,
            text_data=dumps({"stream": payload}),
        )

    async def fake_synth(_llm_if, working_convo, _packet, *, stream_func=None):
        await stream_func({"content": "partial", "done": False})
        return working_convo + [
            AssistantMessage(
                content="Grounded answer [doc:doc-a chunk:1]",
                stop_reason="end_turn",
            )
        ]

    monkeypatch.setattr(
        rag_pipeline,
        "_run_vector_search",
        lambda *_args: _results_payload(),
    )
    monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", fake_synth)

    outcome = await run_direct_rag_turn(
        consumer,
        SimpleNamespace(),
        convo,
        stream_func=safe_stream,
    )

    assert outcome == "handled"
    assert consumer.transport_connected is False
    assert consumer.convo[-1].content == "Grounded answer [doc:doc-a chunk:1]"
