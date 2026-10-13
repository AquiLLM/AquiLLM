"""Tests for the direct RAG pipeline orchestrator (Tasks 4 and 5)."""

from __future__ import annotations

from json import dumps
from types import SimpleNamespace
import pytest
from .persisted_direct_test_support import close_direct_worker as close_direct_worker
from .direct_rag_test_support import _synthesize, _admit_query, _retrieval, _assert_default_excerpt

pytestmark = pytest.mark.usefixtures("close_direct_worker")
from unittest.mock import AsyncMock

from apps.chat.consumers.chat_transport import best_effort_send
from apps.chat.services import rag_pipeline
from apps.chat.services.rag_pipeline import run_direct_rag_turn
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage

from .direct_rag_test_support import _consumer, _results_payload, _user_convo


async def test_retry_reuses_last_direct_vector_query(monkeypatch):
    """Retrying a direct retrieval reuses its resolved query, not the word retry."""
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    searched_queries: list[str] = []

    def fake_search(_consumer, query, _top_k):
        _admit_query(query)
        searched_queries.append(query)
        return consumer.case.rows(_results_payload())

    async def fake_synth(_llm_if, working_convo, _packet, *, stream_func=None):
        return await _synthesize(consumer, working_convo, _packet, "Retried answer [doc:doc-a chunk:1].", stream_func=stream_func)

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
    consumer = await _consumer(convo, [203])

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
    consumer = await _consumer(convo, [203])
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
        return await _synthesize(consumer, working_convo, _packet, "Grounded answer [doc:doc-a chunk:1]", stream_func=stream_func)

    monkeypatch.setattr(
        rag_pipeline,
        "_run_vector_search",
        lambda c, q, k: _retrieval(c, q, _results_payload()),
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
    _assert_default_excerpt(consumer, draft="Grounded answer")


async def test_multi_part_direct_rag_preserves_original_before_one_synthesis(
    monkeypatch,
):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    monkeypatch.setenv("RAG_DIRECT_MAX_QUERIES", "3")
    searched: list[str] = []
    synthesized: list = []

    def fake_search(consumer, query, top_k):
        _admit_query(query)
        searched.append(query)
        return consumer.case.rows(_results_payload())

    async def fake_synth(llm_if, convo, packet, *, stream_func=None):
        synthesized.append(packet)
        return await _synthesize(consumer, convo, packet, 'Cited answer ' + packet.citation_tokens[0], stream_func=stream_func)

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", fake_search)
    monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", fake_synth)

    convo = _user_convo("Explain what each paper is about? What overlaps between them?")
    consumer = await _consumer(convo, [1, 2, 3])

    outcome = await run_direct_rag_turn(
        consumer,
        SimpleNamespace(get_message=AsyncMock()),
        convo,
        stream_func=None,
    )

    assert outcome == "handled"
    assert searched == ["Explain what each paper is about? What overlaps between them?"]
    assert len(synthesized) == 1
    assert len(synthesized[0].citation_tokens) == 1



async def test_multi_query_retrieval_uses_successful_variants_when_one_fails(
    monkeypatch,
):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    monkeypatch.setenv("RAG_DIRECT_MAX_QUERIES", "3")
    monkeypatch.setenv("RAG_QUERY_REWRITE_ENABLED", "1")
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "16000")
    searched, ledgers, synthesized = [], [], []
    async def propose(**kwargs):
        return SimpleNamespace(text='{"rewrite":"Find selected documents for calibration"}')


    def fake_search(consumer, query, top_k):
        _admit_query(query)
        from lib.llm.turn_context import current_turn
        searched.append(query)
        ledgers.append(current_turn().budget)
        if query == "Find selected documents for calibration":
            raise RuntimeError("one query backend failure")
        return consumer.case.rows(_results_payload())

    async def fake_synth(llm_if, convo, packet, *, stream_func=None):
        synthesized.append(packet)
        return await _synthesize(consumer, convo, packet, 'Cited answer ' + packet.citation_tokens[0], stream_func=stream_func)

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", fake_search)
    monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", fake_synth)

    convo = _user_convo("Search selected documents for calibration")
    consumer = await _consumer(convo, [1, 2, 3])

    outcome = await run_direct_rag_turn(
        consumer,
        SimpleNamespace(base_args={}, get_message=propose),
        convo,
        stream_func=None,
    )

    assert outcome == "handled"
    assert len(synthesized) == 1
    assert synthesized[0].citation_tokens == [consumer.case.text("[doc:doc-a chunk:1]")]

    assert searched == ["Search selected documents for calibration", "Find selected documents for calibration"]
    assert ledgers[0] is ledgers[1]
    assert ledgers[0].actions_used == 2 and ledgers[0].planner_calls == 1

