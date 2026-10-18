"""Tests for the direct RAG pipeline orchestrator (Tasks 4 and 5)."""

from __future__ import annotations

from json import dumps
from types import SimpleNamespace
import pytest
from .persisted_direct_test_support import close_direct_worker as close_direct_worker
from .direct_rag_test_support import _synthesize, _admit_query, _retrieval, _assert_generated_answer

pytestmark = pytest.mark.usefixtures("close_direct_worker")
from unittest.mock import AsyncMock

from apps.chat.services import rag_pipeline
from apps.chat.services.rag_pipeline import run_direct_rag_turn
from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage

from .direct_rag_test_support import _consumer, _results_payload, _user_convo


async def test_graph_overlay_failure_keeps_direct_rag_on_vector_evidence(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    raw = _results_payload()
    # Task 16 fails graph work open to this unchanged baseline payload.
    monkeypatch.setattr(rag_pipeline, "_run_vector_search", lambda c, q, k: _retrieval(c, q, raw))

    async def fake_synth(llm_if, convo, packet, *, stream_func=None):
        assert packet.citation_tokens == [consumer.case.text("[doc:doc-a chunk:1]")]
        return await _synthesize(consumer, convo, packet, "Vector evidence remains usable [doc:doc-a chunk:1].", stream_func=stream_func)

    monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", fake_synth)
    convo = _user_convo("search the selected documents for calibration")
    consumer = await _consumer(convo, [1])

    outcome = await run_direct_rag_turn(
        consumer,
        SimpleNamespace(get_message=AsyncMock()),
        convo,
        stream_func=None,
    )

    assert outcome == "handled"
    _assert_generated_answer(consumer, draft="Vector evidence remains usable [doc:doc-a chunk:1].")


async def test_direct_rag_logs_safe_graph_contribution(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    raw = _results_payload()
    raw["_retrieval_diagnostics"] = {
        "graph_status": "hit",
        "graph_ms": 4.5,
        "graph_seed_count": 3,
        "graph_candidate_count": 2,
        "graph_path": ["private-node"],
    }
    monkeypatch.setattr(rag_pipeline, "_run_vector_search", lambda c, q, k: _retrieval(c, q, raw))

    async def fake_synth(llm_if, convo, packet, *, stream_func=None):
        return await _synthesize(consumer, convo, packet, 'Cited answer ' + packet.citation_tokens[0], stream_func=stream_func)

    captured: dict = {}

    def capture_metrics(**fields):
        captured.update(fields)

    monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", fake_synth)
    monkeypatch.setattr(rag_pipeline, "log_direct_rag_turn", capture_metrics)
    convo = _user_convo("search the selected documents for calibration")
    consumer = await _consumer(convo, [1])

    outcome = await run_direct_rag_turn(
        consumer,
        SimpleNamespace(get_message=AsyncMock()),
        convo,
        stream_func=None,
    )

    assert outcome == "handled"
    assert captured["graph_status"] == "hit"
    assert captured["graph_candidate_count"] == 2
    serialized = dumps(captured, sort_keys=True)
    for forbidden in (
        "private-node",
        "calibration",
        "Paper A",
        "doc-a",
        "search_string",
    ):
        assert forbidden not in serialized


async def test_handled_appends_synthetic_tool_messages(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    raw = _results_payload()

    captured: dict = {}

    def fake_search(consumer, query, top_k):
        _admit_query(query)
        return consumer.case.rows(raw)

    async def fake_synth(llm_if, convo, packet, *, stream_func=None):
        captured["convo"] = convo
        captured["packet"] = packet
        return await _synthesize(consumer, convo, packet, 'done ' + packet.citation_tokens[0], stream_func=stream_func)

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", fake_search)
    monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", fake_synth)

    convo = _user_convo("search the documents for calibration")
    consumer = await _consumer(convo, [1])
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


async def test_no_collections_prompts_selection(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    called: list = []
    monkeypatch.setattr(
        rag_pipeline, "_run_vector_search", lambda *a, **k: called.append(1)
    )

    convo = _user_convo("search the documents for calibration")
    consumer = await _consumer(convo, [])
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
        _admit_query(query)
        raise RuntimeError("retrieval backend down")

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", boom)

    convo = _user_convo("search the documents for calibration")
    consumer = await _consumer(convo, [1])
    llm_if = SimpleNamespace(get_message=AsyncMock())

    outcome = await run_direct_rag_turn(consumer, llm_if, convo, stream_func=None)

    assert outcome == "skipped"
    # consumer.convo must be untouched so the normal tool loop can run.
    assert consumer.convo is convo
    assert isinstance(consumer.convo[-1], UserMessage)
