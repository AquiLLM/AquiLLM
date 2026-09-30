"""Opt-in observations of the actual direct-RAG handoff."""

import asyncio
from types import SimpleNamespace

from apps.chat.services import rag_pipeline
from lib import replay_observation
from lib.evidence_observation import observe
from lib.llm.types.messages import AssistantMessage

from .direct_rag_test_support import _consumer, _results_payload, _user_convo


async def test_prepared_queries_and_final_packet_follow_actual_search(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    monkeypatch.setenv("RAG_DIRECT_TOP_K", "2")
    monkeypatch.setenv("RAG_DIRECT_MAX_QUERIES", "1")
    searches = []
    events = []

    def search(_consumer, query, top_k):
        searches.append((query, top_k))
        return _results_payload()

    async def synth(_llm, convo, packet, **_kwargs):
        assert len(packet.chunks) == 1
        return convo + [AssistantMessage(content="answer", stop_reason="end_turn")]

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", search)
    monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", synth)
    convo = _user_convo("search the documents for calibration")
    with observe(lambda name, data: events.append((name, data))):
        outcome = await rag_pipeline.run_direct_rag_turn(
            _consumer(convo, [1]), SimpleNamespace(), convo
        )

    assert outcome == "handled"
    assert searches == [("search the documents for calibration", 6)]
    prepared = [data for name, data in events if name == "rag_prepared_queries"]
    assert len(prepared) == 1
    assert prepared[0]["queries"] == ["search the documents for calibration"]
    assert prepared[0]["requested_top_k"] == 2
    assert prepared[0]["candidate_top_k"] == 6
    final = [data for name, data in events if name == "rag_final_selection"]
    assert len(final) == 1
    assert final[0]["rows"][0]["doc_id"] == "doc-a"
    assert final[0]["rows"][0]["chunk_id"] == 1
    assert final[0]["limits"]["max_passages"] == 2


async def test_failing_observation_sink_does_not_change_direct_turn(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    monkeypatch.setenv("RAG_DIRECT_MAX_QUERIES", "1")
    searches = []

    def search(_consumer, query, top_k):
        searches.append((query, top_k))
        return _results_payload()

    async def synth(_llm, convo, _packet, **_kwargs):
        return convo + [AssistantMessage(content="answer", stop_reason="end_turn")]

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", search)
    monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", synth)
    convo = _user_convo("search the documents for calibration")

    def fail(_name, _data):
        raise RuntimeError("sink failed")

    with observe(fail) as state:
        outcome = await rag_pipeline.run_direct_rag_turn(
            _consumer(convo, [1]), SimpleNamespace(), convo
        )
    assert outcome == "handled"
    assert len(searches) == 1
    assert state.failed.is_set()


async def test_partial_multi_query_outcomes_stay_in_prepared_order(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    monkeypatch.setenv("RAG_DIRECT_MAX_QUERIES", "3")
    events = []
    searches = []
    question = "Explain what each paper is about? What overlaps between them?"

    def search(_consumer, query, _top_k):
        searches.append(query)
        if query == "Explain what each paper is about":
            raise RuntimeError("backend failed")
        if query == "What overlaps between them":
            raise TimeoutError("search timed out")
        return _results_payload()

    async def synth(_llm, convo, packet, **_kwargs):
        assert packet.chunks
        return convo + [AssistantMessage(content="answer", stop_reason="end_turn")]

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", search)
    monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", synth)
    convo = _user_convo(question)
    with observe(lambda name, data: events.append((name, data))):
        outcome = await rag_pipeline.run_direct_rag_turn(
            _consumer(convo, [1]), SimpleNamespace(), convo
        )

    assert outcome == "handled"
    assert len(searches) == 3
    prepared = [data for name, data in events if name == "rag_prepared_queries"]
    terminal = [data for name, data in events if name == "rag_search_outcomes"]
    assert len(prepared) == len(terminal) == 1
    assert terminal[0]["queries"] == [
        {"query": question, "status": "ok"},
        {"query": "Explain what each paper is about", "status": "error"},
        {"query": "What overlaps between them", "status": "timeout"},
    ]
    assert [row["query"] for row in terminal[0]["queries"]] == prepared[0]["queries"]


def test_inactive_search_outcome_observation_does_not_iterate_inputs():
    class Unreadable:
        def __iter__(self):
            raise AssertionError("inactive observation inspected outcomes")

    replay_observation.search_outcomes(Unreadable(), Unreadable())


def test_cancelled_child_search_is_reported_as_error_without_stringifying_it():
    events = []
    with observe(lambda name, data: events.append((name, data))):
        replay_observation.search_outcomes(
            ["synthetic query"], [asyncio.CancelledError("private detail")]
        )
    assert events[0][1]["queries"] == [{"query": "synthetic query", "status": "error"}]
    assert "private detail" not in repr(events)
