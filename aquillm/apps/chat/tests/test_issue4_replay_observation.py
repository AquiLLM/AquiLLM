"""Opt-in observations of the actual direct-RAG handoff."""

import asyncio
from types import SimpleNamespace
import pytest
from .persisted_direct_test_support import close_direct_worker as close_direct_worker
from .direct_rag_test_support import _synthesize, _admit_query

pytestmark = pytest.mark.usefixtures("close_direct_worker")

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
        _admit_query(query)
        searches.append((query, top_k))
        return consumer.case.rows(_results_payload())

    async def synth(_llm, convo, packet, **_kwargs):
        assert len(packet.chunks) == 1
        return await _synthesize(consumer, convo, packet, "answer " + packet.citation_tokens[0])

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", search)
    monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", synth)
    convo = _user_convo("search the documents for calibration")
    consumer = await _consumer(convo, [1])
    with observe(lambda name, data: events.append((name, data))):
        outcome = await rag_pipeline.run_direct_rag_turn(
            consumer, SimpleNamespace(), convo
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
    assert final[0]["rows"][0]["doc_id"] == str(consumer.case.documents["doc-a"].id)
    assert final[0]["rows"][0]["chunk_id"] == consumer.case.chunks[("doc-a", 1)].pk
    assert final[0]["limits"]["max_passages"] == 2


async def test_failing_observation_sink_does_not_change_direct_turn(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    monkeypatch.setenv("RAG_DIRECT_MAX_QUERIES", "1")
    searches = []

    def search(_consumer, query, top_k):
        _admit_query(query)
        searches.append((query, top_k))
        return consumer.case.rows(_results_payload())

    async def synth(_llm, convo, _packet, **_kwargs):
        return await _synthesize(consumer, convo, _packet, "answer " + _packet.citation_tokens[0])

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", search)
    monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", synth)
    convo = _user_convo("search the documents for calibration")
    consumer = await _consumer(convo, [1])

    def fail(_name, _data):
        raise RuntimeError("sink failed")

    with observe(fail) as state:
        outcome = await rag_pipeline.run_direct_rag_turn(
            consumer, SimpleNamespace(), convo
        )
    assert outcome == "handled"
    assert len(searches) == 1
    assert state.failed.is_set()


async def test_explicit_parallel_search_outcomes_stay_in_prepared_order(monkeypatch):
    """Service-composition probe; does not claim automatic clause fanout/publication."""
    from apps.chat.services.rag_pipeline_retrieval import run_parallel_searches
    from apps.chat.services.rag_turn import preservation_turn
    from lib.llm.turn_context import current_turn
    monkeypatch.setenv("RAG_DIRECT_MAX_QUERIES", "3")
    queries = ["Original constraint-preserving question", "Explicit failure query", "Explicit timeout query"]
    events, searches = [], []
    convo = _user_convo(queries[0])
    consumer = await _consumer(convo, [1])
    def search(consumer, query, _top_k):
        _admit_query(query)
        searches.append(query)
        if query == queries[1]:
            raise RuntimeError("backend failed")
        if query == queries[2]:
            raise TimeoutError("search timed out")
        return consumer.case.rows(_results_payload())
    async with preservation_turn(consumer, max_func_calls=3):
        budget = current_turn().budget
        with observe(lambda name, data: events.append((name, data))):
            replay_observation.prepared_queries(queries, 3, 9)
            outcomes = await run_parallel_searches(consumer, queries, 9, vector_runner=search)
            replay_observation.search_outcomes(queries, outcomes)
        assert budget.actions_used == 3 and budget.planner_calls == 0
    assert sorted(searches) == sorted(queries)
    assert isinstance(outcomes[0], dict)
    assert type(outcomes[1]) is RuntimeError and type(outcomes[2]) is TimeoutError
    prepared = [data for name, data in events if name == "rag_prepared_queries"]
    terminal = [data for name, data in events if name == "rag_search_outcomes"]
    assert len(prepared) == len(terminal) == 1
    assert terminal[0]["queries"] == [
        {"query": queries[0], "status": "ok"}, {"query": queries[1], "status": "error"},
        {"query": queries[2], "status": "timeout"}]
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
