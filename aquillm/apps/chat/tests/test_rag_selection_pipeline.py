"""Focused direct-RAG selection integration contracts."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from apps.chat.refs import CollectionsRef
from apps.chat.services import rag_pipeline, rag_selection_coordinator
from apps.chat.services.rag_config import EvidenceSelectionConfig
from apps.chat.services.rag_retrieval import FusedRetrievalPool
from apps.chat.services.rag_selection_scoring import PreparedSelection
from apps.chat.services.rag_selection_types import (
    EvidenceSelection,
    SelectionCandidate,
    SelectionProfile,
)
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage


def _row(pk):
    return {
        "rank": pk,
        "chunk_id": pk,
        "doc_id": "00000000-0000-0000-0000-000000000001",
        "chunk": pk,
        "title": "Paper",
        "text": f"Passage {pk}",
        "citation": f"[doc:00000000-0000-0000-0000-000000000001 chunk:{pk}]",
    }


def test_retry_profile_question_ignores_prior_title():
    convo = Conversation(
        system="sys",
        messages=[
            UserMessage(content="Compare evidence across papers"),
            AssistantMessage(
                content="",
                stop_reason="tool_use",
                tool_call_id="c1",
                tool_call_name="vector_search",
                tool_call_input={
                    "search_string": "A title: compare evidence across papers",
                    "top_k": 10,
                },
            ),
            ToolMessage(
                tool_name="vector_search",
                for_whom="assistant",
                content="",
                arguments={},
                result_dict={"retrieved_documents": ["A title"]},
            ),
            AssistantMessage(content="Earlier", stop_reason="end_turn"),
            UserMessage(content="retry"),
        ],
    )
    assert (
        rag_pipeline._selection_question(convo, "retry")
        == "Compare evidence across papers"
    )


@pytest.mark.parametrize("shadow_scoring", [False, True])
def test_shadow_new_scoring_obeys_explicit_flag(monkeypatch, shadow_scoring):
    observed = {}
    fake_collection = SimpleNamespace(
        objects=SimpleNamespace(filter=lambda **_kwargs: object()),
        get_user_accessible_documents=lambda *_args: [object()],
    )
    monkeypatch.setattr(rag_selection_coordinator, "Collection", fake_collection)
    monkeypatch.setattr(
        rag_selection_coordinator,
        "resolve_document_retrieval_authorization",
        lambda *_args: object(),
    )

    def fake_prepare(**kwargs):
        observed["allow_new_scores"] = kwargs["allow_new_scores"]
        return PreparedSelection((), "rank_fallback", 0, 0, 0.0, "scorer_unavailable")

    monkeypatch.setattr(
        rag_selection_coordinator,
        "prepare_selection_candidates",
        fake_prepare,
    )
    consumer = SimpleNamespace(user=object(), col_ref=CollectionsRef([1]))
    rag_selection_coordinator.prepare_selection_turn(
        consumer,
        FusedRetrievalPool((), (), ()),
        "query",
        "Compare papers",
        EvidenceSelectionConfig("shadow", 3000, shadow_scoring),
        10,
    )
    assert observed["allow_new_scores"] is shadow_scoring


@pytest.mark.parametrize("drop_selected", [False, True])
async def test_adaptive_selects_full_union_once_and_persists_only_selected(
    monkeypatch, drop_selected
):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    monkeypatch.setenv("RAG_EVIDENCE_SELECTION_MODE", "adaptive")
    monkeypatch.setenv("RAG_DIRECT_TOP_K", "1")
    monkeypatch.setenv("RAG_DIRECT_MAX_QUERIES", "2")
    rows = [_row(i) for i in range(1, 4)]
    seen = {"synthesis": 0}

    def search(_consumer, query, _top_k):
        return {"result": rows[:2] if ";" in query else rows[1:]}

    def prepare(_consumer, pool, primary_query, question, config, _top_k):
        seen["pool"] = len(pool.rows)
        seen["query"] = primary_query
        seen["question"] = question
        candidate = SelectionCandidate(
            3,
            rows[2]["doc_id"],
            3,
            rows[2]["text"],
            0.9,
            3,
            "fingerprint",
            rows[2],
        )
        return SimpleNamespace(
            selection=EvidenceSelection(
                (candidate,),
                3,
                SelectionProfile("breadth", 0.8, 0.15, "v1"),
                "rank_fallback",
            ),
            authorization=object(),
            candidate_count=3,
            selection_duration_ms=1.0,
            prepared=SimpleNamespace(
                scoring_duration_ms=0.0,
                reused_pairs=0,
                new_pairs=0,
                fallback_reason="scorer_unavailable",
                candidates=(candidate,),
            ),
        )

    def revalidate(selection, _authorization):
        if drop_selected:
            return EvidenceSelection((), 0, selection.profile, selection.score_status)
        return selection

    async def synth(_llm, convo, packet, **_kwargs):
        seen["synthesis"] += 1
        seen["packet"] = packet
        seen["persisted"] = convo[-1].result_dict
        return convo + [AssistantMessage(content="Answer", stop_reason="end_turn")]

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", search)
    monkeypatch.setattr(rag_pipeline, "_prepare_selection_sync", prepare)
    monkeypatch.setattr(rag_pipeline, "_revalidate_selection_sync", revalidate)
    monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", synth)
    convo = Conversation(
        system="sys",
        messages=[UserMessage(content="compare papers; and summarize trials")],
    )
    consumer = SimpleNamespace(user=object(), col_ref=CollectionsRef([1]), convo=convo)
    assert (
        await rag_pipeline.run_direct_rag_turn(consumer, AsyncMock(), convo)
        == "handled"
    )
    assert seen["pool"] == 3
    assert seen["query"] == "compare papers; and summarize trials"
    assert seen["question"] == "compare papers; and summarize trials"
    assert seen["synthesis"] == 1
    expected = [] if drop_selected else [3]
    assert [row["chunk_id"] for row in seen["packet"].chunks] == expected
    assert [row["chunk_id"] for row in seen["persisted"]["result"]] == expected
    assert seen["persisted"]["retrieved_count"] == len(expected)
    assert seen["persisted"]["retrieval_status"] == (
        "no_results" if drop_selected else "results_found"
    )
    assert "_retrieval_scores" not in seen["persisted"]


async def test_shadow_failure_preserves_legacy_packet(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    monkeypatch.setenv("RAG_EVIDENCE_SELECTION_MODE", "shadow")
    monkeypatch.setenv("RAG_DIRECT_MAX_QUERIES", "1")
    row = _row(1)
    observed = {}

    def prepare(*_args):
        raise RuntimeError("comparison failed")

    async def synth(_llm, convo, packet, **_kwargs):
        observed["packet"] = packet
        observed["persisted"] = convo[-1].result_dict
        return convo + [AssistantMessage(content="Answer", stop_reason="end_turn")]

    monkeypatch.setattr(
        rag_pipeline, "_run_vector_search", lambda *_: {"result": [row]}
    )
    monkeypatch.setattr(rag_pipeline, "_prepare_selection_sync", prepare)
    monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", synth)
    convo = Conversation(
        system="sys", messages=[UserMessage(content="what is this paper")]
    )
    consumer = SimpleNamespace(user=object(), col_ref=CollectionsRef([1]), convo=convo)
    assert (
        await rag_pipeline.run_direct_rag_turn(consumer, AsyncMock(), convo)
        == "handled"
    )
    assert observed["packet"].chunks == [row]
    assert observed["persisted"]["result"] == [row]


async def test_adaptive_all_queries_failed_leaves_conversation_untouched(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    monkeypatch.setenv("RAG_EVIDENCE_SELECTION_MODE", "adaptive")
    monkeypatch.setenv("RAG_DIRECT_MAX_QUERIES", "2")

    def search(*_args):
        raise RuntimeError("backend unavailable")

    synthesis = AsyncMock()
    monkeypatch.setattr(rag_pipeline, "_run_vector_search", search)
    monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", synthesis)
    convo = Conversation(system="sys", messages=[UserMessage(content="compare papers")])
    consumer = SimpleNamespace(user=object(), col_ref=CollectionsRef([1]), convo=convo)
    assert (
        await rag_pipeline.run_direct_rag_turn(consumer, AsyncMock(), convo)
        == "skipped"
    )
    assert consumer.convo is convo
    synthesis.assert_not_called()


async def test_adaptive_partial_query_failure_selects_surviving_union(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    monkeypatch.setenv("RAG_EVIDENCE_SELECTION_MODE", "adaptive")
    monkeypatch.setenv("RAG_DIRECT_MAX_QUERIES", "2")
    observed = {"searches": [], "synthesis": 0}
    rows = [_row(2), _row(3)]
    rows[0]["_retrieval_scores"] = [0.9]

    def search(_consumer, query, _top_k):
        observed["searches"].append(query)
        if ";" in query:
            raise RuntimeError("one subquery failed")
        return {"result": rows}

    def prepare(_consumer, pool, _query, _question, _config, _top_k):
        observed["pool_ids"] = [row["chunk_id"] for row in pool.rows]
        row = pool.rows[0]
        candidate = SelectionCandidate(
            2, row["doc_id"], 2, row["text"], 0.9, 1, "fp", row
        )
        return SimpleNamespace(
            selection=EvidenceSelection(
                (candidate,),
                3,
                SelectionProfile("breadth", 0.8, 0.15, "v1"),
                "rank_fallback",
            ),
            authorization=object(),
            candidate_count=len(pool.rows),
            selection_duration_ms=1.0,
            prepared=PreparedSelection((candidate,), "rank_fallback", 0, 0, 0.0, None),
        )

    async def synth(_llm, convo, packet, **_kwargs):
        observed["synthesis"] += 1
        observed["packet"] = packet
        observed["persisted"] = convo[-1].result_dict
        return convo + [AssistantMessage(content="Answer", stop_reason="end_turn")]

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", search)
    monkeypatch.setattr(rag_pipeline, "_prepare_selection_sync", prepare)
    monkeypatch.setattr(rag_pipeline, "_revalidate_selection_sync", lambda s, _a: s)
    monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", synth)
    convo = Conversation(
        system="sys",
        messages=[UserMessage(content="compare papers; and summarize trials")],
    )
    consumer = SimpleNamespace(user=object(), col_ref=CollectionsRef([1]), convo=convo)
    assert (
        await rag_pipeline.run_direct_rag_turn(consumer, AsyncMock(), convo)
        == "handled"
    )
    assert len(observed["searches"]) == 2
    assert observed["pool_ids"] == [2, 3]
    assert observed["synthesis"] == 1
    assert [row["chunk_id"] for row in observed["packet"].chunks] == [2]
    assert observed["persisted"]["retrieved_count"] == 1
    assert observed["persisted"]["result"] == observed["packet"].chunks
    assert "_retrieval_scores" not in observed["persisted"]["result"][0]
