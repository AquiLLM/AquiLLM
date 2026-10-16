"""The replay trace must describe the executed chunk search, once."""

from types import SimpleNamespace
from unittest.mock import patch

from django.test import override_settings

from apps.documents.services import chunk_search
from apps.documents.tests.test_chunk_search_graph_overlay import (
    _DOC_A,
    _model,
    _snapshot,
)
from lib.evidence_observation import observe
from lib.replay_observation import retrieval_stages


@override_settings(KG_OVERLAY_ENABLED=False, RAG_CACHE_ENABLED=False)
def test_retrieval_stages_use_actual_candidate_and_rerank_rows(monkeypatch):
    first = SimpleNamespace(pk=11, doc_id=_DOC_A, content="first passage")
    second = SimpleNamespace(pk=12, doc_id=_DOC_A, content="second passage")
    snapshot = _snapshot(baseline=(first, second))
    searches = []
    events = []
    model, _filters = _model([])

    def collect(*args, **_kwargs):
        searches.append(args[1])
        return snapshot

    monkeypatch.setattr(chunk_search, "collect_hybrid_candidate_snapshot", collect)
    monkeypatch.setattr(
        chunk_search, "_fallback_rerank", lambda _model, _rows, _k: [second, first]
    )
    with (
        patch("aquillm.utils.get_embedding", return_value=[0.1]),
        observe(lambda name, data: events.append((name, data))),
    ):
        result = chunk_search.text_chunk_search(
            model, "test query", 2, list(snapshot.documents)
        )

    assert searches == ["test query"]
    assert result[2] == [second, first]
    traces = [data for name, data in events if name == "retrieval_stages"]
    assert len(traces) == 1
    trace = traces[0]
    assert trace["query"] == "test query"
    assert trace["requested_limits"]["top_k"] == 2
    assert [row["chunk_id"] for row in trace["baseline_branches"]["vector"]] == [11, 12]
    assert [row["chunk_id"] for row in trace["materialized_union"]] == [11, 12]
    assert [row["chunk_id"] for row in trace["post_rerank"]] == [12, 11]
    assert trace["graph"]["ready"] is None
    assert trace["graph"]["candidates"] == []


@override_settings(KG_OVERLAY_ENABLED=False, RAG_CACHE_ENABLED=False)
def test_inactive_observer_never_reads_candidate_content(monkeypatch):
    class Protected:
        pk = 13
        doc_id = _DOC_A

        @property
        def content(self):
            raise AssertionError("inactive observation inspected content")

    row = Protected()
    snapshot = _snapshot(baseline=(row,))
    model, _filters = _model([])
    monkeypatch.setattr(
        chunk_search, "collect_hybrid_candidate_snapshot", lambda *_a, **_k: snapshot
    )
    monkeypatch.setattr(
        chunk_search, "_fallback_rerank", lambda _m, rows, _k: list(rows)
    )
    with patch("aquillm.utils.get_embedding", return_value=[0.1]):
        result = chunk_search.text_chunk_search(
            model, "test query", 2, list(snapshot.documents)
        )
    assert result[2] == [row]


@override_settings(KG_OVERLAY_ENABLED=False, RAG_CACHE_ENABLED=False)
def test_failed_sink_and_unreadable_observation_leave_search_result_intact(monkeypatch):
    row = SimpleNamespace(pk=14, doc_id=_DOC_A, content="selected passage")
    snapshot = _snapshot(baseline=(row,))
    model, _filters = _model([])
    searches = []

    def collect(*_args, **_kwargs):
        searches.append(1)
        return snapshot

    monkeypatch.setattr(chunk_search, "collect_hybrid_candidate_snapshot", collect)
    monkeypatch.setattr(
        chunk_search, "_fallback_rerank", lambda _m, rows, _k: list(rows)
    )

    def fail(_name, _data):
        raise RuntimeError("sink failed")

    with (
        patch("aquillm.utils.get_embedding", return_value=[0.1]),
        observe(fail) as state,
    ):
        result = chunk_search.text_chunk_search(
            model, "test query", 2, list(snapshot.documents)
        )
    assert result[2] == [row]
    assert searches == [1]
    assert state.failed.is_set()

    class Unreadable:
        pk = 15
        doc_id = _DOC_A

        @property
        def content(self):
            raise RuntimeError("observation data unavailable")

    unreadable = Unreadable()
    monkeypatch.setattr(
        chunk_search,
        "collect_hybrid_candidate_snapshot",
        lambda *_a, **_k: _snapshot(baseline=(unreadable,)),
    )
    with (
        patch("aquillm.utils.get_embedding", return_value=[0.1]),
        observe(lambda *_a: None) as state,
    ):
        result = chunk_search.text_chunk_search(
            model, "test query", 2, list(snapshot.documents)
        )
    assert result[2] == [unreadable]
    assert state.failed.is_set()


def test_duplicate_only_graph_observation_identifies_fused_provenance():
    row = SimpleNamespace(pk=16, doc_id=_DOC_A, content="baseline passage")
    snapshot = _snapshot(baseline=(row,))
    ranking = SimpleNamespace(combined_candidates=(row,), graph_candidates=())
    events = []
    with observe(lambda name, data: events.append((name, data))):
        retrieval_stages(
            "test query",
            1,
            snapshot,
            ranking,
            (row,),
            {"graph_status": "miss", "graph_direct_status": "succeeded_duplicates"},
            overlay_enabled=True,
            hybrid_pool=(row,),
            graph_seed_attempted=True,
        )
    graph = [data["graph"] for name, data in events if name == "retrieval_stages"][0]
    assert graph["status"] == "miss"
    assert graph["candidate_provenance"] == "novel_fused"
    assert graph["candidates"] == []
    assert graph["branch_statuses"] == {"direct": "succeeded_duplicates"}


def test_non_readiness_graph_failure_does_not_claim_zero_readiness_failures():
    row = SimpleNamespace(pk=17, doc_id=_DOC_A, content="baseline passage")
    snapshot = _snapshot(baseline=(row,))
    ranking = SimpleNamespace(combined_candidates=(row,), graph_candidates=())
    events = []
    with observe(lambda name, data: events.append((name, data))):
        retrieval_stages(
            "test query",
            1,
            snapshot,
            ranking,
            (row,),
            {"graph_status": "error", "graph_direct_reason": "backend_unavailable"},
            overlay_enabled=True,
            hybrid_pool=(row,),
            graph_seed_attempted=True,
        )
    trace = [data for name, data in events if name == "retrieval_stages"][0]
    assert trace["graph"]["reasons"] == ["backend_unavailable"]
    assert trace["readiness_failure_count"] is None
