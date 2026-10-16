"""The replay trace must describe the executed chunk search, once."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from django.test import override_settings

from apps.chat.evals.replay_observations import summarize
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


def _native_graph_observation(diagnostics, *, enabled=True, sink=None):
    row = SimpleNamespace(pk=18, doc_id=_DOC_A, content="baseline passage")
    events = []
    with observe(sink or (lambda name, data: events.append((name, data)))) as state:
        retrieval_stages(
            "test query",
            1,
            _snapshot(baseline=(row,)),
            SimpleNamespace(combined_candidates=None, graph_candidates=()),
            (row,),
            diagnostics,
            overlay_enabled=enabled,
            hybrid_pool=(),
            graph_seed_attempted=enabled,
            acquisition_schedule_observed="native_closure",
        )
    return events, state


def _unavailable_coverage(reasons):
    return {
        "selected_documents": 530,
        "ready_documents": 0,
        "status": "unavailable",
        "reasons": reasons,
        "exhaustive": False,
    }


def test_empty_native_handoff_preserves_mixed_coverage_in_compact_json():
    events, state = _native_graph_observation(
        {
            "graph_status": "error",
            "graph_failure_reason": "frontier_budget",
            "graph_coverage_v2": _unavailable_coverage(
                ("frontier_budget", "not_ready")
            ),
        }
    )
    assert not state.failed.is_set()
    graph = events[0][1]["graph"]
    assert graph["finalized_candidates"] == []
    assert graph["reasons"] == ["frontier_budget", "not_ready"]
    assert graph["failure_reason"] == "frontier_budget"
    persisted = json.loads(json.dumps(summarize(events, [], [])))
    compact = persisted["stages"][0]["graph"]
    assert compact["finalized_count"] == 0
    assert compact["reasons"] == ["frontier_budget", "not_ready"]
    assert compact["failure_reason"] == "frontier_budget"


@pytest.mark.parametrize(
    "diagnostics,enabled,reasons,failure_reason",
    [
        (
            {"graph_coverage_v2": _unavailable_coverage(("not_ready",))},
            True,
            ["not_ready"],
            None,
        ),
        ({}, True, None, None),
        ({}, False, None, None),
        (
            {"graph_direct_reason": "readiness_mismatch"},
            True,
            ["readiness_mismatch"],
            None,
        ),
        (
            {
                "graph_direct_reason": "readiness_mismatch",
                "graph_extended_reason": "readiness_mismatch",
                "graph_failure_reason": "frontier_budget",
                "graph_coverage_v2": _unavailable_coverage(
                    ("frontier_budget", "not_ready")
                ),
            },
            True,
            ["readiness_mismatch", "frontier_budget", "not_ready"],
            "frontier_budget",
        ),
        (
            {
                "graph_failure_reason": "deadline",
                "graph_coverage_v2": _unavailable_coverage(
                    (
                        "deadline",
                        "evidence_budget",
                        "frontier_budget",
                        "incompatible_generation",
                        "not_ready",
                    )
                ),
            },
            True,
            [
                "deadline",
                "evidence_budget",
                "frontier_budget",
                "incompatible_generation",
                "not_ready",
            ],
            "deadline",
        ),
        ({"graph_failure_reason": "not_ready"}, True, None, None),
    ],
)
def test_graph_reason_order_and_legacy_disabled_observations(
    diagnostics, enabled, reasons, failure_reason
):
    events, state = _native_graph_observation(diagnostics, enabled=enabled)
    assert not state.failed.is_set()
    graph = events[0][1]["graph"]
    assert graph["reasons"] == reasons
    assert graph["failure_reason"] == failure_reason
    assert graph["finalized_candidates"] == []
    compact = json.loads(json.dumps(summarize(events, [], [])))["stages"][0]["graph"]
    assert compact["reasons"] == reasons
    assert compact["failure_reason"] == failure_reason


class _UntrustedCoverage:
    def __str__(self):
        raise AssertionError("untrusted coverage stringified")

    def __deepcopy__(self, _memo):
        raise AssertionError("untrusted coverage copied")


class _CoverageDict(dict):
    def get(self, *_args):
        raise AssertionError("non-exact coverage inspected")


@pytest.mark.parametrize(
    "coverage",
    [
        None,
        _UntrustedCoverage(),
        _CoverageDict(_unavailable_coverage(("not_ready",))),
        {"reasons": ("not_ready",)},
        _unavailable_coverage(["not_ready"]),
        _unavailable_coverage("not_ready"),
        _unavailable_coverage(("unknown private reason",)),
        _unavailable_coverage((_UntrustedCoverage(),)),
        _unavailable_coverage(("not_ready",) * 6),
        {**_unavailable_coverage(("not_ready",)), "selected_documents": True},
        {**_unavailable_coverage(("not_ready",)), "ready_documents": 531},
        {**_unavailable_coverage(("not_ready",)), "status": _UntrustedCoverage()},
        {**_unavailable_coverage(("not_ready",)), "exhaustive": True},
        {**_unavailable_coverage(("not_ready",)), "extra": _UntrustedCoverage()},
    ],
)
def test_malformed_coverage_is_ignored_without_copying_or_stringifying(coverage):
    events, state = _native_graph_observation(
        {"graph_failure_reason": "deadline", "graph_coverage_v2": coverage}
    )
    assert not state.failed.is_set()
    graph = events[0][1]["graph"]
    assert graph["reasons"] == ["deadline"]
    assert graph["failure_reason"] == "deadline"
    compact = json.loads(json.dumps(summarize(events, [], [])))["stages"][0]["graph"]
    assert compact["reasons"] == ["deadline"]
    assert compact["failure_reason"] == "deadline"


@pytest.mark.parametrize("failure_reason", [None, "not_ready", 1, _UntrustedCoverage()])
def test_compact_failure_reason_rejects_unsupported_or_nonexact_values(failure_reason):
    compact = summarize(
        [("retrieval_stages", {"graph": {"failure_reason": failure_reason}})], [], []
    )["stages"][0]["graph"]
    assert compact["failure_reason"] is None
    json.dumps(compact)


def test_coverage_observation_sink_failure_remains_passive():
    def fail(_name, _data):
        raise RuntimeError("sink failed")

    events, state = _native_graph_observation(
        {"graph_coverage_v2": _unavailable_coverage(("not_ready",))}, sink=fail
    )
    assert events == []
    assert state.failed.is_set()
