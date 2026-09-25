"""Materialization, fusion, and disabled-path fail-open contracts."""

from __future__ import annotations

import pytest
from django.test import override_settings

from apps.documents.services.chunk_search import (
    HybridGraphRetrievalDependencies,
    text_chunk_search,
)
from apps.documents.tests.hybrid_graph_test_support import (
    KEYS,
    Policy,
    authorization,
    chunk,
    hybrid_settings,
    run_hybrid_test_search,
    selected_snapshot,
    successful_branch,
)
from apps.documents.tests.test_chunk_search_graph_overlay import (
    _DOC_A,
    _DOC_B,
    _model,
)
from apps.knowledge_graph.retrieval.branch_contracts import GraphBranchCandidateV1
from apps.knowledge_graph.retrieval.materialization import MaterializedGraphChunkV1
from apps.knowledge_graph.retrieval.topology.contracts import HybridBranchKind


@pytest.mark.parametrize(
    "failure", ("materialization", "mismatched_candidate_object", "fusion")
)
@override_settings(KG_OVERLAY_ENABLED=True, RAG_CACHE_ENABLED=False)
def test_materialization_or_fusion_failure_discards_entire_graph_pool_once(
    monkeypatch,
    failure,
) -> None:
    baseline = (chunk(1), chunk(2))
    snapshot = selected_snapshot(baseline=baseline)

    class Runtime:
        def prepare_shared(self, **_kwargs):
            return object()

        def run_direct(self, **_kwargs):
            return successful_branch(
                HybridBranchKind.DIRECT,
                (GraphBranchCandidateV1(KEYS[0], 1, 0.9),),
            )

        def prepare_extended(self, **_kwargs):
            return object()

        def run_extended(self, **_kwargs):
            return successful_branch(
                HybridBranchKind.EXTENDED,
                (GraphBranchCandidateV1(KEYS[1], 1, 0.8),),
            )

    def materialize(**_kwargs):
        if failure == "materialization":
            return ()
        if failure == "mismatched_candidate_object":
            return (
                MaterializedGraphChunkV1(KEYS[0], 3, _DOC_A, 2, chunk(99, _DOC_B)),
                MaterializedGraphChunkV1(KEYS[1], 4, _DOC_B, 3, chunk(4, _DOC_B)),
            )
        return (
            MaterializedGraphChunkV1(KEYS[0], 3, _DOC_A, 2, chunk(3)),
            MaterializedGraphChunkV1(KEYS[1], 3, _DOC_B, 3, chunk(3, _DOC_B)),
        )

    returned, rerank_inputs = run_hybrid_test_search(
        monkeypatch,
        snapshot=snapshot,
        authorization_context=authorization(Policy()),
        runtime=Runtime(),
        materialize=materialize,
    )

    assert rerank_inputs == [(1, 2)]
    assert tuple(row.pk for row in returned[2]) == (1, 2)
    assert returned[3]["graph_status"] == "error"
    assert returned[3]["graph_candidate_count"] == 0


@override_settings(KG_OVERLAY_ENABLED=True, RAG_CACHE_ENABLED=False)
def test_disabled_hybrid_flags_never_start_runtime(monkeypatch) -> None:
    from apps.documents.services import chunk_search

    baseline = (chunk(1), chunk(2))
    snapshot = selected_snapshot(baseline=baseline)
    calls: list[str] = []

    class Runtime:
        def prepare_shared(self, **_kwargs):
            calls.append("shared")
            raise AssertionError

        def run_direct(self, **_kwargs):
            raise AssertionError

        def prepare_extended(self, **_kwargs):
            raise AssertionError

        def run_extended(self, **_kwargs):
            raise AssertionError

    settings = hybrid_settings()
    settings.memgraph_traversal_enabled = False
    rerank_inputs: list[tuple[int, ...]] = []
    monkeypatch.setattr(
        chunk_search,
        "collect_hybrid_candidate_snapshot",
        lambda *_args, **_kwargs: snapshot,
    )
    monkeypatch.setattr("aquillm.utils.get_embedding", lambda _query: (0.1, 0.2))
    monkeypatch.setattr(
        chunk_search,
        "rerank_chunks",
        lambda _model, _query, rows, _top_k: (
            rerank_inputs.append(tuple(row.pk for row in rows)) or tuple(rows)
        ),
    )
    model, _filters = _model([])
    returned = text_chunk_search(
        model,
        "query",
        1,
        list(snapshot.documents),
        authorization_context=authorization(Policy()),
        hybrid_graph_dependencies=HybridGraphRetrievalDependencies(
            runtime=Runtime(),
            settings=settings,
            materialize=lambda **_kwargs: calls.append("materialize"),
        ),
    )

    assert calls == []
    assert rerank_inputs == [(1, 2)]
    assert tuple(row.pk for row in returned[2]) == (1, 2)


@override_settings(KG_OVERLAY_ENABLED=False, RAG_CACHE_ENABLED=False)
def test_global_overlay_off_does_not_apply_hybrid_authorization(monkeypatch) -> None:
    from apps.documents.services import chunk_search

    baseline = (chunk(1), chunk(2))
    snapshot = selected_snapshot(baseline=baseline)
    calls: list[str] = []
    forwarded_authorization: list[object] = []

    class Runtime:
        def prepare_shared(self, **_kwargs):
            calls.append("shared")
            raise AssertionError

        def run_direct(self, **_kwargs):
            raise AssertionError

        def prepare_extended(self, **_kwargs):
            raise AssertionError

        def run_extended(self, **_kwargs):
            raise AssertionError

    def collect(*_args, **kwargs):
        forwarded_authorization.append(kwargs["authorization_context"])
        return snapshot

    monkeypatch.setattr(chunk_search, "collect_hybrid_candidate_snapshot", collect)
    monkeypatch.setattr("aquillm.utils.get_embedding", lambda _query: (0.1, 0.2))
    monkeypatch.setattr(
        chunk_search,
        "rerank_chunks",
        lambda _model, _query, rows, _top_k: tuple(rows),
    )
    model, _filters = _model([])
    returned = text_chunk_search(
        model,
        "query",
        1,
        list(snapshot.documents),
        authorization_context=object(),
        hybrid_graph_dependencies=HybridGraphRetrievalDependencies(
            runtime=Runtime(),
            settings=hybrid_settings(),
            materialize=lambda **_kwargs: calls.append("materialize"),
        ),
    )

    assert forwarded_authorization == [None]
    assert calls == []
    assert tuple(row.pk for row in returned[2]) == (1, 2)
