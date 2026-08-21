"""End-to-end success and authorization preflight contracts."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from django.test import override_settings

from apps.documents.services.chunk_search import (
    HybridGraphRetrievalDependencies,
    text_chunk_search,
)
from apps.documents.services.hybrid_graph_orchestration import (
    hybrid_graph_candidate_pool,
)
from apps.documents.tests.hybrid_graph_test_support import (
    KEYS,
    Policy,
    authorization,
    chunk,
    hybrid_settings,
    selected_snapshot,
    successful_branch,
)
from apps.documents.tests.test_chunk_search_graph_overlay import (
    _DOC_A,
    _DOC_B,
    _model,
    _snapshot,
)
from apps.knowledge_graph.retrieval.branch_contracts import GraphBranchCandidateV1
from apps.knowledge_graph.retrieval.materialization import MaterializedGraphChunkV1
from apps.knowledge_graph.retrieval.topology.contracts import HybridBranchKind


@override_settings(KG_OVERLAY_ENABLED=True, RAG_CACHE_ENABLED=False)
def test_one_baseline_concurrent_branches_shared_dependency_and_one_reranker(
    monkeypatch,
) -> None:
    from apps.documents.services import chunk_search

    baseline = (chunk(1), chunk(2))
    graph_three = chunk(3)
    graph_four = chunk(4, _DOC_B)
    snapshot = selected_snapshot(baseline=baseline)
    collect_calls: list[object] = []
    branch_barrier = threading.Barrier(2, timeout=2)

    class Runtime:
        def __init__(self) -> None:
            self.extended_baseline = None
            self.authorization = None

        def prepare_shared(self, *, authorization, settings, deadline):
            self.authorization = authorization
            return object()

        def run_direct(self, **_kwargs):
            branch_barrier.wait()
            return successful_branch(
                HybridBranchKind.DIRECT,
                (GraphBranchCandidateV1(KEYS[0], 1, 0.9),),
            )

        def prepare_extended(self, *, baseline, **_kwargs):
            self.extended_baseline = baseline
            branch_barrier.wait()
            return object()

        def run_extended(self, **_kwargs):
            return successful_branch(
                HybridBranchKind.EXTENDED,
                (
                    GraphBranchCandidateV1(KEYS[0], 1, 0.9),
                    GraphBranchCandidateV1(KEYS[1], 2, 0.8),
                ),
            )

    runtime = Runtime()
    auth = authorization(Policy())
    materialize_calls: list[tuple[object, ...]] = []

    def materialize(*, chunk_keys, authorization, outcome):
        materialize_calls.append((chunk_keys, authorization, outcome))
        return (
            MaterializedGraphChunkV1(KEYS[0], 3, _DOC_A, 2, graph_three),
            MaterializedGraphChunkV1(KEYS[1], 4, _DOC_B, 3, graph_four),
        )

    dependencies = HybridGraphRetrievalDependencies(
        runtime=runtime,
        settings=hybrid_settings(),
        materialize=materialize,
    )

    def collect(*args, **kwargs):
        collect_calls.append((args, kwargs))
        return snapshot

    rerank_calls: list[tuple[int, ...]] = []
    monkeypatch.setattr(chunk_search, "collect_hybrid_candidate_snapshot", collect)
    monkeypatch.setattr("aquillm.utils.get_embedding", lambda _query: (0.1, 0.2))
    monkeypatch.setattr(
        chunk_search,
        "rerank_chunks",
        lambda _model, _query, rows, _top_k: (
            rerank_calls.append(tuple(row.pk for row in rows)) or tuple(rows)
        ),
    )
    model, _filters = _model([])

    _vector, _trigram, results, diagnostics = text_chunk_search(
        model,
        "hybrid graph query",
        2,
        list(snapshot.documents),
        authorization_context=auth,
        hybrid_graph_dependencies=dependencies,
    )

    assert len(collect_calls) == 1
    assert runtime.extended_baseline is snapshot
    assert runtime.authorization is auth
    assert len(materialize_calls) == 1
    assert tuple(key.value for key in materialize_calls[0][0]) == KEYS[:2]
    assert materialize_calls[0][1] is auth
    assert rerank_calls == [(1, 2, 3, 4)]
    assert tuple(row.pk for row in results) == (1, 2, 3, 4)
    assert diagnostics["graph_status"] == "hit"
    assert diagnostics["graph_candidate_count"] == 2


@pytest.mark.parametrize("authorization_context", (None, object()))
@override_settings(KG_OVERLAY_ENABLED=True, RAG_CACHE_ENABLED=False)
def test_missing_or_malformed_authorization_never_starts_graph(
    monkeypatch,
    authorization_context,
) -> None:
    from apps.documents.services import chunk_search

    baseline = (chunk(1), chunk(2))
    snapshot = _snapshot(baseline=baseline)
    calls: list[str] = []

    class Runtime:
        def prepare_shared(self, **_kwargs):
            calls.append("shared")
            raise AssertionError("authorization must be rejected before scheduling")

        def run_direct(self, **_kwargs):
            raise AssertionError

        def prepare_extended(self, **_kwargs):
            raise AssertionError

        def run_extended(self, **_kwargs):
            raise AssertionError

    dependencies = HybridGraphRetrievalDependencies(
        runtime=Runtime(),
        settings=SimpleNamespace(
            graph_overall_timeout_ms=300,
            graph_direct_timeout_ms=125,
            graph_extended_timeout_ms=125,
            graph_direct_max_candidates=20,
            graph_extended_max_candidates=20,
            graph_fusion_rrf_k=60,
        ),
        materialize=lambda **_kwargs: calls.append("materialize"),
    )
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
        authorization_context=authorization_context,
        hybrid_graph_dependencies=dependencies,
    )

    assert calls == []
    assert rerank_inputs == [(1, 2)]
    assert tuple(row.pk for row in returned[2]) == (1, 2)
    assert returned[3]["graph_candidate_count"] == 0


@pytest.mark.parametrize(
    ("direct_enabled", "extended_enabled", "expected_calls"),
    (
        (True, False, ("shared", "direct")),
        (False, True, ("shared", "prepare_extended", "extended")),
    ),
)
def test_disabled_branch_is_never_invoked_or_fused(
    direct_enabled,
    extended_enabled,
    expected_calls,
) -> None:
    baseline = (chunk(1), chunk(2))
    graph_row = chunk(3)
    snapshot = selected_snapshot(baseline=baseline)
    calls: list[str] = []

    class Runtime:
        def prepare_shared(self, **_kwargs):
            calls.append("shared")
            return object()

        def run_direct(self, **_kwargs):
            calls.append("direct")
            if not direct_enabled:
                raise AssertionError("disabled direct branch was invoked")
            return successful_branch(
                HybridBranchKind.DIRECT,
                (GraphBranchCandidateV1(KEYS[0], 1, 0.9),),
            )

        def prepare_extended(self, **_kwargs):
            calls.append("prepare_extended")
            if not extended_enabled:
                raise AssertionError("disabled extended branch was invoked")
            return object()

        def run_extended(self, **_kwargs):
            calls.append("extended")
            return successful_branch(
                HybridBranchKind.EXTENDED,
                (GraphBranchCandidateV1(KEYS[0], 1, 0.9),),
            )

    settings = hybrid_settings()
    settings.graph_direct_enabled = direct_enabled
    settings.graph_extended_enabled = extended_enabled
    materialized_keys: list[tuple[object, ...]] = []

    def materialize(*, chunk_keys, **_kwargs):
        materialized_keys.append(chunk_keys)
        return (MaterializedGraphChunkV1(KEYS[0], 3, _DOC_A, 2, graph_row),)

    candidates, diagnostics = hybrid_graph_candidate_pool(
        snapshot,
        "query",
        authorization(Policy()),
        HybridGraphRetrievalDependencies(
            runtime=Runtime(),
            settings=settings,
            materialize=materialize,
        ),
    )

    assert tuple(calls) == expected_calls
    assert tuple(row.pk for row in candidates) == (1, 2, 3)
    assert tuple(key.value for key in materialized_keys[0]) == (KEYS[0],)
    assert diagnostics["graph_candidate_count"] == 1
