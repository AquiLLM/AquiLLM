"""Complete-reranker authority and dependency-seam privacy contracts."""

from __future__ import annotations

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
    selected_snapshot,
    successful_branch,
)
from apps.documents.tests.test_chunk_search_graph_overlay import _DOC_A, _model
from apps.knowledge_graph.retrieval.branch_contracts import GraphBranchCandidateV1
from apps.knowledge_graph.retrieval.materialization import MaterializedGraphChunkV1
from apps.knowledge_graph.retrieval.topology.contracts import HybridBranchKind


@override_settings(KG_OVERLAY_ENABLED=True, RAG_CACHE_ENABLED=False)
def test_successful_hybrid_pool_always_uses_complete_reranker_once(
    monkeypatch,
) -> None:
    from apps.documents.services import chunk_search

    baseline = (chunk(1), chunk(2))
    graph_row = chunk(3)
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
                (GraphBranchCandidateV1(KEYS[0], 1, 0.9),),
            )

    complete_reranks: list[tuple[int, ...]] = []
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
            complete_reranks.append(tuple(row.pk for row in rows)) or tuple(rows)
        ),
    )
    monkeypatch.setattr(
        chunk_search,
        "_fallback_rerank",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("successful hybrid pool used fallback reranker")
        ),
    )
    model, _filters = _model([])

    returned = text_chunk_search(
        model,
        "query",
        5,
        list(snapshot.documents),
        authorization_context=authorization(Policy()),
        hybrid_graph_dependencies=HybridGraphRetrievalDependencies(
            runtime=Runtime(),
            settings=hybrid_settings(),
            materialize=lambda **_kwargs: (
                MaterializedGraphChunkV1(KEYS[0], 3, _DOC_A, 2, graph_row),
            ),
        ),
    )

    assert complete_reranks == [(1, 2, 3)]
    assert tuple(row.pk for row in returned[2]) == (1, 2, 3)


def test_hybrid_dependencies_repr_never_exposes_adapter_objects() -> None:
    canary = "HYBRID_DEPENDENCY_SECRET_CANARY"

    class Runtime:
        def __repr__(self):
            return canary

        prepare_shared = run_direct = prepare_extended = run_extended = lambda self: (
            None
        )

    class Secret:
        def __repr__(self):
            return canary

        def __call__(self, **_kwargs):
            return ()

    dependencies = HybridGraphRetrievalDependencies(
        runtime=Runtime(),
        settings=Secret(),
        materialize=Secret(),
    )

    assert canary not in repr(dependencies)
