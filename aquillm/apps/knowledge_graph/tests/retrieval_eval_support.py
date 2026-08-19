"""Shared deterministic fakes for Task20 retrieval-evaluation tests."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from uuid import UUID

from apps.knowledge_graph.retrieval.ppr import (
    PPRAlgorithmConfig,
    graph_algorithm_signature,
)
from apps.knowledge_graph.retrieval.types import (
    GraphExpansionConfig,
    GraphExpansionDiagnostics,
    GraphExpansionResult,
    GraphExpansionSeed,
)

DOC_A = UUID("11111111-1111-4111-8111-111111111111")
DOC_B = UUID("22222222-2222-4222-8222-222222222222")
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64


def shipping_config() -> PPRAlgorithmConfig:
    return PPRAlgorithmConfig(
        canonical_resolver_version="canonical-v1",
        max_scope_documents=4,
        max_scope_collections=4,
        ppr_iterations=8,
    )


def acquisition_config() -> GraphExpansionConfig:
    shipping = shipping_config()
    return GraphExpansionConfig(
        rrf_k=shipping.rrf_k,
        max_seeds=shipping.max_seeds,
        max_scope_documents=shipping.max_scope_documents,
        max_scope_collections=shipping.max_scope_collections,
        max_candidates=shipping.max_candidates,
        algorithm_signature=graph_algorithm_signature(shipping),
    )


def trace(
    *,
    algorithm_signature: str,
    graph_version_signature: str,
    max_hops: int,
    candidates: tuple[tuple[int, int, int, float], ...],
    node_count: int = 1,
    edge_count: int = 0,
) -> bytes:
    return json.dumps(
        {
            "algorithm_signature": algorithm_signature,
            "candidate_contributions": [
                [chunk_id, contribution.hex(), distance, seed_rank, str(DOC_B), 0]
                for chunk_id, distance, seed_rank, contribution in candidates
            ],
            "effective_max_hops": max_hops,
            "graph_version_signature": graph_version_signature,
            "ppr_scores": [
                [["canonical", index + 1], (1.0 / node_count).hex()]
                for index in range(node_count)
            ],
            "restart_vector": [[["canonical", 1], (1.0).hex()]],
            "retained_groups": [
                [
                    ["canonical", index + 1],
                    "uses",
                    ["canonical", index + 2],
                    "forward",
                    (1.0).hex(),
                    1,
                    (1.0).hex(),
                ]
                for index in range(edge_count)
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def ranking(
    ranked_ids: tuple[int, ...],
    *,
    graph_ids: tuple[int, ...] = (),
    inaccessible: int = 0,
    materialization_ms: float = 0.0,
    rerank_ms: float = 0.0,
):
    rows = tuple(SimpleNamespace(pk=value, doc_id=DOC_A) for value in ranked_ids)
    graph_id_set = set(graph_ids)
    return SimpleNamespace(
        combined_candidates=rows,
        graph_candidates=tuple(row for row in rows if row.pk in graph_id_set),
        ranked_results=rows,
        inaccessible_candidate_count=inaccessible,
        materialization_ms=materialization_ms,
        rerank_ms=rerank_ms,
    )


def stable_materializer(
    _model_cls,
    _query,
    _top_k,
    baseline_candidates,
    *,
    graph_chunk_ids=(),
    **_kwargs,
):
    baseline_ids = tuple(row.pk for row in baseline_candidates)
    graph_ids = tuple(value for value in graph_chunk_ids if value not in baseline_ids)
    return ranking((*baseline_ids, *graph_ids), graph_ids=graph_ids)


def core_fixture(*, baseline_ids: tuple[int, ...] = (1, 2, 3)):
    scope = SimpleNamespace(
        documents=(SimpleNamespace(id=DOC_A, collection_id=1),),
        allowed_doc_ids=(DOC_A,),
        allowed_collection_ids=(1,),
    )
    candidates = SimpleNamespace(
        documents=scope.documents,
        vector_chunk_ids=(baseline_ids[0],),
        trigram_chunk_ids=baseline_ids[1:2],
        exact_chunk_ids=baseline_ids[2:],
        baseline_candidates=tuple(SimpleNamespace(pk=value) for value in baseline_ids),
        graph_seeds=(GraphExpansionSeed(baseline_ids[0], 1, 1.0),),
        exact_terms=("exact",),
        vector_error=None,
        graph_seed_error=False,
    )
    graph = SimpleNamespace(
        config=shipping_config(),
        load_max_hops=2,
        allowed_doc_ids=scope.allowed_doc_ids,
        allowed_collection_ids=scope.allowed_collection_ids,
        scope_version_signature=HASH_A,
        identity_keys=(("canonical", 1), ("canonical", 2), ("canonical", 3)),
        seed_identities=(
            SimpleNamespace(
                seed_chunk_id=baseline_ids[0],
                identity_key=("canonical", 1),
            ),
        ),
        relation_groups=(),
        mentions=(),
        raw_audit_rows=(),
        artifact_provenance=(),
    )
    return scope, candidates, graph


@contextmanager
def snapshot_context(**_kwargs):
    yield object()


def hit_ranker(graph, *, one_hop_ids=(4,), ppr_ids=(5, 6)):
    def rank(_snapshot, request, *, effective_max_hops, _eval_trace, **_kwargs):
        algorithm = graph_algorithm_signature(
            replace(graph.config, max_hops=effective_max_hops)
        )
        version = HASH_C if effective_max_hops == 1 else HASH_D
        chunk_ids = one_hop_ids if effective_max_hops == 1 else ppr_ids
        _eval_trace.sink(
            trace(
                algorithm_signature=algorithm,
                graph_version_signature=version,
                max_hops=effective_max_hops,
                candidates=tuple(
                    (
                        chunk_id,
                        2 if effective_max_hops == 2 and index else 1,
                        1,
                        0.5 / (index + 1),
                    )
                    for index, chunk_id in enumerate(chunk_ids)
                ),
                node_count=2 if effective_max_hops == 1 else 3,
                edge_count=1 if effective_max_hops == 1 else 2,
            )
        )
        return GraphExpansionResult(
            chunk_ids=chunk_ids,
            diagnostics=GraphExpansionDiagnostics(
                status="hit",
                seed_count=1,
                candidate_count=len(chunk_ids),
                algorithm_signature=algorithm,
                graph_version_signature=version,
            ),
            seed_chunk_ids=tuple(item.chunk_id for item in request.seeds),
        )

    return rank


def comparison_kwargs(
    *,
    baseline_ids: tuple[int, ...] = (1, 2, 3),
    rank_graph=None,
    materializer=stable_materializer,
):
    """Return complete default dependencies plus their immutable fake snapshots."""

    scope, candidates, graph = core_fixture(baseline_ids=baseline_ids)
    kwargs = {
        "query": "relationship query",
        "collection_ids": (1,),
        "top_k": 10,
        "model_cls": object,
        "graph_config": acquisition_config(),
        "timeout_ms": 150,
        "prepare_embedding": lambda _query: (0.0,),
        "resolve_scope": lambda _collections, _config: scope,
        "authorized_snapshot": snapshot_context,
        "collect_candidates": lambda *_args, **_kwargs: candidates,
        "load_graph": lambda *_args, **_kwargs: graph,
        "rank_graph": rank_graph or hit_ranker(graph),
        "materialize_and_rerank": materializer,
    }
    return kwargs, scope, candidates, graph
