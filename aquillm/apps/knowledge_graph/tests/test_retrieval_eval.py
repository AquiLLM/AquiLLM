"""Core one-snapshot contracts for the Task20 retrieval comparison."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

from apps.knowledge_graph.evals import run_kg_eval
from apps.knowledge_graph.retrieval.ppr import graph_algorithm_signature
from apps.knowledge_graph.retrieval.types import (
    GraphExpansionDiagnostics,
    GraphExpansionResult,
    GraphExpansionSeed,
)
from apps.knowledge_graph.tests import retrieval_eval_support as support

_DOC_A = support.DOC_A
_HASH_A = support.HASH_A
_HASH_C = support.HASH_C
_HASH_D = support.HASH_D
_shipping_config = support.shipping_config
_acquisition_config = support.acquisition_config
_trace = support.trace
_ranking = support.ranking
_stable_materializer = support.stable_materializer
_core_fixture = support.core_fixture
_snapshot_context = support.snapshot_context
_hit_ranker = support.hit_ranker


def test_one_snapshot_comparison_uses_shipping_seams_once_in_required_order():
    calls: list[object] = []
    first_documents = (SimpleNamespace(id=_DOC_A, collection_id=1),)
    second_documents = (SimpleNamespace(id=_DOC_A, collection_id=1),)
    scopes = iter(
        (
            SimpleNamespace(
                documents=first_documents,
                allowed_doc_ids=(_DOC_A,),
                allowed_collection_ids=(1,),
            ),
            SimpleNamespace(
                documents=second_documents,
                allowed_doc_ids=(_DOC_A,),
                allowed_collection_ids=(1,),
            ),
        )
    )
    seeds = (GraphExpansionSeed(101, 1, 1.0),)
    candidate_snapshot = SimpleNamespace(
        documents=second_documents,
        vector_chunk_ids=(101, 102),
        trigram_chunk_ids=(102, 103),
        exact_chunk_ids=(104,),
        baseline_candidates=(
            SimpleNamespace(pk=101),
            SimpleNamespace(pk=102),
            SimpleNamespace(pk=103),
            SimpleNamespace(pk=104),
        ),
        graph_seeds=seeds,
        exact_terms=("Orion",),
        vector_error=None,
        graph_seed_error=False,
    )
    graph_snapshot = SimpleNamespace(
        config=_shipping_config(),
        load_max_hops=2,
        allowed_doc_ids=(_DOC_A,),
        allowed_collection_ids=(1,),
        scope_version_signature=_HASH_A,
        identity_keys=(("canonical", 1), ("canonical", 2)),
        seed_identities=(
            SimpleNamespace(seed_chunk_id=101, identity_key=("canonical", 1)),
        ),
        relation_groups=(),
        mentions=(),
        raw_audit_rows=((0, "artifact", (1, "ontology-v1")),),
    )
    one_hop_algorithm = graph_algorithm_signature(
        replace(_shipping_config(), max_hops=1)
    )
    ppr_algorithm = graph_algorithm_signature(_shipping_config())

    def prepare_embedding(query):
        calls.append(("embedding", query))
        return (0.1, 0.2)

    def resolve_scope(collection_ids, graph_config):
        calls.append(("scope", collection_ids, graph_config))
        return next(scopes)

    @contextmanager
    def authorized_snapshot(*, timeout_ms):
        calls.append(("enter", timeout_ms))
        try:
            yield "deadline"
        finally:
            calls.append("exit")

    def collect_candidates(model_cls, query, top_k, documents, **kwargs):
        calls.append(("candidates", model_cls, query, top_k, documents, kwargs))
        assert kwargs["query_embedding"] == (0.1, 0.2)
        assert kwargs["graph_config"] is acquisition_config
        return candidate_snapshot

    def load_graph(request, *, load_max_hops):
        calls.append(("load", request, load_max_hops))
        assert load_max_hops == 2
        assert request.seeds == seeds
        return graph_snapshot

    def rank_graph(snapshot, request, *, effective_max_hops, _eval_trace, _deadline):
        calls.append(("rank", effective_max_hops, snapshot, request, _deadline))
        algorithm = one_hop_algorithm if effective_max_hops == 1 else ppr_algorithm
        version = _HASH_C if effective_max_hops == 1 else _HASH_D
        candidates = (
            ((201, 1, 1, 0.4),)
            if effective_max_hops == 1
            else ((201, 1, 1, 0.4), (202, 2, 1, 0.2))
        )
        _eval_trace.sink(
            _trace(
                algorithm_signature=algorithm,
                graph_version_signature=version,
                max_hops=effective_max_hops,
                candidates=candidates,
            )
        )
        return GraphExpansionResult(
            chunk_ids=tuple(item[0] for item in candidates),
            diagnostics=GraphExpansionDiagnostics(
                status="hit",
                seed_count=1,
                candidate_count=len(candidates),
                algorithm_signature=algorithm,
                graph_version_signature=version,
            ),
            seed_chunk_ids=(101,),
        )

    acquisition_config = _acquisition_config()
    result = run_kg_eval.run_one_snapshot_comparison(
        query="How does Orion rank evidence?",
        collection_ids=(1,),
        top_k=10,
        model_cls=object,
        graph_config=acquisition_config,
        timeout_ms=150,
        prepare_embedding=prepare_embedding,
        resolve_scope=resolve_scope,
        authorized_snapshot=authorized_snapshot,
        collect_candidates=collect_candidates,
        load_graph=load_graph,
        rank_graph=rank_graph,
        materialize_and_rerank=_stable_materializer,
    )

    assert [call[0] if isinstance(call, tuple) else call for call in calls] == [
        "embedding",
        "scope",
        "enter",
        "scope",
        "exit",
        "enter",
        "candidates",
        "exit",
        "enter",
        "load",
        "exit",
        "enter",
        "rank",
        "exit",
        "enter",
        "rank",
        "exit",
        "enter",
        "rank",
        "exit",
    ]
    assert (
        sum(isinstance(call, tuple) and call[0] == "candidates" for call in calls) == 1
    )
    assert sum(isinstance(call, tuple) and call[0] == "load" for call in calls) == 1
    assert result["collection_scope"] == (1,)
    assert set(result["arms"]) == {"vector_only", "one_hop", "ppr_v1"}
    assert {
        arm["comparison_snapshot_signature"] for arm in result["arms"].values()
    } == {result["comparison_snapshot_signature"]}
    assert (
        result["arms"]["one_hop"]["algorithm_signature"]
        != result["arms"]["ppr_v1"]["algorithm_signature"]
    )
    assert (
        result["arms"]["one_hop"]["graph_version_signature"]
        != result["arms"]["ppr_v1"]["graph_version_signature"]
    )
    assert result["arms"]["one_hop"]["ppr_iterations"] == 8
    assert result["arms"]["ppr_v1"]["ppr_iterations"] == 8
    assert result["arms"]["ppr_v1"]["distance_2_novel_fraction"] == 0.5
    assert result["deterministic_repeated_ppr"] is True
    assert "private_trace" not in json.dumps(run_kg_eval._thaw(result))
    assert "ppr_scores" not in json.dumps(run_kg_eval._thaw(result))


def test_all_arms_share_full_baseline_and_exact_production_rerank_seam():
    scope, candidates, graph = _core_fixture()
    call_names = iter(
        ("baseline", "fail_open_miss", "fail_open_error", "one_hop", "ppr", "repeat")
    )
    calls: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    def materialize(
        _model,
        _query,
        _top_k,
        baseline_candidates,
        *,
        graph_chunk_ids=(),
        **_kwargs,
    ):
        baseline_ids = tuple(row.pk for row in baseline_candidates)
        graph_ids = tuple(graph_chunk_ids)
        calls.append((next(call_names), baseline_ids, graph_ids))
        return _ranking(
            tuple(reversed((*baseline_ids, *graph_ids))),
            graph_ids=graph_ids,
        )

    result = run_kg_eval.run_one_snapshot_comparison(
        query="exact trigram changes rank",
        collection_ids=(1,),
        top_k=10,
        model_cls=object,
        graph_config=_acquisition_config(),
        timeout_ms=150,
        prepare_embedding=lambda _query: (0.0,),
        resolve_scope=lambda _collections, _config: scope,
        authorized_snapshot=_snapshot_context,
        collect_candidates=lambda *_args, **_kwargs: candidates,
        load_graph=lambda *_args, **_kwargs: graph,
        rank_graph=_hit_ranker(graph),
        materialize_and_rerank=materialize,
    )

    assert calls == [
        ("baseline", (1, 2, 3), ()),
        ("fail_open_miss", (1, 2, 3), ()),
        ("fail_open_error", (1, 2, 3), ()),
        ("one_hop", (1, 2, 3), (4,)),
        ("ppr", (1, 2, 3), (5, 6)),
        ("repeat", (1, 2, 3), (5, 6)),
    ]
    assert result["arms"]["vector_only"]["ranked_chunk_ids"] == (3, 2, 1)
    assert result["arms"]["vector_only"]["ranked_chunk_ids"] != (
        candidates.vector_chunk_ids
    )
