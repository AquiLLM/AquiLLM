"""Security, timing, and retained-trace contracts for Task20 retrieval eval."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from apps.knowledge_graph.evals import run_kg_eval
from apps.knowledge_graph.tests import retrieval_eval_support as support
from apps.knowledge_graph.tests.test_retrieval_eval_metrics import task23_inputs


def test_stable_graph_but_nondeterministic_final_reranker_aborts_comparison():
    calls = 0

    def materialize(_model, _query, _top_k, baseline, *, graph_chunk_ids=(), **_kw):
        nonlocal calls
        baseline_ids = tuple(row.pk for row in baseline)
        if tuple(graph_chunk_ids) == (5, 6):
            calls += 1
            ordered = (1, 2, 3, 5, 6) if calls == 1 else (1, 2, 3, 6, 5)
            return support.ranking(ordered, graph_ids=(5, 6))
        return support.ranking(
            (*baseline_ids, *graph_chunk_ids), graph_ids=graph_chunk_ids
        )

    kwargs, *_rest = support.comparison_kwargs(materializer=materialize)
    with pytest.raises(run_kg_eval.ComparisonAborted, match="deterministic repeated"):
        run_kg_eval.run_one_snapshot_comparison(**kwargs)


def test_permission_refetch_leak_count_is_measured_not_hardcoded():
    def materialize(_model, _query, _top_k, baseline, *, graph_chunk_ids=(), **_kw):
        baseline_ids = tuple(row.pk for row in baseline)
        return support.ranking(
            (*baseline_ids, *graph_chunk_ids),
            graph_ids=graph_chunk_ids,
            inaccessible=1 if tuple(graph_chunk_ids) == (5, 6) else 0,
        )

    kwargs, *_rest = support.comparison_kwargs(materializer=materialize)
    result = run_kg_eval.run_one_snapshot_comparison(**kwargs)

    assert result["arms"]["vector_only"]["inaccessible_result_count"] == 0
    assert result["arms"]["one_hop"]["inaccessible_result_count"] == 0
    assert result["arms"]["ppr_v1"]["inaccessible_result_count"] == 1


def test_latency_includes_candidate_graph_load_materialization_and_rerank():
    times = iter((0.000, 0.001, 0.001, 0.003, 0.003, 0.006, 0.006, 0.010, 0.010, 0.015))

    def materialize(_model, _query, _top_k, baseline, *, graph_chunk_ids=(), **_kw):
        baseline_ids = tuple(row.pk for row in baseline)
        return support.ranking(
            (*baseline_ids, *graph_chunk_ids),
            graph_ids=graph_chunk_ids,
            materialization_ms=5.0,
            rerank_ms=7.0,
        )

    kwargs, *_rest = support.comparison_kwargs(materializer=materialize)
    result = run_kg_eval.run_one_snapshot_comparison(
        **kwargs,
        clock=lambda: next(times),
    )

    assert result["arms"]["vector_only"]["latency_ms"] == 13.0
    assert result["arms"]["one_hop"]["latency_ms"] == 18.0
    assert result["arms"]["ppr_v1"]["latency_ms"] == 19.0
    assert result["arms"]["vector_only"]["graph_added_latency_ms"] == 0.0
    assert result["arms"]["one_hop"]["graph_added_latency_ms"] == 10.0
    assert result["arms"]["ppr_v1"]["graph_added_latency_ms"] == 11.0


def test_hop_counts_and_distance_two_value_come_from_retained_returned_trace():
    kwargs, *_rest = support.comparison_kwargs()
    result = run_kg_eval.run_one_snapshot_comparison(**kwargs)

    assert result["arms"]["one_hop"]["node_count"] == 2
    assert result["arms"]["one_hop"]["edge_count"] == 1
    assert result["arms"]["ppr_v1"]["node_count"] == 3
    assert result["arms"]["ppr_v1"]["edge_count"] == 2
    assert result["arms"]["ppr_v1"]["distance_2_novel_fraction"] == 0.5

    def omit_distance_two(
        _model, _query, _top_k, baseline, *, graph_chunk_ids=(), **_kw
    ):
        baseline_ids = tuple(row.pk for row in baseline)
        returned = tuple(value for value in graph_chunk_ids if value != 6)
        return support.ranking((*baseline_ids, *returned), graph_ids=returned)

    kwargs, *_rest = support.comparison_kwargs(materializer=omit_distance_two)
    omitted = run_kg_eval.run_one_snapshot_comparison(**kwargs)
    assert omitted["arms"]["ppr_v1"]["distance_2_novel_fraction"] == 0.0


def test_comparison_aborts_on_scope_mismatch_before_candidate_search():
    kwargs, scope, _candidates, _graph = support.comparison_kwargs()
    changed = SimpleNamespace(
        documents=scope.documents,
        allowed_doc_ids=(support.DOC_B,),
        allowed_collection_ids=(1,),
    )
    scopes = iter((scope, changed))
    candidate_called = False

    def collect(*_args, **_kwargs):
        nonlocal candidate_called
        candidate_called = True

    kwargs["resolve_scope"] = lambda *_args: next(scopes)
    kwargs["collect_candidates"] = collect
    with pytest.raises(run_kg_eval.ComparisonAborted, match="scope"):
        run_kg_eval.run_one_snapshot_comparison(**kwargs)
    assert candidate_called is False


@pytest.mark.parametrize("failure", ("permission", "reranker", "snapshot"))
def test_task23_hybrid_report_rejects_permission_reranker_and_snapshot_drift(failure):
    cases, observations, freshness, parity = task23_inputs()
    if failure == "permission":
        observations["combined"][0]["ranked_chunk_ids"].append("private")
        observations["combined"][0]["inaccessible_result_chunk_ids"] = ["private"]
    elif failure == "reranker":
        observations["combined_reranked"][0]["reranker_calls"] = 0
    else:
        observations["direct"][0]["comparison_snapshot_signature"] = "9" * 64

    with pytest.raises(run_kg_eval.Task21HybridEvalError):
        run_kg_eval.build_task21_hybrid_report(
            cases=cases,
            observations=observations,
            freshness=freshness,
            backend_parity=parity,
        )


def test_task23_hybrid_report_requires_observed_adversarial_candidates():
    cases, observations, freshness, parity = task23_inputs()
    for rows in observations.values():
        rows[0]["adversarial_candidate_chunk_ids"] = []

    with pytest.raises(run_kg_eval.Task21HybridEvalError, match="adversarial"):
        run_kg_eval.build_task21_hybrid_report(
            cases=cases,
            observations=observations,
            freshness=freshness,
            backend_parity=parity,
        )


def test_task23_hybrid_report_rejects_fabricated_adversarial_fixture_ids():
    cases, observations, freshness, parity = task23_inputs()
    for rows in observations.values():
        rows[0]["adversarial_candidate_chunk_ids"] = ["invented-private"]

    with pytest.raises(run_kg_eval.Task21HybridEvalError, match="fixture"):
        run_kg_eval.build_task21_hybrid_report(
            cases=cases,
            observations=observations,
            freshness=freshness,
            backend_parity=parity,
        )


def test_task23_hybrid_report_requires_privacy_fixture_in_every_arm():
    cases, observations, freshness, parity = task23_inputs()
    observations["direct"][0]["adversarial_candidate_chunk_ids"] = []

    with pytest.raises(run_kg_eval.Task21HybridEvalError, match="fixture"):
        run_kg_eval.build_task21_hybrid_report(
            cases=cases,
            observations=observations,
            freshness=freshness,
            backend_parity=parity,
        )


def test_task23_hybrid_report_requires_each_intended_privacy_case_fixture():
    cases, observations, freshness, parity = task23_inputs()
    cases = list(cases)
    cases.append(
        {
            "id": "privacy-without-fixture",
            "privacy_intent": "security-sensitive: private evidence must stay excluded",
            "accessible_collection_ids": ["collection-a"],
            "documents": [
                {
                    "doc_id": "public-only",
                    "collection_id": "collection-a",
                    "chunks": [{"chunk_id": "public-only", "text": "Public."}],
                }
            ],
            "expected_retrieval_chunk_ids": ["public-only"],
            "quality_tags": ["inaccessible_neighbor"],
        }
    )
    for arm, rows in observations.items():
        row = copy.deepcopy(rows[0])
        row.update(
            case_id="privacy-without-fixture",
            ranked_chunk_ids=["public-only"],
            graph_chunk_ids=[],
            citation_evidence_chunk_ids=["public-only"],
            seed_chunk_ids=["public-only"],
            mapped_seed_chunk_ids=["public-only"],
            projected_ranks=[],
            repeated_projected_ranks=[],
            adversarial_candidate_chunk_ids=[],
            inaccessible_result_chunk_ids=[],
        )
        observations[arm].append(row)

    with pytest.raises(run_kg_eval.Task21HybridEvalError, match="privacy fixture"):
        run_kg_eval.build_task21_hybrid_report(
            cases=cases,
            observations=observations,
            freshness=freshness,
            backend_parity=parity,
        )


def test_task23_hybrid_report_allows_empty_adversarial_for_nonprivacy_case():
    cases, observations, freshness, parity = task23_inputs()
    cases = list(cases)
    cases.append(
        {
            "id": "public-only",
            "privacy_intent": "public collection retrieval",
            "accessible_collection_ids": ["collection-a"],
            "documents": [
                {
                    "doc_id": "public-only",
                    "collection_id": "collection-a",
                    "chunks": [{"chunk_id": "public-only", "text": "Public."}],
                }
            ],
            "expected_retrieval_chunk_ids": ["public-only"],
            "quality_tags": ["relationship"],
        }
    )
    for rows in observations.values():
        row = copy.deepcopy(rows[0])
        row.update(
            case_id="public-only",
            ranked_chunk_ids=["public-only"],
            graph_chunk_ids=[],
            citation_evidence_chunk_ids=["public-only"],
            seed_chunk_ids=["public-only"],
            mapped_seed_chunk_ids=["public-only"],
            projected_ranks=[],
            repeated_projected_ranks=[],
            adversarial_candidate_chunk_ids=[],
            inaccessible_result_chunk_ids=[],
        )
        rows.append(row)

    report = run_kg_eval.build_task21_hybrid_report(
        cases=cases,
        observations=observations,
        freshness=freshness,
        backend_parity=parity,
    )

    assert report["observed_adversarial_candidate_count"] == len(observations)
