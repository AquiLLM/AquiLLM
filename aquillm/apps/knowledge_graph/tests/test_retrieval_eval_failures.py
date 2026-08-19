"""Atomic failure and deterministic-repeat contracts for Task20 retrieval eval."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace

import pytest

from apps.documents.services.chunk_search_candidates import CandidateScopeLimit
from apps.knowledge_graph.evals import run_kg_eval
from apps.knowledge_graph.retrieval.ppr import graph_algorithm_signature
from apps.knowledge_graph.retrieval.types import (
    GraphExpansionDiagnostics,
    GraphExpansionResult,
)
from apps.knowledge_graph.tests import retrieval_eval_support as support


def test_comparison_aborts_when_repeated_ppr_order_or_trace_metrics_change():
    kwargs, _scope, _candidates, graph = support.comparison_kwargs()
    ppr_calls = 0

    def rank(_snapshot, request, *, effective_max_hops, _eval_trace, **_kwargs):
        nonlocal ppr_calls
        algorithm = graph_algorithm_signature(
            replace(graph.config, max_hops=effective_max_hops)
        )
        version = support.HASH_C if effective_max_hops == 1 else support.HASH_D
        chunk_ids = (4,) if effective_max_hops == 1 else (5, 6)
        if effective_max_hops == 2:
            ppr_calls += 1
        contribution = 0.5 if ppr_calls <= 1 else 0.25
        _eval_trace.sink(
            support.trace(
                algorithm_signature=algorithm,
                graph_version_signature=version,
                max_hops=effective_max_hops,
                candidates=tuple(
                    (chunk_id, 1 + index, 1, contribution / (index + 1))
                    for index, chunk_id in enumerate(chunk_ids)
                ),
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
            seed_chunk_ids=tuple(seed.chunk_id for seed in request.seeds),
        )

    kwargs["rank_graph"] = rank
    with pytest.raises(run_kg_eval.ComparisonAborted, match="deterministic repeated"):
        run_kg_eval.run_one_snapshot_comparison(**kwargs)


def test_all_hit_comparison_still_measures_exact_fail_open_composition():
    kwargs, *_rest = support.comparison_kwargs()

    result = run_kg_eval.run_one_snapshot_comparison(**kwargs)

    assert result["fail_open_miss_observation_count"] == 1
    assert result["fail_open_error_observation_count"] == 1
    assert result["exact_fail_open_parity"] is True


def test_measured_fail_open_requires_production_overlay_error_result(monkeypatch):
    from apps.documents.services import chunk_search

    kwargs, *_rest = support.comparison_kwargs()

    def invalid_overlay(*_args, **_kwargs):
        return (99,), {"graph_status": "hit", "graph_candidate_count": 1}

    monkeypatch.setattr(chunk_search, "_apply_graph_overlay", invalid_overlay)

    with pytest.raises(run_kg_eval.ComparisonAborted, match="fail-open overlay"):
        run_kg_eval.run_one_snapshot_comparison(**kwargs)


def test_measured_fail_open_uses_exact_production_miss_and_error_capabilities(
    monkeypatch,
):
    from apps.documents.services import chunk_search

    kwargs, *_rest = support.comparison_kwargs()
    original = chunk_search._apply_graph_overlay
    observed: list[tuple[str | None, object | None]] = []

    def recording_overlay(*args, **kwargs):
        observed.append(
            (kwargs.get("preflight_status"), kwargs.get("_eval_failure_capability"))
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(chunk_search, "_apply_graph_overlay", recording_overlay)

    result = run_kg_eval.run_one_snapshot_comparison(**kwargs)

    assert observed == [
        ("miss", chunk_search._EVALUATION_GRAPH_MISS),
        (None, chunk_search._EVALUATION_GRAPH_FAILURE),
    ]
    assert result["fail_open_miss_observation_count"] == 1
    assert result["fail_open_error_observation_count"] == 1


@pytest.mark.parametrize("status", ("miss", "error"))
def test_ordinary_graph_miss_or_error_returns_exact_reranked_baseline(status):
    kwargs, _scope, _candidates, graph = support.comparison_kwargs()

    def rank(_snapshot, request, *, effective_max_hops, **_kwargs):
        algorithm = graph_algorithm_signature(
            replace(graph.config, max_hops=effective_max_hops)
        )
        return GraphExpansionResult(
            chunk_ids=(),
            diagnostics=GraphExpansionDiagnostics(
                status=status,
                seed_count=1,
                candidate_count=0,
                algorithm_signature=algorithm,
            ),
            seed_chunk_ids=tuple(seed.chunk_id for seed in request.seeds),
        )

    kwargs["rank_graph"] = rank
    result = run_kg_eval.run_one_snapshot_comparison(**kwargs)
    baseline = result["arms"]["vector_only"]["ranked_chunk_ids"]

    assert result["arms"]["one_hop"]["ranked_chunk_ids"] == baseline
    assert result["arms"]["ppr_v1"]["ranked_chunk_ids"] == baseline
    assert result["arms"]["one_hop"]["graph_hit_rate"] == 0.0
    assert result["arms"]["ppr_v1"]["graph_hit_rate"] == 0.0
    expected_miss = 3 if status == "miss" else 1
    expected_error = 3 if status == "error" else 1
    assert result["fail_open_miss_observation_count"] == expected_miss
    assert result["fail_open_error_observation_count"] == expected_error


@pytest.mark.parametrize("failure", ("timeout", "cap"))
def test_timeout_or_snapshot_cap_failure_aborts_entire_comparison(failure):
    kwargs, *_rest = support.comparison_kwargs()
    if failure == "timeout":

        @contextmanager
        def snapshot(**_kwargs):
            raise TimeoutError("deadline")
            yield

        kwargs["authorized_snapshot"] = snapshot
    else:
        kwargs["collect_candidates"] = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            CandidateScopeLimit("cap")
        )

    with pytest.raises(run_kg_eval.ComparisonAborted):
        run_kg_eval.run_one_snapshot_comparison(**kwargs)


def test_slow_reranking_does_not_consume_fresh_graph_query_budgets():
    kwargs, *_rest = support.comparison_kwargs()
    now = [0.0]
    entered_at: list[float] = []

    @contextmanager
    def fresh_query_budget(**_kwargs):
        started = now[0]
        entered_at.append(started)
        yield object()
        if now[0] - started > 0.150:
            raise TimeoutError("query budget")

    original_materializer = kwargs["materialize_and_rerank"]

    def slow_reranker(*args, **inner_kwargs):
        now[0] += 0.200
        return original_materializer(*args, **inner_kwargs)

    kwargs.update(
        authorized_snapshot=fresh_query_budget,
        materialize_and_rerank=slow_reranker,
        clock=lambda: now[0],
    )

    result = run_kg_eval.run_one_snapshot_comparison(**kwargs)

    assert result["deterministic_repeated_ppr"] is True
    assert len(entered_at) == 6


def test_slow_graph_phase_still_aborts_its_fresh_query_budget():
    kwargs, *_rest = support.comparison_kwargs()
    now = [0.0]

    @contextmanager
    def fresh_query_budget(**_kwargs):
        started = now[0]
        yield object()
        if now[0] - started > 0.150:
            raise TimeoutError("query budget")

    original_ranker = kwargs["rank_graph"]

    def slow_ranker(*args, **inner_kwargs):
        now[0] += 0.200
        return original_ranker(*args, **inner_kwargs)

    kwargs.update(
        authorized_snapshot=fresh_query_budget,
        rank_graph=slow_ranker,
        clock=lambda: now[0],
    )

    with pytest.raises(run_kg_eval.ComparisonAborted, match="timed out"):
        run_kg_eval.run_one_snapshot_comparison(**kwargs)
