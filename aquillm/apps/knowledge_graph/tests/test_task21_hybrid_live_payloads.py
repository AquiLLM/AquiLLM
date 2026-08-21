from __future__ import annotations

from types import SimpleNamespace

import pytest

from apps.knowledge_graph.evals import task21_hybrid_live_observations as live
from apps.knowledge_graph.evals import task21_hybrid_live_parity as live_parity
from apps.knowledge_graph.evals import task21_hybrid_live_payloads as payloads
from apps.knowledge_graph.evals import task21_hybrid_live_runner as live_runner
from apps.knowledge_graph.evals import task21_hybrid_live_trace as live_trace


def _observation(case_id: str, arm: str) -> dict[str, object]:
    reranked = arm == "combined_reranked"
    return {
        "case_id": case_id,
        "ranked_chunk_ids": ["visible-chunk"],
        "graph_chunk_ids": [],
        "citation_evidence_chunk_ids": ["visible-chunk"],
        "seed_chunk_ids": ["visible-chunk"],
        "mapped_seed_chunk_ids": ["visible-chunk"],
        "projected_ranks": [],
        "repeated_projected_ranks": [],
        "latency_ms": 1.25,
        "reranker_calls": int(reranked),
        "comparison_snapshot_signature": "a" * 64,
        "candidate_trace": [
            {
                "chunk_id": "visible-chunk",
                "ordinal": 1,
                "sources": ["baseline"],
                "baseline_rank": 1,
                "direct_rank": None,
                "direct_score_hex": None,
                "extended_rank": None,
                "extended_score_hex": None,
                "fusion_score_hex": "0x0.0p+0",
                "reranker_rank": 1 if reranked else None,
            }
        ],
        "timing_trace": {
            "candidate_ms": 0.25,
            "branch_ms": 0.5,
            "fusion_ms": 0.25,
            "rerank_ms": 0.25 if reranked else 0.0,
            "total_ms": 1.25,
        },
        "authorization_status": "current",
        "graph_scheduled": arm != "vector_only",
        "inaccessible_candidate_count": 0,
        "adversarial_candidate_chunk_ids": [],
        "inaccessible_result_chunk_ids": [],
    }


def _freshness() -> dict[str, object]:
    return {
        "generation_key": "5" * 64,
        "projection_checksum": "6" * 64,
        "age_seconds": 0.0,
        "max_age_seconds": 300.0,
        "projection_keys": ["7" * 64],
        "generation_keys": ["8" * 64],
        "graph_checksums": ["9" * 64],
        "ready_bundle_checksums": ["a" * 64],
        "ontology_version": "research-v1",
        "ontology_checksum": "b" * 64,
    }


def _parity() -> dict[str, object]:
    result = {
        f"{backend}_{kind}_sha256": "c" * 64
        for backend in ("postgres", "memgraph")
        for kind in ("snapshot", "scores", "trace", "ties")
    }
    result.update(
        postgres_projected_ranks=["visible-chunk"],
        memgraph_projected_ranks=["visible-chunk"],
        comparison_inputs=[
            {
                "branch": branch,
                "ready_bundle_checksum": "d" * 64,
                "seed_checksum": character * 64,
                "seed_count": 1,
                "max_depth": depth,
                "max_nodes": 20,
                "max_edges": 40,
                "max_results": 10,
                "projection_keys": ["e" * 64],
                "generation_keys": ["f" * 64],
                "authorized_document_keys": [character * 64],
            }
            for branch, depth, character in (
                ("direct", 1, "1"),
                ("extended", 2, "2"),
            )
        ],
    )
    return result


def _payloads():
    combined = {
        arm: [_observation("visible-case", arm)] for arm in live.TASK21_HYBRID_ARMS
    }
    return payloads.build_live_evidence_payloads(
        run_id="1" * 32,
        source_commit="2" * 40,
        manifest=SimpleNamespace(
            fixture_id="kg-task20-synthetic-v1",
            fixture_checksum="3" * 64,
            manifest_checksum="4" * 64,
            chunks={"visible-chunk": object()},
        ),
        combined_observations=combined,
        combined_freshness=_freshness(),
        combined_backend_parity=_parity(),
    )


def test_live_payloads_split_frozen_metrics_from_strict_trace():
    evidence = _payloads()

    assert "candidate_trace" not in evidence.observations["direct"][0]
    assert tuple(evidence.live_trace["arms"]["direct"][0]) == (
        "case_id",
        "candidate_trace",
        "timing_trace",
        "authorization_status",
        "graph_scheduled",
        "inaccessible_candidate_count",
    )
    assert set(evidence.freshness) == {
        "generation_key",
        "projection_checksum",
        "age_seconds",
        "max_age_seconds",
    }
    assert "comparison_inputs" not in evidence.backend_parity
    live_trace.validate_live_trace(
        evidence.live_trace,
        expected_case_ids=("visible-case",),
        fixture_chunk_ids=("visible-chunk",),
    )
    live_trace.validate_live_trace_observations(
        evidence.live_trace, evidence.observations
    )


def test_parity_comparison_inputs_select_first_exact_input_for_each_branch():
    direct, extended = _parity()["comparison_inputs"]
    later_case = dict(direct, max_nodes=21)

    assert live_parity.canonical_comparison_inputs(
        [direct, extended, later_case, dict(extended)]
    ) == [direct, extended]
    with pytest.raises(RuntimeError, match="direct and extended"):
        live_parity.canonical_comparison_inputs([direct, later_case])


def test_runner_publishes_only_frozen_metrics_plus_bound_live_trace(monkeypatch):
    case = {
        "id": "visible-case",
        "accessible_collection_ids": (),
        "documents": (
            {
                "collection_id": "private",
                "chunks": ({"chunk_id": "visible-chunk"},),
            },
        ),
    }
    authority = SimpleNamespace(
        manifest=SimpleNamespace(
            fixture_id="kg-task20-synthetic-v1",
            fixture_checksum="3" * 64,
            manifest_checksum="4" * 64,
            chunks={"visible-chunk": object()},
        ),
        prepare_case=lambda _case: None,
    )

    class FakeExecutor:
        def __init__(self, _settings):
            self.ready_scopes = [object()]
            self.parity_calls = [object()]

    monkeypatch.setattr(
        live_runner, "load_live_fixture_authority", lambda _path: authority
    )
    monkeypatch.setattr(live_runner, "ProductionArmExecutor", FakeExecutor)
    monkeypatch.setattr(
        live_runner,
        "logical_fixture",
        lambda: SimpleNamespace(retrieval_cases=(case,)),
    )
    monkeypatch.setattr(live_runner, "_freshness", lambda *_args: _freshness())
    monkeypatch.setattr(
        live_runner, "build_live_backend_parity", lambda **_kwargs: _parity()
    )
    monkeypatch.setattr(
        live,
        "generate_case_arms",
        lambda **_kwargs: {
            arm: [_observation("visible-case", arm)]
            for arm in live.TASK21_HYBRID_ARMS
        },
    )
    report = {}
    published = {}
    monkeypatch.setattr(
        live_runner,
        "build_task21_hybrid_report",
        lambda **kwargs: report.update(kwargs),
    )
    from apps.documents.services import hybrid_graph_dependencies
    from apps.knowledge_graph.evals import task21_hybrid_live_evidence

    monkeypatch.setattr(
        hybrid_graph_dependencies,
        "django_hybrid_retrieval_settings",
        lambda: object(),
    )
    monkeypatch.setattr(
        task21_hybrid_live_evidence,
        "publish_live_artifacts",
        lambda **kwargs: published.update(kwargs),
    )
    live_runner.run_production_live_observations(
        run_id="1" * 32,
        source_commit="2" * 40,
        fixture_manifest=SimpleNamespace(),
        runtime_identity={"complete": True},
        output_paths=object(),
    )

    assert "candidate_trace" not in report["observations"]["direct"][0]
    assert "comparison_inputs" not in report["backend_parity"]
    assert "projection_keys" not in report["freshness"]
    assert published["observations"] == report["observations"]
    assert published["live_trace"]["source_commit"] == "2" * 40
    assert published["expected_case_ids"] == ("visible-case",)
    assert published["fixture_chunk_ids"] == ("visible-chunk",)
