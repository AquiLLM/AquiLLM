from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from apps.knowledge_graph.evals import task21_hybrid_live_execution as execution
from apps.knowledge_graph.evals import task21_hybrid_live_observations as live
from apps.knowledge_graph.retrieval.branch_contracts import BranchStatusV1


def _trace(chunk_id: str, ordinal: int) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "ordinal": ordinal,
        "sources": ["baseline"],
        "baseline_rank": ordinal,
        "direct_rank": None,
        "direct_score_hex": None,
        "extended_rank": None,
        "extended_score_hex": None,
        "fusion_score_hex": "0x0.0p+0",
        "reranker_rank": None,
    }


def _observation(case_id: str, arm: str) -> dict[str, object]:
    reranked = arm == "combined_reranked"
    trace = _trace("visible-chunk", 1)
    trace["reranker_rank"] = 1 if reranked else None
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
        "reranker_calls": 1 if reranked else 0,
        "comparison_snapshot_signature": "a" * 64,
        "candidate_trace": [trace],
        "timing_trace": {
            "candidate_ms": 0.25,
            "branch_ms": 0.5,
            "fusion_ms": 0.25,
            "rerank_ms": 0.25 if reranked else 0.0,
            "total_ms": 1.25,
        },
        "authorization_status": "current",
        "graph_scheduled": arm not in {"vector_only"},
        "inaccessible_candidate_count": 0,
        "adversarial_candidate_chunk_ids": [],
        "inaccessible_result_chunk_ids": [],
    }


class Executor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool, bool, int]] = []

    def run_arm(self, *, case, prepared, spec):
        assert prepared is _PREPARED
        self.calls.append(
            (
                spec["name"],
                spec["direct_enabled"],
                spec["extended_enabled"],
                spec["reranker_calls"],
            )
        )
        return _observation(case["id"], spec["name"])


_PREPARED = object()


@dataclass(frozen=True)
class _Settings:
    graph_direct_enabled: bool = True
    graph_extended_enabled: bool = True


def test_case_runner_originates_all_five_frozen_arms_through_executor():
    executor = Executor()

    result = live.generate_case_arms(
        case={"id": "visible-case"}, prepared=_PREPARED, executor=executor
    )

    assert tuple(result) == live.TASK21_HYBRID_ARMS
    assert executor.calls == [
        ("vector_only", False, False, 0),
        ("direct", True, False, 0),
        ("extended", False, True, 0),
        ("combined", True, True, 0),
        ("combined_reranked", True, True, 1),
    ]
    assert all(len(rows) == 1 for rows in result.values())


def test_empty_authorized_case_never_constructs_or_schedules_graph_provider():
    class ForbiddenExecutor:
        def run_arm(self, **_kwargs):
            raise AssertionError("negative authorization case scheduled graph work")

    rows = live.generate_case_arms(
        case={
            "id": "no_accessible_evidence",
            "adversarial_chunk_ids": ("private-only-001",),
        },
        prepared=None,
        executor=ForbiddenExecutor(),
    )

    for arm, observations in rows.items():
        assert observations == [
            {
                **live.negative_observation(
                    case_id="no_accessible_evidence",
                    arm=arm,
                    adversarial_chunk_ids=("private-only-001",),
                ),
                "graph_scheduled": False,
                "adversarial_candidate_chunk_ids": ["private-only-001"],
                "inaccessible_result_chunk_ids": [],
            }
        ]


def test_production_executor_records_repeatable_branch_scores_and_one_rerank(
    monkeypatch,
):
    seed, graph = SimpleNamespace(pk=11), SimpleNamespace(pk=12)
    snapshot = SimpleNamespace(
        baseline_candidates=(seed,),
        graph_seeds=(SimpleNamespace(chunk_id=11),),
        vector_ms=1.0,
        trigram_ms=2.0,
        exact_ms=3.0,
    )

    def envelope(key, score):
        return SimpleNamespace(
            status=BranchStatusV1.SUCCEEDED,
            result=SimpleNamespace(
                candidates=(SimpleNamespace(chunk_key=key, rank=1, score=score),)
            ),
        )
    traces = iter(
        execution.GraphExecutionTrace(
            runtime=SimpleNamespace(
                direct=envelope("d" * 64, 0.75),
                extended=envelope("e" * 64, 0.5),
                shared=None,
            ),
            topology=SimpleNamespace(calls=[]),
            materializer=SimpleNamespace(
                by_key={"d" * 64: 12, "e" * 64: 12}
            ),
            pool=(seed, graph),
            graph_ms=4.0,
        )
        for _index in range(2)
    )
    monkeypatch.setattr(execution, "_graph_pool", lambda *_args: next(traces))
    monkeypatch.setattr(
        execution._trace, "mapped_seed_symbols", lambda *_args: ("seed",)
    )
    monkeypatch.setattr(
        execution,
        "_ranked_pool",
        lambda _prepared, _pool, calls: (
            ((graph, seed), 5.0, 0) if calls == 1 else None
        ),
    )
    prepared = SimpleNamespace(
        authorization=SimpleNamespace(authorization_context_signature="a" * 64),
        snapshot=snapshot,
        case={
            "id": "case",
            "query": "query",
            "task21_expected_arms": ["combined_reranked"],
        },
        chunk_symbols_by_pk={11: "seed", 12: "graph"},
        accessible_chunk_symbols=frozenset({"seed", "graph"}),
        adversarial_chunk_symbols=("private",),
    )

    observed = execution.ProductionArmExecutor(_Settings()).run_arm(
        case=prepared.case,
        prepared=prepared,
        spec={
            "name": "combined_reranked",
            "direct_enabled": True,
            "extended_enabled": True,
            "reranker_calls": 1,
        },
    )

    assert observed["ranked_chunk_ids"] == ["graph", "seed"]
    assert observed["projected_ranks"] == ["graph"]
    assert observed["repeated_projected_ranks"] == ["graph"]
    assert observed["candidate_trace"][1]["direct_score_hex"] == (0.75).hex()
    assert observed["candidate_trace"][1]["extended_score_hex"] == (0.5).hex()
    assert observed["reranker_calls"] == 1


def test_atomic_live_publish_is_0600_attested_and_never_overwrites(tmp_path):
    paths = live.LiveOutputPaths(
        observations=tmp_path / "observations.json",
        freshness=tmp_path / "freshness.json",
        backend_parity=tmp_path / "backend-parity.json",
        attestation=tmp_path / "observation-attestation.json",
    )
    runtime = {
        "schema": live.RUNTIME_IDENTITY_SCHEMA,
        "run_id": "1" * 32,
        "source_commit": "2" * 40,
        "config_sha256": "3" * 64,
        "images": {name: "sha256:" + "4" * 64 for name in live.RUNTIME_SERVICES},
        "complete": True,
    }
    artifacts = live.publish_live_artifacts(
        paths=paths,
        run_id="1" * 32,
        source_commit="2" * 40,
        runtime_identity=runtime,
        observations={arm: [] for arm in live.TASK21_HYBRID_ARMS},
        freshness={
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
        },
        backend_parity={"status": "exact"},
    )

    assert artifacts == paths
    for path in paths:
        assert path.is_file()
        if os.name != "nt":
            assert path.stat().st_mode & 0o777 == 0o600
    attestation = json.loads(paths.attestation.read_text(encoding="utf-8"))
    assert attestation["run_id"] == "1" * 32
    assert attestation["source_commit"] == "2" * 40
    assert attestation["config_sha256"] == "3" * 64
    assert attestation["images"] == runtime["images"]
    assert attestation["artifact_sha256"] == {
        name: hashlib.sha256(getattr(paths, name).read_bytes()).hexdigest()
        for name in ("observations", "freshness", "backend_parity")
    }
    with pytest.raises(FileExistsError):
        live.publish_live_artifacts(
            paths=paths,
            run_id="1" * 32,
            source_commit="2" * 40,
            runtime_identity=runtime,
            observations={arm: [] for arm in live.TASK21_HYBRID_ARMS},
            freshness=json.loads(paths.freshness.read_text(encoding="utf-8")),
            backend_parity={"status": "exact"},
        )
