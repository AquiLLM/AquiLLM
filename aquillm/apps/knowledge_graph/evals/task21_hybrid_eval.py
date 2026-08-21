"""Provider-neutral five-arm Task21 hybrid retrieval evaluation contracts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType

from .retrieval_eval_statistics import mean, percentile_95
from .task21_hybrid_eval_validation import (
    Task21HybridEvalError,
    exact_ids,
    exact_sha,
    score_case,
    validate_cases,
    validate_freshness,
    validate_parity,
)

TASK21_HYBRID_ARMS = (
    "vector_only",
    "direct",
    "extended",
    "combined",
    "combined_reranked",
)
_ARM_SPECS = (
    ("vector_only", False, False, 0),
    ("direct", True, False, 0),
    ("extended", False, True, 0),
    ("combined", True, True, 0),
    ("combined_reranked", True, True, 1),
)
_OBSERVATION_FIELDS = frozenset(
    {
        "case_id",
        "ranked_chunk_ids",
        "graph_chunk_ids",
        "citation_evidence_chunk_ids",
        "seed_chunk_ids",
        "mapped_seed_chunk_ids",
        "projected_ranks",
        "repeated_projected_ranks",
        "latency_ms",
        "reranker_calls",
        "comparison_snapshot_signature",
    }
)


def task21_hybrid_arm_specs() -> tuple[Mapping[str, object], ...]:
    return tuple(
        MappingProxyType(
            {
                "name": name,
                "direct_enabled": direct,
                "extended_enabled": extended,
                "reranker_calls": rerankers,
            }
        )
        for name, direct, extended, rerankers in _ARM_SPECS
    )


def build_task21_hybrid_report(*, cases, observations, freshness, backend_parity):
    indexed = validate_cases(cases)
    if (
        not isinstance(observations, Mapping)
        or tuple(observations) != TASK21_HYBRID_ARMS
    ):
        raise Task21HybridEvalError("observations require the exact five ordered arms")
    snapshots: dict[str, str] = {}
    arms: dict[str, object] = {}
    specs = {row[0]: row for row in _ARM_SPECS}
    for arm in TASK21_HYBRID_ARMS:
        rows = observations[arm]
        if not isinstance(rows, Sequence) or len(rows) != len(indexed):
            raise Task21HybridEvalError(f"{arm} observations are incomplete")
        scored = {}
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != _OBSERVATION_FIELDS:
                raise Task21HybridEvalError(f"{arm} observation fields are not exact")
            case_id = row["case_id"]
            if case_id not in indexed or case_id in scored:
                raise Task21HybridEvalError(f"{arm} case identity is invalid")
            signature = exact_sha(
                row["comparison_snapshot_signature"], "comparison snapshot"
            )
            if case_id in snapshots and snapshots[case_id] != signature:
                raise Task21HybridEvalError("comparison snapshot drifted across arms")
            snapshots[case_id] = signature
            projected = exact_ids(row["projected_ranks"], "projected ranks")
            repeated = exact_ids(
                row["repeated_projected_ranks"], "repeated projected ranks"
            )
            if projected != repeated:
                raise Task21HybridEvalError("deterministic projected ranks changed")
            if (
                type(row["reranker_calls"]) is not int
                or row["reranker_calls"] != specs[arm][3]
            ):
                raise Task21HybridEvalError(
                    "exactly one final reranker is allowed only for combined_reranked"
                )
            if arm == "vector_only" and (row["graph_chunk_ids"] or projected):
                raise Task21HybridEvalError("vector_only cannot contain graph material")
            scored[case_id] = score_case(indexed[case_id], row)
        ordered = [scored[case_id] for case_id in sorted(scored)]
        arms[arm] = {
            "policy": dict(task21_hybrid_arm_specs()[TASK21_HYBRID_ARMS.index(arm)]),
            "cases": scored,
            "metrics": {
                field: mean([float(item[field]) for item in ordered])
                for field in (
                    "recall_at_10",
                    "ndcg_at_10",
                    "graph_hit_rate",
                    "citation_evidence_coverage",
                    "seed_coverage",
                    "distance_2_novel_fraction",
                )
            }
            | {
                "latency_p95_ms": percentile_95(
                    [float(item["latency_ms"]) for item in ordered]
                ),
                "inaccessible_result_count": 0,
            },
        }
    return MappingProxyType(
        {
            "schema_version": "task21-hybrid-eval-v1",
            "arms": MappingProxyType(arms),
            "freshness": validate_freshness(freshness),
            "backend_parity": validate_parity(backend_parity),
            "comparison_snapshot_signatures": MappingProxyType(
                dict(sorted(snapshots.items()))
            ),
            "deterministic_projected_ranks": True,
            "permission_isolation": True,
        }
    )


def canonical_json_bytes(value: object) -> bytes:
    def thaw(item):
        if isinstance(item, Mapping):
            return {str(key): thaw(inner) for key, inner in item.items()}
        if isinstance(item, (tuple, list)):
            return [thaw(inner) for inner in item]
        return item

    return json.dumps(thaw(value), sort_keys=True, separators=(",", ":")).encode()
