"""Generate Task21 five-arm observations from live production retrieval seams."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path

from .task21_hybrid_eval import TASK21_HYBRID_ARMS, task21_hybrid_arm_specs
from .task21_hybrid_live_evidence import (
    RUNTIME_IDENTITY_SCHEMA,
    RUNTIME_SERVICES,
    LiveOutputPaths,
    publish_live_artifacts,
)


def negative_observation(
    *, case_id: str, arm: str, adversarial_chunk_ids: tuple[str, ...]
) -> dict[str, object]:
    if arm not in TASK21_HYBRID_ARMS:
        raise ValueError("arm is not frozen")
    if (
        type(case_id) is not str
        or not case_id
        or type(adversarial_chunk_ids) is not tuple
        or any(type(value) is not str or not value for value in adversarial_chunk_ids)
        or len(set(adversarial_chunk_ids)) != len(adversarial_chunk_ids)
    ):
        raise ValueError("negative observation identity is invalid")
    from hashlib import sha256

    signature = sha256(f"task21-negative\0{case_id}".encode()).hexdigest()
    return {
        "case_id": case_id,
        "ranked_chunk_ids": [],
        "graph_chunk_ids": [],
        "citation_evidence_chunk_ids": [],
        "seed_chunk_ids": [],
        "mapped_seed_chunk_ids": [],
        "projected_ranks": [],
        "repeated_projected_ranks": [],
        "latency_ms": 0.0,
        "reranker_calls": 0,
        "comparison_snapshot_signature": signature,
        "candidate_trace": [],
        "timing_trace": {
            "candidate_ms": 0.0,
            "branch_ms": 0.0,
            "fusion_ms": 0.0,
            "rerank_ms": 0.0,
            "total_ms": 0.0,
        },
        "authorization_status": "absent_negative",
        "graph_scheduled": False,
        "inaccessible_candidate_count": 0,
        "adversarial_candidate_chunk_ids": list(adversarial_chunk_ids),
        "inaccessible_result_chunk_ids": [],
    }


def generate_case_arms(*, case: Mapping[str, object], prepared, executor):
    case_id = case.get("id") if isinstance(case, Mapping) else None
    if type(case_id) is not str or not case_id:
        raise ValueError("case identity is invalid")
    result = {arm: [] for arm in TASK21_HYBRID_ARMS}
    if prepared is None:
        adversarial = tuple(case.get("adversarial_chunk_ids", ()))
        for arm in TASK21_HYBRID_ARMS:
            result[arm].append(
                negative_observation(
                    case_id=case_id,
                    arm=arm,
                    adversarial_chunk_ids=adversarial,
                )
            )
        return result
    for spec in task21_hybrid_arm_specs():
        arm = spec["name"]
        result[arm].append(executor.run_arm(case=case, prepared=prepared, spec=spec))
    return result


def _absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("live generator paths must be absolute")
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--fixture-manifest", required=True)
    parser.add_argument("--runtime-identity", required=True)
    parser.add_argument("--observations-output", required=True)
    parser.add_argument("--freshness-output", required=True)
    parser.add_argument("--backend-parity-output", required=True)
    parser.add_argument("--attestation-output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "aquillm.settings")
    import django

    django.setup()
    from .task21_hybrid_live_runner import run_production_live_observations

    run_production_live_observations(
        run_id=args.run_id,
        source_commit=args.source_commit,
        fixture_manifest=_absolute(args.fixture_manifest),
        runtime_identity=json.loads(
            _absolute(args.runtime_identity).read_text(encoding="utf-8")
        ),
        output_paths=LiveOutputPaths(
            _absolute(args.observations_output),
            _absolute(args.freshness_output),
            _absolute(args.backend_parity_output),
            _absolute(args.attestation_output),
        ),
    )
    return 0


__all__ = [
    "LiveOutputPaths",
    "RUNTIME_IDENTITY_SCHEMA",
    "RUNTIME_SERVICES",
    "TASK21_HYBRID_ARMS",
    "generate_case_arms",
    "negative_observation",
    "publish_live_artifacts",
]


if __name__ == "__main__":
    raise SystemExit(main())
