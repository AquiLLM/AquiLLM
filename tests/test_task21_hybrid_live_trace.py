from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
from task21_hybrid_live_trace_support import ARMS, valid_trace

REPO = Path(__file__).resolve().parents[1]
ATTESTATION = REPO / "scripts" / "task21_hybrid_observation_attestation.py"


def _module():
    return importlib.import_module(
        "apps.knowledge_graph.evals.task21_hybrid_live_trace"
    )


def _attestation_module():
    spec = importlib.util.spec_from_file_location("task21_attestation", ATTESTATION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ATTESTATION.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_live_trace_writer_is_canonical_atomic_0600_and_never_overwrites(tmp_path):
    module = _module()
    destination = tmp_path / "live-trace.json"
    payload = valid_trace()

    written = module.write_live_trace(
        destination,
        payload,
        expected_case_ids=("inaccessible_collection_is_excluded",),
        fixture_chunk_ids=("public-token-001",),
    )

    assert written == destination
    assert destination.read_bytes() == module.canonical_live_trace_bytes(
        payload,
        expected_case_ids=("inaccessible_collection_is_excluded",),
        fixture_chunk_ids=("public-token-001",),
    )
    if os.name != "nt":
        assert destination.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        module.write_live_trace(destination, payload)


@pytest.mark.parametrize(
    "mutation",
    ("unknown_field", "noncanonical_score", "nonzero_inaccessible", "wrong_cases"),
)
def test_live_trace_rejects_open_or_unbound_claims(mutation):
    module = _module()
    payload = copy.deepcopy(valid_trace())
    if mutation == "unknown_field":
        payload["arms"]["direct"][0]["extra"] = True
    elif mutation == "noncanonical_score":
        payload["arms"]["direct"][0]["candidate_trace"][0][
            "direct_score_hex"
        ] = "0.5"
    elif mutation == "nonzero_inaccessible":
        payload["arms"]["direct"][0]["inaccessible_candidate_count"] = 1
    else:
        payload["arms"]["direct"][0]["case_id"] = "fabricated-case"

    with pytest.raises(ValueError):
        module.validate_live_trace(
            payload,
            expected_case_ids=("inaccessible_collection_is_excluded",),
            fixture_chunk_ids=("public-token-001",),
        )


@pytest.mark.parametrize("mutation", ("vector_direct", "missing_final_rerank"))
def test_live_trace_enforces_exact_arm_candidate_policy(mutation):
    module = _module()
    payload = valid_trace()
    if mutation == "vector_direct":
        candidate = payload["arms"]["vector_only"][0]["candidate_trace"][0]
        candidate["sources"].append("direct")
        candidate["direct_rank"] = 1
        candidate["direct_score_hex"] = 0.5.hex()
    else:
        payload["arms"]["combined_reranked"][0]["candidate_trace"][0][
            "reranker_rank"
        ] = None

    with pytest.raises(ValueError, match="arm|reranker"):
        module.validate_live_trace(payload)


def test_live_trace_timing_and_leakage_match_metric_observations():
    module = _module()
    payload = valid_trace()
    observations = {
        arm: [
            {
                "case_id": "inaccessible_collection_is_excluded",
                "latency_ms": 1.0,
                "inaccessible_result_chunk_ids": [],
                "ranked_chunk_ids": ["public-token-001"],
                "candidate_trace": copy.deepcopy(
                    payload["arms"][arm][0]["candidate_trace"]
                ),
            }
        ]
        for arm in ARMS
    }

    module.validate_live_trace_observations(payload, observations)
    observations["direct"][0]["latency_ms"] = 2.0
    with pytest.raises(ValueError, match="timing"):
        module.validate_live_trace_observations(payload, observations)


@pytest.mark.parametrize(
    ("arm", "field", "value"),
    (
        ("direct", "chunk_id", "other-token"),
        ("direct", "sources", ["baseline"]),
        ("direct", "direct_rank", 2),
        ("direct", "direct_score_hex", 0.25.hex()),
        ("combined_reranked", "reranker_rank", 2),
    ),
)
def test_live_trace_candidates_are_exactly_bound_to_observations(arm, field, value):
    module = _module()
    payload = valid_trace()
    observations = {
        name: [
            {
                "case_id": "inaccessible_collection_is_excluded",
                "latency_ms": 1.0,
                "inaccessible_result_chunk_ids": [],
                "ranked_chunk_ids": ["public-token-001"],
                "candidate_trace": copy.deepcopy(
                    payload["arms"][name][0]["candidate_trace"]
                ),
            }
        ]
        for name in ARMS
    }
    observations[arm][0]["candidate_trace"][0][field] = value

    with pytest.raises(ValueError, match="candidate"):
        module.validate_live_trace_observations(payload, observations)


def test_live_trace_total_timing_contains_every_stage():
    module = _module()
    payload = valid_trace()
    payload["arms"]["direct"][0]["timing_trace"]["candidate_ms"] = 1.1

    with pytest.raises(ValueError, match="total timing"):
        module.validate_live_trace(payload)


def test_live_trace_hash_binds_attestation_runtime_and_artifact_bytes(tmp_path):
    runtime = importlib.import_module("scripts.task21_hybrid_failure_bundle")
    attestation = _attestation_module()
    contract = _module()
    artifacts = {}
    observations = {
        arm: [
            {
                "case_id": "inaccessible_collection_is_excluded",
                "latency_ms": 1.0,
                "inaccessible_result_chunk_ids": [],
                "ranked_chunk_ids": ["public-token-001"],
                "candidate_trace": copy.deepcopy(
                    valid_trace()["arms"][arm][0]["candidate_trace"]
                ),
            }
        ]
        for arm in ARMS
    }
    bodies = {
        "observations": json.dumps(observations).encode() + b"\n",
        "freshness": json.dumps(
            {
                "generation_key": "a" * 64,
                "projection_checksum": "b" * 64,
                "age_seconds": 0,
                "max_age_seconds": 60,
            }
        ).encode(),
        "backend_parity": b'{"status":"exact"}\n',
        "live_trace": b'{"schema":"task21-hybrid-live-trace-v1"}\n',
    }
    for name, body in bodies.items():
        artifacts[name] = tmp_path / f"{name}.json"
        artifacts[name].write_bytes(body)
    captured = runtime.CapturedRuntime(
        members={},
        images={service: "sha256:" + "c" * 64 for service in runtime.SERVICES},
        config_sha256="d" * 64,
    )
    payload = {
        "schema": attestation.SCHEMA,
        "run_id": "e" * 32,
        "source_commit": "f" * 40,
        "config_sha256": "d" * 64,
        "images": captured.images,
        "projection_checksums": {
            "generation_key": "a" * 64,
            "projection_checksum": "b" * 64,
        },
        "artifact_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in artifacts.items()
        },
    }

    with pytest.raises(ValueError, match="live trace"):
        attestation.verify_attestation(
            payload=payload,
            captured=captured,
            artifacts=artifacts,
            run_id="e" * 32,
            source_commit="f" * 40,
        )
    trace_payload = valid_trace()
    trace_payload["run_id"] = "e" * 32
    trace_payload["source_commit"] = "f" * 40
    artifacts["live_trace"].write_bytes(
        contract.canonical_live_trace_bytes(trace_payload)
    )
    payload["artifact_sha256"]["live_trace"] = hashlib.sha256(
        artifacts["live_trace"].read_bytes()
    ).hexdigest()
    attestation.verify_attestation(
        payload=payload,
        captured=captured,
        artifacts=artifacts,
        run_id="e" * 32,
        source_commit="f" * 40,
    )
    artifacts["live_trace"].write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="artifact bytes"):
        attestation.verify_attestation(
            payload=payload,
            captured=captured,
            artifacts=artifacts,
            run_id="e" * 32,
            source_commit="f" * 40,
        )
    artifacts["live_trace"].write_bytes(
        contract.canonical_live_trace_bytes(trace_payload)
    )
    with pytest.raises(ValueError, match="runtime capture"):
        attestation.verify_attestation(
            payload=payload,
            captured=runtime.CapturedRuntime(
                captured.members,
                captured.images,
                captured.config_sha256,
                complete=False,
            ),
            artifacts=artifacts,
            run_id="e" * 32,
            source_commit="f" * 40,
        )
    payload["run_id"] = "0" * 32
    with pytest.raises(ValueError, match="run identity"):
        attestation.verify_attestation(
            payload=payload,
            captured=captured,
            artifacts=artifacts,
            run_id="e" * 32,
            source_commit="f" * 40,
        )
