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

ARMS = (
    "vector_only",
    "direct",
    "extended",
    "combined",
    "combined_reranked",
)
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


def _candidate(arm):
    direct = arm in {"direct", "combined", "combined_reranked"}
    extended = arm in {"extended", "combined", "combined_reranked"}
    sources = ["baseline"]
    if direct:
        sources.append("direct")
    if extended:
        sources.append("extended")
    return {
        "chunk_id": "public-token-001",
        "ordinal": 1,
        "sources": sources,
        "baseline_rank": 1,
        "direct_rank": 1 if direct else None,
        "direct_score_hex": 0.5.hex() if direct else None,
        "extended_rank": 1 if extended else None,
        "extended_score_hex": 0.25.hex() if extended else None,
        "fusion_score_hex": 0.75.hex(),
        "reranker_rank": 1 if arm == "combined_reranked" else None,
    }


def valid_trace():
    return {
        "schema": "task21-hybrid-live-trace-v1",
        "run_id": "a" * 32,
        "source_commit": "b" * 40,
        "fixture_id": "kg-task20-synthetic-v1",
        "fixture_checksum": "c" * 64,
        "manifest_checksum": "d" * 64,
        "arms": {
            arm: [
                {
                    "case_id": "inaccessible_collection_is_excluded",
                    "candidate_trace": [_candidate(arm)],
                    "timing_trace": {
                        "candidate_ms": 0.2,
                        "branch_ms": 0.2,
                        "fusion_ms": 0.2,
                        "rerank_ms": 0.2,
                        "total_ms": 1.0,
                    },
                    "authorization_status": "current",
                    "graph_scheduled": arm != "vector_only",
                    "inaccessible_candidate_count": 0,
                }
            ]
            for arm in ARMS
        },
        "freshness_attestation": {
            "projection_keys": ["e" * 64],
            "generation_keys": ["f" * 64],
            "graph_checksums": ["1" * 64],
            "ready_bundle_checksums": ["2" * 64],
            "ontology_version": "ontology-v1",
            "ontology_checksum": "3" * 64,
        },
        "backend_parity_inputs": [
            {
                "branch": branch,
                "ready_bundle_checksum": "4" * 64,
                "seed_checksum": "5" * 64,
                "seed_count": 1,
                "max_depth": depth,
                "max_nodes": 20,
                "max_edges": 40,
                "max_results": 10,
                "projection_keys": ["6" * 64],
                "generation_keys": ["7" * 64],
                "authorized_document_keys": [character * 64],
            }
            for branch, depth, character in (
                ("direct", 1, "8"),
                ("extended", 2, "9"),
            )
        ],
    }


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


def test_live_trace_timing_and_leakage_match_metric_observations():
    module = _module()
    observations = {
        arm: [
            {
                "case_id": "inaccessible_collection_is_excluded",
                "latency_ms": 1.0,
                "inaccessible_result_chunk_ids": [],
            }
        ]
        for arm in ARMS
    }

    module.validate_live_trace_observations(valid_trace(), observations)
    observations["direct"][0]["latency_ms"] = 2.0
    with pytest.raises(ValueError, match="timing"):
        module.validate_live_trace_observations(valid_trace(), observations)


def test_live_trace_hash_binds_attestation_runtime_and_artifact_bytes(tmp_path):
    runtime = importlib.import_module("scripts.task21_hybrid_failure_bundle")
    attestation = _attestation_module()
    artifacts = {}
    bodies = {
        "observations": b'{"combined":[]}\n',
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
    artifacts["live_trace"].write_bytes(b'{"schema":"task21-hybrid-live-trace-v1"}\n')
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
