from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path

from task21_hybrid_live_trace_support import (
    ARMS,
    valid_observation_attestation,
    valid_trace,
)

REPO = Path(__file__).resolve().parents[1]


def _script(name):
    path = REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _trace(run_id="d" * 32, source_commit="e" * 40):
    contract = importlib.import_module(
        "apps.knowledge_graph.evals.task21_hybrid_live_trace"
    )
    payload = valid_trace()
    payload["run_id"] = run_id
    payload["source_commit"] = source_commit
    return payload, contract.canonical_live_trace_bytes(payload)


def _observations(trace):
    return {
        arm: [
            {
                "case_id": "inaccessible_collection_is_excluded",
                "ranked_chunk_ids": ["public-token-001"],
                "candidate_trace": trace["arms"][arm][0]["candidate_trace"],
                "latency_ms": 1.0,
                "inaccessible_result_chunk_ids": [],
            }
        ]
        for arm in ARMS
    }


def _publisher_inputs(module, tmp_path):
    trace, trace_body = _trace()
    bodies = {
        "arm_results": b'{"arms":{}}\n',
        "timings": json.dumps(
            {
                "elapsed_ms": 0.000001,
                "finished_ns": 2,
                "original_exit_code": 0,
                "started_ns": 1,
                "status": "passed",
            }
        ).encode(),
        "projection_checksums": json.dumps(
            {"generation_key": "a" * 64, "projection_checksum": "b" * 64}
        ).encode(),
        "live_trace": trace_body,
    }
    artifacts = {}
    for name, body in bodies.items():
        artifacts[name] = tmp_path / f"{name}.json"
        artifacts[name].write_bytes(body)
    projections = {"generation_key": "a" * 64, "projection_checksum": "b" * 64}
    attestation = valid_observation_attestation(
        trace_body,
        run_id="d" * 32,
        source_commit="e" * 40,
        config_sha256="c" * 64,
        images={},
        projections=projections,
    )
    return trace, bodies, {
        "output_root": tmp_path / "evidence",
        "run_id": "d" * 32,
        "captured": module.CapturedRuntime(
            {
                "runtime/observation-attestation.json": json.dumps(
                    attestation
                ).encode()
            },
            {},
            "c" * 64,
        ),
        "artifacts": artifacts,
        "cleanup_proof": {"complete": True},
        "source": {"commit": "e" * 40, "clean": True},
        "expected_source_commit": "e" * 40,
        "claim_scope": "cloud",
        "signing_key": b"task23-signing-key",
        "signing_key_version": "task23-v3",
    }


def test_publisher_copies_and_signs_one_immutable_artifact_snapshot(
    tmp_path, monkeypatch
):
    module = _script("task21_hybrid_failure_bundle")
    _trace_payload, bodies, arguments = _publisher_inputs(module, tmp_path)
    publisher = module._publisher
    original = publisher._validate_timings

    def mutate_after_snapshot(value):
        arguments["artifacts"]["projection_checksums"].write_text(
            json.dumps({"generation_key": "f" * 64}), encoding="utf-8"
        )
        return original(value)

    monkeypatch.setattr(publisher, "_validate_timings", mutate_after_snapshot)
    destination = module.publish_bundle(**arguments)
    copied = destination / "projection" / "checksums.json"
    manifest = json.loads((destination / "bundle.json").read_text(encoding="utf-8"))

    assert copied.read_bytes() == bodies["projection_checksums"]
    assert manifest["projection_checksums"] == {
        "generation_key": "a" * 64,
        "projection_checksum": "b" * 64,
    }


def test_publisher_cannot_copy_live_trace_mutated_after_validation(
    tmp_path, monkeypatch
):
    module = _script("task21_hybrid_failure_bundle")
    _trace_payload, bodies, arguments = _publisher_inputs(module, tmp_path)
    publisher = module._publisher
    original = publisher.validate_live_trace_artifact_bytes

    def mutate_after_validation(value, **identity):
        result = original(value, **identity)
        arguments["artifacts"]["live_trace"].write_bytes(b'{"invalid":true}\n')
        return result

    monkeypatch.setattr(
        publisher, "validate_live_trace_artifact_bytes", mutate_after_validation
    )
    destination = module.publish_bundle(**arguments)

    assert (destination / "trace" / "live-trace.json").read_bytes() == bodies[
        "live_trace"
    ]


def test_publisher_semantically_validates_copied_observation_attestation(tmp_path):
    module = _script("task21_hybrid_failure_bundle")
    _trace_payload, _bodies, arguments = _publisher_inputs(module, tmp_path)
    arguments["captured"].members[
        "runtime/observation-attestation.json"
    ] = b'{"schema":"untrusted"}\n'

    try:
        module.publish_bundle(**arguments)
    except ValueError as error:
        assert "observation attestation" in str(error)
    else:
        raise AssertionError("invalid observation attestation was published")


def test_publisher_requires_exact_projection_checksum_semantics(tmp_path):
    module = _script("task21_hybrid_failure_bundle")
    _trace_payload, _bodies, arguments = _publisher_inputs(module, tmp_path)
    incomplete = {"generation_key": "a" * 64}
    arguments["artifacts"]["projection_checksums"].write_text(
        json.dumps(incomplete), encoding="utf-8"
    )
    attestation_path = "runtime/observation-attestation.json"
    attestation = json.loads(arguments["captured"].members[attestation_path])
    attestation["projection_checksums"] = incomplete
    arguments["captured"].members[attestation_path] = json.dumps(attestation).encode()

    try:
        module.publish_bundle(**arguments)
    except ValueError as error:
        assert "projection checksums" in str(error)
    else:
        raise AssertionError("incomplete projection checksums were published")


def test_attestation_reads_each_artifact_once_and_validates_same_snapshot(
    tmp_path, monkeypatch
):
    runtime = _script("task21_hybrid_failure_bundle")
    attestation = _script("task21_hybrid_observation_attestation")
    trace, trace_body = _trace(run_id="e" * 32, source_commit="f" * 40)
    bodies = {
        "observations": json.dumps(_observations(trace)).encode() + b"\n",
        "freshness": json.dumps(
            {
                "generation_key": "a" * 64,
                "projection_checksum": "b" * 64,
                "age_seconds": 0,
                "max_age_seconds": 60,
            }
        ).encode(),
        "backend_parity": b'{"status":"exact"}\n',
        "live_trace": trace_body,
    }
    artifacts = {}
    for name, body in bodies.items():
        artifacts[name] = tmp_path / f"{name}.json"
        artifacts[name].write_bytes(body)
    reads = {path.resolve(): 0 for path in artifacts.values()}
    original_read_bytes = Path.read_bytes
    original_read_text = Path.read_text

    def counted_read_bytes(path):
        resolved = path.resolve()
        if resolved in reads:
            reads[resolved] += 1
        return original_read_bytes(path)

    def reject_read_text(path, *args, **kwargs):
        if path.resolve() in reads:
            raise AssertionError("artifact path was reopened after snapshot")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)
    monkeypatch.setattr(Path, "read_text", reject_read_text)
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
            name: hashlib.sha256(body).hexdigest() for name, body in bodies.items()
        },
    }

    attestation.verify_attestation(
        payload=payload,
        captured=captured,
        artifacts=artifacts,
        run_id="e" * 32,
        source_commit="f" * 40,
    )

    assert set(reads.values()) == {1}
