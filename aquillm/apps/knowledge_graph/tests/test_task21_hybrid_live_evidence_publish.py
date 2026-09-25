from __future__ import annotations

import hashlib
import json
import os

import pytest

from apps.knowledge_graph.evals import task21_hybrid_live_observations as live
from apps.knowledge_graph.tests.test_task21_hybrid_live_payloads import _payloads


def test_atomic_live_publish_is_0600_attested_and_never_overwrites(tmp_path):
    paths = live.LiveOutputPaths(
        observations=tmp_path / "observations.json",
        freshness=tmp_path / "freshness.json",
        backend_parity=tmp_path / "backend-parity.json",
        live_trace=tmp_path / "live-trace.json",
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
    evidence = _payloads()
    artifacts = live.publish_live_artifacts(
        paths=paths,
        run_id="1" * 32,
        source_commit="2" * 40,
        runtime_identity=runtime,
        observations=evidence.observations,
        freshness=evidence.freshness,
        backend_parity=evidence.backend_parity,
        live_trace=evidence.live_trace,
        expected_case_ids=("visible-case",),
        fixture_chunk_ids=("visible-chunk",),
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
        for name in ("observations", "freshness", "backend_parity", "live_trace")
    }
    with pytest.raises(FileExistsError):
        live.publish_live_artifacts(
            paths=paths,
            run_id="1" * 32,
            source_commit="2" * 40,
            runtime_identity=runtime,
            observations=evidence.observations,
            freshness=json.loads(paths.freshness.read_text(encoding="utf-8")),
            backend_parity=evidence.backend_parity,
            live_trace=evidence.live_trace,
            expected_case_ids=("visible-case",),
            fixture_chunk_ids=("visible-chunk",),
        )


def test_cli_requires_an_absolute_live_trace_output():
    parsed = live._parser().parse_args(
        [
            "--run-id",
            "1" * 32,
            "--source-commit",
            "2" * 40,
            "--fixture-manifest",
            "C:/fixture.json",
            "--runtime-identity",
            "C:/runtime.json",
            "--observations-output",
            "C:/observations.json",
            "--freshness-output",
            "C:/freshness.json",
            "--backend-parity-output",
            "C:/parity.json",
            "--live-trace-output",
            "C:/live-trace.json",
            "--attestation-output",
            "C:/attestation.json",
        ]
    )

    assert parsed.live_trace_output == "C:/live-trace.json"
