#!/usr/bin/env python3
"""Verify that live Task21 observations came from the captured cloud runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

try:
    from scripts import task21_hybrid_failure_bundle as _runtime
except ImportError:
    import task21_hybrid_failure_bundle as _runtime

SCHEMA = "task21-hybrid-live-observation-v1"
_HEX32 = re.compile(r"[0-9a-f]{32}")
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_ARTIFACTS = ("observations", "freshness", "backend_parity")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_attestation(*, payload, captured, artifacts, run_id, source_commit) -> None:
    if set(payload) != {
        "schema",
        "run_id",
        "source_commit",
        "config_sha256",
        "images",
        "projection_checksums",
        "artifact_sha256",
    }:
        raise ValueError("live observation attestation fields are not exact")
    if payload["schema"] != SCHEMA:
        raise ValueError("live observation attestation schema is invalid")
    if _HEX32.fullmatch(run_id) is None or payload["run_id"] != run_id:
        raise ValueError("live observation run identity changed")
    if (
        _HEX40.fullmatch(source_commit) is None
        or payload["source_commit"] != source_commit
    ):
        raise ValueError("live observation source identity changed")
    if payload["config_sha256"] != captured.config_sha256:
        raise ValueError("live observation Compose configuration changed")
    if set(captured.images) != set(_runtime.SERVICES):
        raise ValueError("live observation runtime services are incomplete")
    if payload["images"] != captured.images:
        raise ValueError("live observation image digests changed")
    if set(artifacts) != set(_ARTIFACTS):
        raise ValueError("live observation artifacts are not exact")
    expected_hashes = {name: _sha256(Path(artifacts[name])) for name in _ARTIFACTS}
    if payload["artifact_sha256"] != expected_hashes:
        raise ValueError("live observation artifact bytes changed")
    freshness = json.loads(Path(artifacts["freshness"]).read_text(encoding="utf-8"))
    projections = {
        name: freshness.get(name) for name in ("generation_key", "projection_checksum")
    }
    if any(_HEX64.fullmatch(value or "") is None for value in projections.values()):
        raise ValueError("live projection identity is invalid")
    if payload["projection_checksums"] != projections:
        raise ValueError("live projection identity changed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--compose-file", action="append", required=True)
    parser.add_argument("--profile", action="append", default=[])
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--freshness", type=Path, required=True)
    parser.add_argument("--backend-parity", type=Path, required=True)
    args = parser.parse_args()
    prefix = (
        "docker",
        "compose",
        "--env-file",
        str(args.env_file.resolve()),
        "--project-name",
        f"aquillm-task21-{args.run_id}",
    )
    for compose_file in args.compose_file:
        prefix += ("--file", str(Path(compose_file).resolve()))
    for profile in args.profile:
        prefix += ("--profile", profile)
    captured = _runtime.capture_runtime(_runtime.CommandRunner(), prefix)
    payload = json.loads(args.attestation.read_text(encoding="utf-8"))
    verify_attestation(
        payload=payload,
        captured=captured,
        artifacts={
            "observations": args.observations,
            "freshness": args.freshness,
            "backend_parity": args.backend_parity,
        },
        run_id=args.run_id,
        source_commit=args.source_commit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
