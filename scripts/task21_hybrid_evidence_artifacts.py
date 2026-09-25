"""Immutable artifact snapshots and semantic evidence validation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

OBSERVATION_ATTESTATION = "runtime/observation-attestation.json"
_ATTESTATION_SCHEMA = "task21-hybrid-live-observation-v1"
_HEX32 = re.compile(r"[0-9a-f]{32}")
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_OBSERVATION_ARTIFACTS = {
    "observations",
    "freshness",
    "backend_parity",
    "live_trace",
}


def snapshot_artifacts(artifacts) -> dict[str, bytes]:
    snapshots = {}
    try:
        for name, path in artifacts.items():
            snapshots[name] = Path(path).read_bytes()
    except OSError as error:
        raise ValueError(f"{name} artifact is unavailable") from error
    return snapshots


def _json_mapping(body: bytes, context: str) -> dict[str, object]:
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} is invalid") from error
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    return value


def projection_checksums(body: bytes) -> dict[str, str]:
    value = _json_mapping(body, "projection checksums")
    if set(value) != {"generation_key", "projection_checksum"} or not all(
        type(key) is str and type(item) is str and _HEX64.fullmatch(item)
        for key, item in value.items()
    ):
        raise ValueError("projection checksums must be a nonempty SHA-256 mapping")
    return dict(sorted(value.items()))


def validate_timings(body: bytes) -> None:
    value = _json_mapping(body, "timings artifact")
    expected = {
        "elapsed_ms",
        "finished_ns",
        "original_exit_code",
        "started_ns",
        "status",
    }
    if set(value) != expected:
        raise ValueError("timings artifact fields are not exact")
    elapsed = value["elapsed_ms"]
    started = value["started_ns"]
    finished = value["finished_ns"]
    original = value["original_exit_code"]
    if (
        type(elapsed) not in (int, float)
        or not math.isfinite(float(elapsed))
        or elapsed < 0
        or type(started) is not int
        or type(finished) is not int
        or started < 0
        or finished < started
        or type(original) is not int
        or float(elapsed) != (finished - started) / 1_000_000
    ):
        raise ValueError("timings artifact values are invalid or clock-inconsistent")
    status = value["status"]
    if status not in {"passed", "failed"} or (status == "passed") != (original == 0):
        raise ValueError("timings artifact status is invalid")


def validate_observation_attestation(
    body: bytes,
    *,
    run_id: str,
    source_commit: str,
    config_sha256: str,
    images,
    projections,
    live_trace: bytes,
) -> None:
    value = _json_mapping(body, "observation attestation")
    fields = {
        "schema",
        "run_id",
        "source_commit",
        "config_sha256",
        "images",
        "projection_checksums",
        "artifact_sha256",
    }
    if set(value) != fields or value["schema"] != _ATTESTATION_SCHEMA:
        raise ValueError("observation attestation fields or schema are invalid")
    if (
        _HEX32.fullmatch(run_id) is None
        or value["run_id"] != run_id
        or _HEX40.fullmatch(source_commit) is None
        or value["source_commit"] != source_commit
    ):
        raise ValueError("observation attestation identity changed")
    if value["config_sha256"] != config_sha256 or value["images"] != images:
        raise ValueError("observation attestation runtime changed")
    if value["projection_checksums"] != projections:
        raise ValueError("observation attestation projection identity changed")
    hashes = value["artifact_sha256"]
    if (
        not isinstance(hashes, dict)
        or set(hashes) != _OBSERVATION_ARTIFACTS
        or any(
            type(item) is not str or _HEX64.fullmatch(item) is None
            for item in hashes.values()
        )
        or hashes["live_trace"] != hashlib.sha256(live_trace).hexdigest()
    ):
        raise ValueError("observation attestation artifact hashes are invalid")
