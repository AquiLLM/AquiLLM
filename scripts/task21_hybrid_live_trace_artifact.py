"""Load and strictly validate a Task21 live-trace artifact."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


def _contract():
    package_root = Path(__file__).resolve().parents[1] / "aquillm"
    inserted = str(package_root) not in sys.path
    if inserted:
        sys.path.insert(0, str(package_root))
    try:
        return importlib.import_module(
            "apps.knowledge_graph.evals.task21_hybrid_live_trace"
        )
    finally:
        if inserted:
            sys.path.remove(str(package_root))


def _json_mapping(body: bytes, context: str):
    if type(body) is not bytes:
        raise ValueError(f"{context} bytes are invalid")
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} is unavailable or invalid") from error
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    return value


def validate_live_trace_artifact_bytes(
    body: bytes, *, run_id: str, source_commit: str, observations_body=None
) -> bytes:
    """Return canonical bytes after strict schema and identity validation."""
    payload = _json_mapping(body, "live trace artifact")
    contract = _contract()
    try:
        if observations_body is None:
            contract.validate_live_trace(payload)
        else:
            observations = _json_mapping(observations_body, "live trace observations")
            contract.validate_live_trace_observations(payload, observations)
        canonical = contract.canonical_live_trace_bytes(payload)
    except (TypeError, ValueError) as error:
        raise ValueError("live trace artifact failed strict validation") from error
    if body != canonical:
        raise ValueError("live trace artifact is not canonical")
    if payload["run_id"] != run_id or payload["source_commit"] != source_commit:
        raise ValueError("live trace identity differs from the attested run")
    return body


def validate_live_trace_artifact(
    path, *, run_id: str, source_commit: str, observations_path=None
) -> bytes:
    """Compatibility wrapper that snapshots each supplied path exactly once."""
    try:
        body = Path(path).read_bytes()
        observations_body = (
            None if observations_path is None else Path(observations_path).read_bytes()
        )
    except OSError as error:
        raise ValueError("live trace artifact is unavailable") from error
    return validate_live_trace_artifact_bytes(
        body,
        run_id=run_id,
        source_commit=source_commit,
        observations_body=observations_body,
    )
