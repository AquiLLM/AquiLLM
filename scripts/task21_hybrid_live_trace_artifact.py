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


def _json_mapping(path: Path, context: str):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} is unavailable or invalid") from error
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    return value


def validate_live_trace_artifact(
    path, *, run_id: str, source_commit: str, observations_path=None
) -> bytes:
    """Return canonical bytes after strict schema and identity validation."""
    trace_path = Path(path)
    payload = _json_mapping(trace_path, "live trace artifact")
    contract = _contract()
    try:
        if observations_path is None:
            contract.validate_live_trace(payload)
        else:
            observations = _json_mapping(
                Path(observations_path), "live trace observations"
            )
            contract.validate_live_trace_observations(payload, observations)
        canonical = contract.canonical_live_trace_bytes(payload)
    except (TypeError, ValueError) as error:
        raise ValueError("live trace artifact failed strict validation") from error
    try:
        body = trace_path.read_bytes()
    except OSError as error:
        raise ValueError("live trace artifact is unavailable") from error
    if body != canonical:
        raise ValueError("live trace artifact is not canonical")
    if payload["run_id"] != run_id or payload["source_commit"] != source_commit:
        raise ValueError("live trace identity differs from the attested run")
    return body
