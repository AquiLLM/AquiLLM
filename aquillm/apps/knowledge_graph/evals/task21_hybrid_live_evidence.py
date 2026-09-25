"""Private, no-overwrite publication for Task21 live observation artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

ATTESTATION_SCHEMA = "task21-hybrid-live-observation-v1"
RUNTIME_IDENTITY_SCHEMA = "task21-hybrid-runtime-identity-v1"
RUNTIME_SERVICES = (
    "web",
    "db",
    "redis",
    "memgraph_knowledge_graph",
    "knowledge_graph_query_extractor",
    "worker_knowledge_graph_projection",
    "worker_knowledge_graph",
    "vllm_embed",
    "vllm_rerank",
)
_HEX32 = re.compile(r"[0-9a-f]{32}")
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_ARTIFACTS = ("observations", "freshness", "backend_parity", "live_trace")


@dataclass(frozen=True, slots=True)
class LiveOutputPaths:
    observations: Path
    freshness: Path
    backend_parity: Path
    live_trace: Path
    attestation: Path

    def __post_init__(self) -> None:
        values = tuple(self)
        if any(not isinstance(path, Path) or not path.is_absolute() for path in values):
            raise ValueError("live output paths must be exact absolute Paths")
        if len(set(values)) != 5 or len({path.parent for path in values}) != 1:
            raise ValueError("live output paths must be unique siblings")

    def __iter__(self) -> Iterator[Path]:
        yield self.observations
        yield self.freshness
        yield self.backend_parity
        yield self.live_trace
        yield self.attestation


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _identity(value: Mapping[str, object], *, run_id: str, source_commit: str):
    if set(value) != {
        "schema",
        "run_id",
        "source_commit",
        "config_sha256",
        "images",
        "complete",
    }:
        raise ValueError("runtime identity fields are not exact")
    if (
        value["schema"] != RUNTIME_IDENTITY_SCHEMA
        or _HEX32.fullmatch(run_id) is None
        or value["run_id"] != run_id
        or _HEX40.fullmatch(source_commit) is None
        or value["source_commit"] != source_commit
        or type(value["config_sha256"]) is not str
        or _HEX64.fullmatch(value["config_sha256"]) is None
        or value["complete"] is not True
    ):
        raise ValueError("runtime/source identity is invalid")
    images = value["images"]
    if (
        type(images) is not dict
        or tuple(images) != RUNTIME_SERVICES
        or any(
            type(digest) is not str
            or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
            for digest in images.values()
        )
    ):
        raise ValueError("runtime image identity is incomplete")
    return value["config_sha256"], images


def _private_temp(path: Path, payload: object) -> Path:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        return temporary
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def _publish_no_replace(temporary: Path, destination: Path) -> None:
    os.link(temporary, destination)
    temporary.unlink()
    os.chmod(destination, 0o600)


def publish_live_artifacts(
    *,
    paths: LiveOutputPaths,
    run_id: str,
    source_commit: str,
    runtime_identity: Mapping[str, object],
    observations: object,
    freshness: Mapping[str, object],
    backend_parity: object,
    live_trace: Mapping[str, object],
    expected_case_ids=(),
    fixture_chunk_ids=(),
) -> LiveOutputPaths:
    """Publish four artifacts then their binding attestation, never overwriting."""

    from .task21_hybrid_live_trace import (
        validate_live_trace_observations,
        write_live_trace,
    )

    if type(paths) is not LiveOutputPaths:
        raise TypeError("paths must be exact")
    config_sha256, images = _identity(
        runtime_identity, run_id=run_id, source_commit=source_commit
    )
    if any(path.exists() for path in paths):
        raise FileExistsError("live evidence destination already exists")
    parent = paths.observations.parent
    if not parent.is_dir():
        raise ValueError("live evidence parent must already exist")
    payloads = {
        "observations": observations,
        "freshness": freshness,
        "backend_parity": backend_parity,
        "live_trace": live_trace,
    }
    temporaries: dict[str, Path] = {}
    published: list[Path] = []
    try:
        for name in _ARTIFACTS[:-1]:
            temporaries[name] = _private_temp(getattr(paths, name), payloads[name])
        validate_live_trace_observations(live_trace, observations)
        trace_stage = paths.live_trace.with_name(
            f".{paths.live_trace.name}.{os.getpid()}.stage"
        )
        temporaries["live_trace"] = write_live_trace(
            trace_stage,
            live_trace,
            expected_case_ids=expected_case_ids,
            fixture_chunk_ids=fixture_chunk_ids,
        )
        hashes = {
            name: hashlib.sha256(temporaries[name].read_bytes()).hexdigest()
            for name in _ARTIFACTS
        }
        projection_checksums = {
            name: freshness[name]
            for name in ("generation_key", "projection_checksum")
        }
        attestation = {
            "schema": ATTESTATION_SCHEMA,
            "run_id": run_id,
            "source_commit": source_commit,
            "config_sha256": config_sha256,
            "images": images,
            "projection_checksums": projection_checksums,
            "artifact_sha256": hashes,
        }
        temporaries["attestation"] = _private_temp(paths.attestation, attestation)
        for name in (*_ARTIFACTS, "attestation"):
            destination = getattr(paths, name)
            _publish_no_replace(temporaries[name], destination)
            published.append(destination)
    except BaseException:
        for temporary in temporaries.values():
            temporary.unlink(missing_ok=True)
        for destination in published:
            destination.unlink(missing_ok=True)
        raise
    return paths


__all__ = [
    "ATTESTATION_SCHEMA",
    "LiveOutputPaths",
    "RUNTIME_IDENTITY_SCHEMA",
    "RUNTIME_SERVICES",
    "canonical_json_bytes",
    "publish_live_artifacts",
]
