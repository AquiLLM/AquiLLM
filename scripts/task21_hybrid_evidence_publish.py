"""Atomic, signed publisher for Task21 hybrid cloud evidence."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath

try:
    from scripts.task21_hybrid_evidence_artifacts import (
        OBSERVATION_ATTESTATION,
        validate_observation_attestation,
    )
    from scripts.task21_hybrid_evidence_artifacts import (
        projection_checksums as _projection_checksums,
    )
    from scripts.task21_hybrid_evidence_artifacts import (
        snapshot_artifacts as _snapshot_artifacts,
    )
    from scripts.task21_hybrid_evidence_artifacts import (
        validate_timings as _validate_timings,
    )
    from scripts.task21_hybrid_live_trace_artifact import (
        validate_live_trace_artifact_bytes,
    )
except ImportError:
    from task21_hybrid_evidence_artifacts import (
        OBSERVATION_ATTESTATION,
        validate_observation_attestation,
    )
    from task21_hybrid_evidence_artifacts import (
        projection_checksums as _projection_checksums,
    )
    from task21_hybrid_evidence_artifacts import (
        snapshot_artifacts as _snapshot_artifacts,
    )
    from task21_hybrid_evidence_artifacts import (
        validate_timings as _validate_timings,
    )
    from task21_hybrid_live_trace_artifact import validate_live_trace_artifact_bytes

SCHEMA = "task21-hybrid-cloud-evidence-v1"
_HEX32 = re.compile(r"[0-9a-f]{32}")
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_SAFE_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _safe_member_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("evidence member path is unsafe")
    return path


def _write_member(root: Path, relative: str, body: bytes) -> Path:
    path = _safe_member_path(relative)
    destination = root.joinpath(*path.parts)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with destination.open("xb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(destination, 0o600)
    return destination


def _validate_publish_inputs(
    *,
    run_id,
    artifacts,
    source,
    expected_source_commit,
    claim_scope,
    signing_key,
    signing_key_version,
) -> None:
    if _HEX32.fullmatch(run_id) is None:
        raise ValueError("run id must be 32 lowercase hexadecimal characters")
    if set(artifacts) != {
        "arm_results",
        "timings",
        "projection_checksums",
        "live_trace",
    }:
        raise ValueError("evidence artifacts are not exact")
    if (
        set(source) != {"commit", "clean"}
        or _HEX40.fullmatch(source["commit"]) is None
        or type(source["clean"]) is not bool
    ):
        raise ValueError("source identity is invalid")
    if _HEX40.fullmatch(expected_source_commit) is None:
        raise ValueError("expected source commit is invalid")
    if source["commit"] != expected_source_commit:
        raise ValueError("source commit changed after observation attestation")
    if claim_scope not in {"cloud", "local-nonacceptance"}:
        raise ValueError("claim scope is invalid")
    if type(signing_key) is not bytes or len(signing_key) < 16:
        raise ValueError("evidence signing key is invalid")
    if _SAFE_VERSION.fullmatch(signing_key_version) is None:
        raise ValueError("signing key version is invalid")


def _copy_members(staging: Path, captured_members, artifacts) -> dict[str, str]:
    roles: dict[str, str] = {}
    for relative, body in captured_members.items():
        _write_member(staging, relative, body)
        roles[relative] = (
            "service_log" if relative.startswith("logs/") else "runtime_capture"
        )
    destinations = {
        "arm_results": "results/arms.json",
        "timings": "results/timings.json",
        "projection_checksums": "projection/checksums.json",
        "live_trace": "trace/live-trace.json",
    }
    for role, relative in destinations.items():
        _write_member(staging, relative, artifacts[role])
        roles[relative] = role
    return roles


def _member_manifest(staging: Path, roles: dict[str, str]) -> list[dict[str, object]]:
    members = []
    for relative in sorted(roles):
        path = staging.joinpath(*PurePosixPath(relative).parts)
        body = path.read_bytes()
        members.append(
            {
                "path": relative,
                "role": roles[relative],
                "size": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
            }
        )
    return members


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    directory = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _fsync_tree(root: Path) -> None:
    if os.name == "nt":
        return
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for directory in sorted(
        directories, key=lambda path: len(path.parts), reverse=True
    ):
        _fsync_directory(directory)
    _fsync_directory(root)


def publish_bundle(
    *,
    output_root,
    run_id,
    captured,
    artifacts,
    cleanup_proof,
    source,
    expected_source_commit,
    claim_scope,
    signing_key,
    signing_key_version,
) -> Path:
    _validate_publish_inputs(
        run_id=run_id,
        artifacts=artifacts,
        source=source,
        expected_source_commit=expected_source_commit,
        claim_scope=claim_scope,
        signing_key=signing_key,
        signing_key_version=signing_key_version,
    )
    snapshots = _snapshot_artifacts(artifacts)
    captured_members = dict(captured.members)
    captured_images = dict(captured.images)
    captured_config = captured.config_sha256
    _validate_timings(snapshots["timings"])
    validate_live_trace_artifact_bytes(
        snapshots["live_trace"],
        run_id=run_id,
        source_commit=expected_source_commit,
    )
    projections = _projection_checksums(snapshots["projection_checksums"])
    attestation = captured_members.get(OBSERVATION_ATTESTATION)
    if type(attestation) is not bytes:
        raise ValueError("observation attestation snapshot is unavailable")
    validate_observation_attestation(
        attestation,
        run_id=run_id,
        source_commit=expected_source_commit,
        config_sha256=captured_config,
        images=captured_images,
        projections=projections,
        live_trace=snapshots["live_trace"],
    )
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    destination = root / run_id
    if destination.exists():
        raise FileExistsError(destination)
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=root))
    claim = root / f".{run_id}.publish"
    claim_acquired = False
    os.chmod(staging, 0o700)
    try:
        with claim.open("xb") as handle:
            handle.write(run_id.encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())
        claim_acquired = True
        os.chmod(claim, 0o600)
        _fsync_directory(root)
        roles = _copy_members(staging, captured_members, snapshots)
        _write_member(
            staging,
            "cleanup/proof.json",
            canonical_json_bytes(cleanup_proof) + b"\n",
        )
        roles["cleanup/proof.json"] = "cleanup_proof"
        config_match = _HEX64.fullmatch(captured_config)
        if config_match is None:
            raise ValueError("config checksum must be an exact SHA-256")
        manifest = {
            "schema": SCHEMA,
            "run_id": run_id,
            "claim_scope": claim_scope,
            "source": dict(sorted(source.items())),
            "images": dict(sorted(captured_images.items())),
            "config_sha256": config_match.group(),
            "projection_checksums": projections,
            "members": _member_manifest(staging, roles),
        }
        signature = hmac.new(
            signing_key, canonical_json_bytes(manifest), hashlib.sha256
        ).hexdigest()
        manifest["signature"] = {
            "algorithm": "hmac-sha256",
            "key_version": signing_key_version,
            "value": signature,
        }
        _write_member(staging, "bundle.json", canonical_json_bytes(manifest) + b"\n")
        _fsync_tree(staging)
        if destination.exists():
            raise FileExistsError(destination)
        staging.rename(destination)
        _fsync_directory(root)
        claim.unlink()
        _fsync_directory(root)
        return destination
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        if claim_acquired and claim.exists():
            claim.unlink()
        raise
