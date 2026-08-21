#!/usr/bin/env python3
"""Capture the exact complete Compose runtime identity for the live generator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

try:
    from scripts import task21_hybrid_failure_bundle as _runtime
except ImportError:
    import task21_hybrid_failure_bundle as _runtime

SCHEMA = "task21-hybrid-runtime-identity-v1"


def _write_private_no_replace(path: Path, payload: object) -> None:
    if not path.is_absolute() or path.exists() or not path.parent.is_dir():
        raise ValueError("runtime identity destination is invalid")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.link(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--compose-file", action="append", required=True)
    parser.add_argument("--profile", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if re.fullmatch(r"[0-9a-f]{32}", args.run_id) is None:
        raise ValueError("run id is invalid")
    if re.fullmatch(r"[0-9a-f]{40}", args.source_commit) is None:
        raise ValueError("source commit is invalid")
    prefix = (
        "docker",
        "compose",
        "--env-file",
        str(args.env_file.resolve()),
        "--project-name",
        f"aquillm-task21-{args.run_id}",
    )
    for compose_file in args.compose_file:
        prefix += ("--file", str(compose_file.resolve()))
    for profile in args.profile:
        prefix += ("--profile", profile)
    captured = _runtime.capture_runtime(_runtime.CommandRunner(), prefix)
    if captured.complete is not True or tuple(captured.images) != _runtime.SERVICES:
        raise RuntimeError("runtime identity capture is incomplete")
    payload = {
        "schema": SCHEMA,
        "run_id": args.run_id,
        "source_commit": args.source_commit,
        "config_sha256": captured.config_sha256,
        "images": captured.images,
        "complete": True,
    }
    _write_private_no_replace(args.output.resolve(), payload)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"runtime_identity_sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
