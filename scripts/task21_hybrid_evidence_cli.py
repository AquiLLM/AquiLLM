"""CLI adapter for the Task21 hybrid cloud evidence publisher."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from task21_hybrid_failure_bundle import (
    CapturedRuntime,
    CommandRunner,
    capture_runtime,
    cleanup_runtime,
    publish_bundle,
)


def _source(runner) -> dict[str, object]:
    revision = runner.run(("git", "rev-parse", "HEAD"))
    status = runner.run(
        ("git", "status", "--porcelain", "--untracked-files=normal")
    )
    commit = revision.stdout.strip() if revision.succeeded else "0" * 40
    clean = revision.succeeded and status.succeeded and not status.stdout.strip()
    return {"commit": commit, "clean": clean}


def _evidence_exit_code(captured, cleanup, source) -> int:
    return 0 if captured.complete and cleanup["complete"] and source["clean"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--project-name", required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--compose-file", action="append", required=True)
    parser.add_argument("--profile", action="append", default=[])
    parser.add_argument("--arm-results", type=Path, required=True)
    parser.add_argument("--timings", type=Path, required=True)
    parser.add_argument("--projection-checksums", type=Path, required=True)
    parser.add_argument("--observation-attestation", type=Path, required=True)
    parser.add_argument("--live-trace", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument(
        "--claim-scope",
        choices=("cloud", "local-nonacceptance"),
        required=True,
    )
    parser.add_argument("--signing-key-version", required=True)
    args = parser.parse_args()
    runner = CommandRunner()
    prefix = (
        "docker",
        "compose",
        "--env-file",
        str(args.env_file.resolve()),
        "--project-name",
        args.project_name,
    )
    for compose_file in args.compose_file:
        prefix += ("--file", str(Path(compose_file).resolve()))
    for profile in args.profile:
        prefix += ("--profile", profile)
    captured = capture_runtime(runner, prefix)
    captured = CapturedRuntime(
        captured.members
        | {
            "runtime/observation-attestation.json": (
                args.observation_attestation.read_bytes()
            )
        },
        captured.images,
        captured.config_sha256,
        complete=captured.complete,
    )
    cleanup = cleanup_runtime(
        runner,
        prefix,
        project_label=f"com.docker.compose.project={args.project_name}",
    )
    source = _source(runner)
    destination = publish_bundle(
        output_root=args.output_root,
        run_id=args.run_id,
        captured=captured,
        artifacts={
            "arm_results": args.arm_results,
            "timings": args.timings,
            "projection_checksums": args.projection_checksums,
            "live_trace": args.live_trace,
        },
        cleanup_proof=cleanup,
        source=source,
        expected_source_commit=args.expected_source_commit,
        claim_scope=args.claim_scope,
        signing_key=os.environ.get("TASK21_EVIDENCE_SIGNING_KEY", "").encode(),
        signing_key_version=args.signing_key_version,
    )
    bundle = destination / "bundle.json"
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    print(f"evidence={destination} size={bundle.stat().st_size} sha256={digest}")
    return _evidence_exit_code(captured, cleanup, source)


if __name__ == "__main__":
    raise SystemExit(main())
