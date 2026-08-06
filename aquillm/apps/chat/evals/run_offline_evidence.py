"""Command-line interface for reproducible offline evidence evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from apps.chat.evals.offline import runner


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--fixtures", type=Path, required=True)
    run.add_argument("--test-manifest", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--timing-repeats", type=int, required=True)
    run.add_argument("--skip-tests", action="store_true")
    validate = commands.add_parser("validate")
    validate.add_argument("output", type=Path)
    compare = commands.add_parser("compare")
    compare.add_argument("output_a", type=Path)
    compare.add_argument("output_b", type=Path)
    table = commands.add_parser("table")
    table.add_argument("aggregate_json", type=Path)
    table.add_argument("--output", type=Path, required=True)
    provenance = commands.add_parser("provenance")
    provenance.add_argument("aggregate_json", type=Path)
    provenance.add_argument("--artifact-commit", required=True)
    provenance.add_argument("--output", type=Path, required=True)
    return parser


def _unavailable_tests(path: Path) -> dict:
    if not path.is_file():
        return {
            "schema_version": runner.SCHEMA_VERSION,
            "entries": [],
            "summary": {
                "collected": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "errors": 0,
                "unavailable": 0,
            },
            "network_scope": "component_execution_only",
            "declared_network_policy": "no_network",
        }
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries = [
        {
            **entry,
            "outcome": "unavailable",
            "reason": f"{entry['reason']} Test execution was skipped.",
        }
        for entry in data["entries"]
    ]
    return {
        "schema_version": runner.SCHEMA_VERSION,
        "entries": entries,
        "summary": {
            "collected": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "unavailable": len(entries),
        },
        "network_scope": "component_execution_only",
        "declared_network_policy": "no_network",
    }


def _run(args) -> int:
    project_root = Path.cwd()
    source_state = runner._git_source_state(project_root)
    if source_state[1]:
        return 1
    result = runner.run_component_evaluation(args.fixtures, args.timing_repeats)
    tests = (
        _unavailable_tests(args.test_manifest)
        if args.skip_tests
        else runner.run_test_manifest(args.test_manifest, project_root)
    )
    result["tests"] = tests
    result["aggregate"]["tests"] = tests["summary"]
    if "manifest" not in result:
        result["manifest"] = runner.build_manifest(
            args.fixtures,
            args.test_manifest,
            project_root,
            args.timing_repeats,
            result["network_attempts"],
            source_state=source_state,
        )
    else:
        result["manifest"].setdefault(
            "component_network_attempts",
            result.get("network_attempts", {"total": 0, "details": []}),
        )
        if (
            result["manifest"].get("source_commit") != source_state[0]
            or result["manifest"].get("source_dirty") is not False
        ):
            return 1
    result["aggregate"]["run"] = {
        "run_id": result["manifest"]["run_id"],
        "timestamp_utc": result["manifest"]["timestamp_utc"],
        "source_commit": result["manifest"]["source_commit"],
    }
    attempts = result["manifest"]["component_network_attempts"].get("total", 0)
    if result["manifest"].get("source_dirty") or attempts:
        return 1
    summary = tests["summary"]
    invalid_included_outcomes = [
        entry
        for entry in tests.get("entries", [])
        if entry.get("status") == "included" and entry.get("outcome") != "passed"
    ]
    if (
        tests.get("exit_code", 0) != 0
        or tests.get("integrity_failure")
        or tests.get("subprocess_network_attempts", {}).get("total", 0)
        or summary.get("failed", 0)
        or summary.get("errors", 0)
        or summary.get("skipped", 0)
        or invalid_included_outcomes
    ):
        return 1
    runner.write_artifacts(result, args.output)
    runner.validate_artifacts(args.output)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            return _run(args)
        if args.command == "validate":
            runner.validate_artifacts(args.output)
            return 0
        if args.command == "compare":
            runner.validate_artifacts(args.output_a)
            runner.validate_artifacts(args.output_b)
            return (
                0
                if runner.normalized_reproducibility_bytes(args.output_a)
                == runner.normalized_reproducibility_bytes(args.output_b)
                else 1
            )
        if args.command == "table":
            if args.output.exists():
                raise FileExistsError(args.output)
            args.output.write_text(
                runner.regenerate_paper_table(args.aggregate_json),
                encoding="utf-8",
                newline="\n",
            )
            return 0
        if args.command == "provenance":
            runner.write_provenance(
                args.aggregate_json, args.artifact_commit, args.output
            )
            return 0
    except Exception as exc:
        print(
            json.dumps({"error": str(exc), "type": type(exc).__name__}), file=sys.stderr
        )
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
