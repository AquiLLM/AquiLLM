"""Command-line interface for reproducible offline evidence evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import django
import yaml

from aquillm.startup import offline_evaluation_startup

# This module is a standalone CLI, so it must initialize Django before importing
# the runner and its model-dependent production services.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "aquillm.settings")
with offline_evaluation_startup():
    django.setup()

from apps.chat.evals.offline import runner  # noqa: E402


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

    document_run = commands.add_parser("document-run")
    document_run.add_argument("--real-corpus", type=Path)
    document_run.add_argument("--inventory", type=Path)
    document_run.add_argument("--review", type=Path)
    document_run.add_argument("--output", type=Path, required=True)
    document_run.add_argument("--sweeps", type=int)
    document_run.add_argument("--memory-repeats", type=int)
    document_run.add_argument("--noncanonical-smoke", action="store_true")
    document_run.add_argument("--synthetic-pages", type=int)
    document_validate = commands.add_parser("document-validate")
    document_validate.add_argument("output", type=Path)
    document_compare = commands.add_parser("document-compare")
    document_compare.add_argument("output_a", type=Path)
    document_compare.add_argument("output_b", type=Path)
    document_table = commands.add_parser("document-table")
    document_table.add_argument("aggregate_json", type=Path)
    document_table.add_argument("--output", type=Path, required=True)
    document_provenance = commands.add_parser("document-provenance")
    document_provenance.add_argument("--aggregate", type=Path)
    document_provenance.add_argument("--artifact-commit")
    document_provenance.add_argument("--output", type=Path)
    document_provenance.add_argument("--validate", type=Path)
    document_provenance.add_argument("--repository", type=Path, required=True)
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


def _operation_only_audit(attempts) -> dict:
    """Discard socket addresses while retaining an exact operation audit."""

    return {
        "guard": "deny_network",
        "scope": "fixture_generation_through_artifact_validation",
        "total_attempts": attempts.total,
        "details": [{"operation": detail["operation"]} for detail in attempts.details],
    }


def _remove_verified_staging(staging_container: Path, output_parent: Path) -> None:
    staging_container = staging_container.resolve()
    if staging_container.parent != output_parent.resolve():
        raise RuntimeError("refusing to clean an unverified document staging directory")
    if staging_container.exists():
        shutil.rmtree(staging_container)


def _document_smoke(args) -> int:
    if args.sweeps is not None or args.memory_repeats is not None:
        raise ValueError("noncanonical smoke mode cannot use canonical 30/3 settings")
    if type(args.synthetic_pages) is not int or args.synthetic_pages <= 0:
        raise ValueError("noncanonical smoke mode requires positive --synthetic-pages")
    if "smoke" not in args.output.name.casefold():
        raise ValueError(
            "noncanonical smoke output must be obviously named as smoke data"
        )
    if args.output.exists():
        raise FileExistsError(f"smoke output already exists: {args.output}")

    from apps.chat.evals.offline.network import deny_network

    with deny_network() as attempts:
        from apps.chat.evals.offline import document_pipeline_runner as document_runner
        from apps.chat.evals.offline.document_pipeline_schema import (
            generate_synthetic_pdf,
        )

        document_runner._initialize_dependencies()
        pdf_bytes, _ = generate_synthetic_pdf(args.synthetic_pages)
        case = {
            "arm": "synthetic",
            "case_id": "noncanonical-smoke-001",
            "pdf_bytes": pdf_bytes,
            "raw_sha256": hashlib.sha256(pdf_bytes).hexdigest(),
            "page_count": document_runner._page_count(pdf_bytes),
        }
        result = document_runner.run_document_case(case, chunk_size=2048, overlap=384)
    audit = _operation_only_audit(attempts)
    if audit["total_attempts"]:
        operations = [detail["operation"] for detail in audit["details"]]
        raise RuntimeError(f"network attempts observed: operations={operations!r}")

    from apps.chat.evals.offline.schema import canonical_json_bytes

    args.output.mkdir(parents=True)
    try:
        (args.output / "smoke-result.json").write_bytes(
            canonical_json_bytes(
                {
                    "mode": "noncanonical-smoke",
                    "synthetic_pages": args.synthetic_pages,
                    "case": result,
                    "network_audit": audit,
                }
            )
        )
    except Exception:
        if (
            args.output.exists()
            and args.output.resolve().parent == args.output.parent.resolve()
        ):
            shutil.rmtree(args.output)
        raise
    return 0


def _document_run(args) -> int:
    """Own one guard scope and promote only a zero-attempt validated result."""

    if args.noncanonical_smoke:
        return _document_smoke(args)
    if args.synthetic_pages is not None:
        raise ValueError(
            "--synthetic-pages is available only with --noncanonical-smoke"
        )
    if args.real_corpus is None or args.inventory is None:
        raise ValueError(
            "canonical document-run requires --real-corpus and --inventory"
        )
    if args.sweeps != 30 or args.memory_repeats != 3:
        raise ValueError(
            "canonical document-run requires exactly 30 sweeps and 3 memory passes"
        )
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"document artifact output already exists: {output}")
    review = args.review or args.inventory.with_name("document_corpus_review.yaml")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging_container = Path(
        tempfile.mkdtemp(prefix=f".{output.name}-guarded-", dir=output.parent)
    )
    staged_output = staging_container / "artifacts"

    from apps.chat.evals.offline.network import deny_network

    try:
        with deny_network() as attempts:
            from apps.chat.evals.offline import (
                document_pipeline_artifacts as document_artifacts,
            )
            from apps.chat.evals.offline import (
                document_pipeline_runner as document_runner,
            )

            source = document_runner._source_state()
            if source["dirty"]:
                raise RuntimeError("canonical document benchmark source must be clean")
            zero_audit = {
                "guard": "deny_network",
                "scope": "fixture_generation_through_artifact_validation",
                "total_attempts": 0,
                "details": [],
            }
            result = document_runner.run_document_benchmark(
                args.real_corpus,
                args.inventory,
                review,
                network_audit=zero_audit,
                sweeps=args.sweeps,
                memory_repeats=args.memory_repeats,
            )
            document_artifacts.write_document_artifacts(result, staged_output)
            document_artifacts.validate_document_artifacts(staged_output)
        audit = _operation_only_audit(attempts)
        if audit["total_attempts"]:
            operations = [detail["operation"] for detail in audit["details"]]
            raise RuntimeError(f"network attempts observed: operations={operations!r}")
        os.replace(staged_output, output)
        staging_container.rmdir()
        return 0
    except Exception:
        _remove_verified_staging(staging_container, output.parent)
        raise


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
        if args.command == "document-run":
            return _document_run(args)
        if args.command == "document-validate":
            from apps.chat.evals.offline.document_pipeline_artifacts import (
                validate_document_artifacts,
            )

            validate_document_artifacts(args.output)
            return 0
        if args.command == "document-compare":
            from apps.chat.evals.offline.document_pipeline_artifacts import (
                compare_document_results,
            )

            compare_document_results(args.output_a, args.output_b)
            return 0
        if args.command == "document-table":
            from apps.chat.evals.offline.document_pipeline_artifacts import (
                regenerate_document_table,
            )

            if args.output.exists():
                raise FileExistsError(args.output)
            args.output.write_bytes(
                regenerate_document_table(args.aggregate_json).encode("utf-8")
            )
            return 0
        if args.command == "document-provenance":
            from apps.chat.evals.offline.document_pipeline_artifacts import (
                validate_document_provenance,
                write_document_provenance,
            )

            if args.validate is not None:
                if any(
                    value is not None
                    for value in (args.aggregate, args.artifact_commit, args.output)
                ):
                    raise ValueError(
                        "--validate cannot be combined with generation arguments"
                    )
                validate_document_provenance(args.validate, args.repository)
                return 0
            if (
                args.aggregate is None
                or args.artifact_commit is None
                or args.output is None
            ):
                raise ValueError(
                    "document provenance generation requires --aggregate, "
                    "--artifact-commit, and --output"
                )
            write_document_provenance(
                args.aggregate,
                args.artifact_commit,
                args.output,
                args.repository,
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
