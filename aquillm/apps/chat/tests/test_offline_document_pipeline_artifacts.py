from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from apps.chat.evals.offline.document_pipeline_runner import _sweep_record
from apps.chat.evals.offline.document_pipeline_schema import (
    REAL_TOTAL_BYTES,
    aggregate_document_results,
    build_document_record,
)
from apps.chat.evals.offline.schema import canonical_json_bytes
from apps.documents.services.text_chunk_plan import plan_text_chunks


def _successful_record(arm: str, case_id: str, input_bytes: int, pages: int) -> dict:
    text = f"safe benchmark output for {case_id}"
    return build_document_record(
        arm=arm,
        case_id=case_id,
        input_bytes=input_bytes,
        page_count=pages,
        success=True,
        diagnostic_code="ok",
        sanitized_text=text,
        chunk_specs=plan_text_chunks(text, chunk_size=2048, overlap=384),
    )


def _canonical_result() -> dict:
    inventory_sizes = [REAL_TOTAL_BYTES - 16, *([1] * 16)]
    inventory_hashes = sorted(
        hashlib.sha256(f"pdf-{index}".encode()).hexdigest() for index in range(1, 18)
    )
    inventory_cases = [
        {
            "case_id": f"real-{index:03d}",
            "sha256": raw_hash,
            "input_bytes": size,
            "selection_rationale": (
                "all PDF members of the fixed astro_test convenience set"
            ),
            "acquisition_lineage": (
                "existing Semantic Extraction Experiment astro_test set"
            ),
            "sensitivity": "public-paper-like local research corpus",
            "redistribution_license_status": (
                "not redistributed; source license not asserted"
            ),
        }
        for index, (size, raw_hash) in enumerate(
            zip(inventory_sizes, inventory_hashes, strict=True), 1
        )
    ]
    real = [
        _successful_record("real", entry["case_id"], entry["input_bytes"], index)
        for index, entry in enumerate(inventory_cases, 1)
    ]
    synthetic = [
        _successful_record("synthetic", f"synthetic-{index:03d}", 1000 * index, pages)
        for index, pages in enumerate((1, 10, 50, 100), 1)
    ]
    static = real + synthetic
    by_identity = {(row["arm"], row["case_id"]): row for row in static}
    timing_cases = []
    timing_sweeps = []
    for arm, rows in (("real", real), ("synthetic", synthetic)):
        for sweep_index in range(30):
            offset = sweep_index % len(rows)
            ordered = rows[offset:] + rows[:offset]
            case_rows = []
            for order_index, static_row in enumerate(ordered):
                row = {
                    "schema_version": "1.0",
                    "arm": arm,
                    "case_id": static_row["case_id"],
                    "sweep_index": sweep_index,
                    "order_index": order_index,
                    "success": True,
                    "diagnostic_code": "ok",
                    "terminal_stage": "complete",
                    "detect_ns": 10 + sweep_index,
                    "extract_ns": 20 + sweep_index,
                    "sanitize_ns": 30 + sweep_index,
                    "chunk_plan_ns": 40 + sweep_index,
                    "combined_ns": 120 + sweep_index,
                    "input_bytes": static_row["input_bytes"],
                    "page_count": static_row["page_count"],
                    "extracted_codepoints": static_row["extracted_codepoints"],
                    "estimated_tokens": static_row["estimated_tokens"],
                }
                timing_cases.append(row)
                case_rows.append(row)
            timing_sweeps.append(
                _sweep_record(
                    arm,
                    sweep_index,
                    [row["case_id"] for row in ordered],
                    case_rows,
                    by_identity,
                )
            )
    memory = [
        {
            "schema_version": "1.0",
            "arm": row["arm"],
            "case_id": row["case_id"],
            "memory_index": repeat,
            "success": True,
            "diagnostic_code": "ok",
            "peak_python_traced_bytes": 10_000 + repeat,
        }
        for row in static
        for repeat in range(3)
    ]
    network = {
        "guard": "deny_network",
        "scope": "fixture_generation_through_artifact_validation",
        "total_attempts": 0,
        "details": [],
    }
    aggregate = aggregate_document_results(
        static, timing_sweeps, memory, network_audit=network
    )
    source_hashes = {
        name: hashlib.sha256(name.encode()).hexdigest()
        for name in (
            "document_pipeline_artifacts",
            "document_pipeline_runner",
            "document_pipeline_schema",
            "ingestion_parsers",
            "network_guard",
            "run_offline_evidence",
            "text_chunk_plan",
        )
    }
    manifest = {
        "schema_version": "1.0",
        "timestamp_utc": "2026-08-07T00:00:00Z",
        "source": {"commit": "a" * 40, "dirty": False},
        "hashes": {
            "source_code": {
                "algorithm": "sha256-utf8-lf-v1",
                "values": source_hashes,
            },
            "inventory_source": {
                "algorithm": "sha256-utf8-lf-v1",
                "sha256": "b" * 64,
            },
            "review_source": {
                "algorithm": "sha256-utf8-lf-v1",
                "sha256": "c" * 64,
            },
            "synthetic_protocol": {
                "algorithm": "sha256-canonical-json-v1",
                "sha256": "d" * 64,
            },
        },
        "synthetic_inputs": [
            {
                "case_id": row["case_id"],
                "pdf_sha256": hashlib.sha256(row["case_id"].encode()).hexdigest(),
                "pdf_hash_algorithm": "sha256-raw-bytes-v1",
                "expected_output_sha256": row["output_sha256"],
                "expected_output_hash_algorithm": "sha256-utf8-bytes-v1",
                "page_count": row["page_count"],
            }
            for row in synthetic
        ],
        "chunk_configuration": {
            "chunk_size_codepoints": 2048,
            "overlap_codepoints": 384,
            "pitch_codepoints": 1664,
        },
        "token_estimator": {
            "name": "production_character_estimator",
            "algorithm": "max(1, len(text) // 4)",
        },
        "dependencies": {"django": "5", "pypdf": "5", "psutil": "7"},
        "environment": {
            "os": "Windows",
            "os_release": "11",
            "machine": "AMD64",
            "python_version": "3.13.5",
            "logical_cpu_count": 8,
            "total_system_ram_bytes": 1_000_000,
            "process_bits": 64,
            "timer": "time.perf_counter_ns",
            "timer_resolution_seconds": 1e-7,
        },
        "execution": {
            "mode": "single_thread_sequential",
            "warmups_per_arm": 1,
            "warmup_role": "static_record_pass",
            "timing_sweeps_per_arm": 30,
            "memory_repeats_per_case": 3,
        },
        "network_audit": network,
        "network_audit_statement": (
            "zero connection attempts observed through the configured process-local "
            "socket guard"
        ),
    }
    inventory = {
        "schema_version": "1.0",
        "inventory_id": "astro-test-real-v1",
        "source_hash_algorithm": "sha256-raw-bytes-v1",
        "case_count": 17,
        "total_bytes": REAL_TOTAL_BYTES,
        "cases": inventory_cases,
    }
    return {
        "manifest": manifest,
        "inventory": inventory,
        "real_records": real,
        "synthetic_records": synthetic,
        "timing_cases": timing_cases,
        "timing_sweeps": timing_sweeps,
        "memory_records": memory,
        "aggregate": aggregate,
    }


def _recompute_timing_and_aggregate(result: dict) -> None:
    static = result["real_records"] + result["synthetic_records"]
    by_identity = {(row["arm"], row["case_id"]): row for row in static}
    result["timing_sweeps"] = [
        _sweep_record(
            arm,
            sweep_index,
            [row["case_id"] for row in case_rows],
            case_rows,
            by_identity,
        )
        for arm in ("real", "synthetic")
        for sweep_index in range(30)
        for case_rows in [
            [
                row
                for row in result["timing_cases"]
                if row["arm"] == arm and row["sweep_index"] == sweep_index
            ]
        ]
    ]
    result["aggregate"] = aggregate_document_results(
        static,
        result["timing_sweeps"],
        result["memory_records"],
        network_audit=result["manifest"]["network_audit"],
    )


def test_exact_membership_cardinality_and_immutable_overwrite(tmp_path: Path):
    from apps.chat.evals.offline.document_pipeline_artifacts import (
        validate_document_artifacts,
        write_document_artifacts,
    )

    output = tmp_path / "canonical"
    write_document_artifacts(_canonical_result(), output)
    assert {path.name for path in output.iterdir()} == {
        "manifest.json",
        "corpus-inventory.json",
        "real-documents.jsonl",
        "synthetic-documents.jsonl",
        "timing-cases.jsonl",
        "timing-sweeps.jsonl",
        "memory.jsonl",
        "aggregate.json",
        "real-documents.csv",
        "synthetic-documents.csv",
        "report.md",
        "paper-table.md",
        "COMPLETE",
    }
    assert len((output / "real-documents.jsonl").read_text().splitlines()) == 17
    assert len((output / "synthetic-documents.jsonl").read_text().splitlines()) == 4
    assert len((output / "timing-cases.jsonl").read_text().splitlines()) == 630
    assert len((output / "timing-sweeps.jsonl").read_text().splitlines()) == 60
    assert len((output / "memory.jsonl").read_text().splitlines()) == 63
    validate_document_artifacts(output)
    with pytest.raises(FileExistsError):
        write_document_artifacts(_canonical_result(), output)


def test_normalized_compare_excludes_timing_but_not_counts(tmp_path: Path):
    from apps.chat.evals.offline.document_pipeline_artifacts import (
        compare_document_results,
        write_document_artifacts,
    )

    first = tmp_path / "first"
    second = tmp_path / "second"
    timing_changed = _canonical_result()
    timing_changed["manifest"]["timestamp_utc"] = "2026-08-08T00:00:00Z"
    for row in timing_changed["timing_cases"]:
        for field in (
            "detect_ns",
            "extract_ns",
            "sanitize_ns",
            "chunk_plan_ns",
            "combined_ns",
        ):
            row[field] += 500
    for row in timing_changed["memory_records"]:
        row["peak_python_traced_bytes"] += 200
    _recompute_timing_and_aggregate(timing_changed)
    write_document_artifacts(_canonical_result(), first)
    write_document_artifacts(timing_changed, second)
    compare_document_results(first, second)

    count_changed = copy.deepcopy(_canonical_result())
    count_changed["real_records"][0]["estimated_tokens"] += 1
    for row in count_changed["timing_cases"]:
        if row["arm"] == "real" and row["case_id"] == "real-001":
            row["estimated_tokens"] += 1
    _recompute_timing_and_aggregate(count_changed)
    third = tmp_path / "third"
    write_document_artifacts(count_changed, third)
    with pytest.raises(ValueError, match="not reproducible"):
        compare_document_results(first, third)


def test_table_contains_exact_scope_and_exclusions(tmp_path: Path):
    from apps.chat.evals.offline.document_pipeline_artifacts import (
        regenerate_document_table,
        write_document_artifacts,
    )

    output = tmp_path / "canonical"
    write_document_artifacts(_canonical_result(), output)
    table = regenerate_document_table(output / "aggregate.json")
    assert table == (output / "paper-table.md").read_text(encoding="utf-8")
    assert "Local document preprocessing measurements" in table
    assert "fixed convenience corpus" in table
    assert "estimated tokens" in table
    assert "warm single-process in-memory" in table
    assert (
        "zero connection attempts observed through the configured process-local "
        "socket guard" in table
    )
    for exclusion in (
        "full ingestion",
        "database writes",
        "embeddings",
        "indexing",
        "vector indexing",
        "retrieval",
        "figure processing",
        "OCR processing",
        "inference",
        "concurrency",
        "GPU utilization",
        "total process memory",
        "cold-storage performance",
        "end-to-end response latency",
        "RSS",
        "GPU memory",
    ):
        assert exclusion in table


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("hashes",), {}),
        (("hashes", "extra"), {}),
        (("hashes", "source_code", "algorithm"), "sha256"),
        (("chunk_configuration", "chunk_size_codepoints"), 2049),
        (("chunk_configuration", "overlap_codepoints"), True),
        (("token_estimator", "name"), "other"),
        (("dependencies", "django"), True),
        (("environment", "logical_cpu_count"), True),
        (("environment", "timer"), "time.time_ns"),
        (("execution", "warmups_per_arm"), 2),
        (("source", "dirty"), 0),
        (("synthetic_inputs", 0, "page_count"), 2),
    ],
)
def test_manifest_exact_schema_and_canonical_values_fail_closed(
    tmp_path: Path, path: tuple[object, ...], value: object
):
    from apps.chat.evals.offline.document_pipeline_artifacts import (
        write_document_artifacts,
    )

    result = _canonical_result()
    target = result["manifest"]
    for segment in path[:-1]:
        target = target[segment]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        write_document_artifacts(result, tmp_path / "invalid")


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("environment", "cpu", "private-paper.pdf"),
        ("environment", "os_release", r"folder\paper.pdf"),
        ("dependencies", "django", "notes.docx"),
        ("environment", "machine", "document-title.txt"),
    ],
)
def test_manifest_rejects_relative_paths_and_document_basenames(
    tmp_path: Path, section: str, field: str, value: str
):
    from apps.chat.evals.offline.document_pipeline_artifacts import (
        write_document_artifacts,
    )

    result = _canonical_result()
    result["manifest"][section][field] = value
    with pytest.raises(ValueError, match="privacy|manifest|version|environment"):
        write_document_artifacts(result, tmp_path / "private")


@pytest.mark.parametrize(
    "prose",
    [
        "Private Paper Title",
        "Galaxy formation results show dark matter",
        "Introduction This paper presents a new method",
    ],
)
def test_manifest_environment_rejects_free_form_document_prose(
    tmp_path: Path, prose: str
):
    from apps.chat.evals.offline.document_pipeline_artifacts import (
        write_document_artifacts,
    )

    result = _canonical_result()
    result["manifest"]["environment"]["cpu"] = prose
    with pytest.raises(ValueError, match="environment"):
        write_document_artifacts(result, tmp_path / "prose")


@pytest.mark.parametrize("field", ["content", "title", "text"])
def test_manifest_rejects_content_sentinel_fields(tmp_path: Path, field: str):
    from apps.chat.evals.offline.document_pipeline_artifacts import (
        write_document_artifacts,
    )

    result = _canonical_result()
    result["manifest"][field] = "sentinel private document material"
    with pytest.raises(ValueError):
        write_document_artifacts(result, tmp_path / field)


def test_validator_rejects_corrupt_derived_and_private_payload(tmp_path: Path):
    from apps.chat.evals.offline.document_pipeline_artifacts import (
        validate_document_artifacts,
        write_document_artifacts,
    )

    output = tmp_path / "canonical"
    write_document_artifacts(_canonical_result(), output)
    (output / "paper-table.md").write_bytes(b"corrupt\n")
    complete = json.loads((output / "COMPLETE").read_text(encoding="utf-8"))
    complete["sha256"]["paper-table.md"] = hashlib.sha256(b"corrupt\n").hexdigest()
    (output / "COMPLETE").write_bytes(canonical_json_bytes(complete))
    with pytest.raises(ValueError, match="regenerate"):
        validate_document_artifacts(output)

    private = _canonical_result()
    private["manifest"]["document_title"] = "private paper"
    with pytest.raises(ValueError, match="privacy"):
        write_document_artifacts(private, tmp_path / "private")


def test_provenance_binds_committed_artifact_blobs_and_lineage(
    tmp_path: Path, monkeypatch
):
    from apps.chat.evals.offline.document_pipeline_artifacts import (
        validate_document_provenance,
        write_document_artifacts,
        write_document_provenance,
    )

    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "core.autocrlf", "false"], cwd=repository, check=True
    )
    subprocess.run(
        ["git", "config", "user.email", "benchmark@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Benchmark Test"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-q", "--allow-empty", "-m", "source"],
        cwd=repository,
        check=True,
    )
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    artifact_dir = repository / "evidence" / "canonical"
    result = _canonical_result()
    result["manifest"]["source"]["commit"] = source_commit
    write_document_artifacts(result, artifact_dir)
    subprocess.run(["git", "add", "evidence"], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "add artifacts"], cwd=repository, check=True
    )
    artifact_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    output = repository / "PROVENANCE.json"

    write_document_provenance(
        artifact_dir / "aggregate.json",
        artifact_commit,
        output,
        repository,
    )
    validate_document_provenance(output, repository)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["artifact_commit"] == artifact_commit
    assert payload["evaluated_source_commit"] == source_commit
    assert payload["aggregate_sha256"] == payload["artifact_hashes"]["aggregate.json"]
    assert set(payload["artifact_hashes"]) == {
        path.name for path in artifact_dir.iterdir()
    }
    assert (
        payload["artifact_hashes"]["COMPLETE"]
        == hashlib.sha256((artifact_dir / "COMPLETE").read_bytes()).hexdigest()
    )
    assert "provenance_commit" not in payload

    fake_source = copy.deepcopy(payload)
    fake_source["evaluated_source_commit"] = "f" * 40
    fake_path = repository / "FAKE-PROVENANCE.json"
    fake_path.write_bytes(canonical_json_bytes(fake_source))
    with pytest.raises(ValueError, match="source commit"):
        validate_document_provenance(fake_path, repository)

    complete_path = artifact_dir / "COMPLETE"
    committed_complete = complete_path.read_bytes()
    complete_path.write_bytes(committed_complete + b" ")
    with pytest.raises(ValueError):
        write_document_provenance(
            artifact_dir / "aggregate.json",
            artifact_commit,
            repository / "MUTATED-COMPLETE-PROVENANCE.json",
            repository,
        )
    complete_path.write_bytes(committed_complete)

    archive_path = tmp_path / "artifact.tar"
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    subprocess.run(
        ["git", "archive", "--format=tar", f"--output={archive_path}", "HEAD"],
        cwd=repository,
        check=True,
    )
    with tarfile.open(archive_path) as archive:
        archive.extractall(extracted, filter="data")
    archive_provenance = tmp_path / "ARCHIVE-PROVENANCE.json"
    monkeypatch.chdir(repository)
    write_document_provenance(
        extracted / "evidence" / "canonical" / "aggregate.json",
        artifact_commit,
        archive_provenance,
        extracted,
    )
    validate_document_provenance(archive_provenance, extracted)


def test_document_cli_parser_preserves_old_and_adds_document_commands():
    from apps.chat.evals.run_offline_evidence import _parser

    choices = next(
        action.choices for action in _parser()._actions if action.dest == "command"
    )
    assert set(choices) == {
        "run",
        "validate",
        "compare",
        "table",
        "provenance",
        "document-run",
        "document-validate",
        "document-compare",
        "document-table",
        "document-provenance",
    }


def test_document_cli_caught_network_attempt_is_operation_only_and_not_promoted(
    tmp_path: Path, monkeypatch
):
    import socket

    from apps.chat.evals import run_offline_evidence as cli
    from apps.chat.evals.offline import document_pipeline_runner
    from apps.chat.evals.offline.network import NetworkAccessError

    def caught_attempt(*args, **kwargs):
        try:
            socket.create_connection(("private.example.invalid", 443))
        except NetworkAccessError:
            pass
        return _canonical_result()

    monkeypatch.setattr(
        document_pipeline_runner, "run_document_benchmark", caught_attempt
    )
    monkeypatch.setattr(
        document_pipeline_runner,
        "_source_state",
        lambda: {"commit": "a" * 40, "dirty": False},
    )
    output = tmp_path / "canonical"
    args = SimpleNamespace(
        real_corpus=tmp_path / "corpus",
        inventory=tmp_path / "inventory.yaml",
        review=tmp_path / "review.yaml",
        output=output,
        sweeps=30,
        memory_repeats=3,
        noncanonical_smoke=False,
        synthetic_pages=None,
    )

    with pytest.raises(RuntimeError) as raised:
        cli._document_run(args)

    message = str(raised.value)
    assert "socket.create_connection" in message
    assert "private.example.invalid" not in message
    assert "443" not in message
    assert not output.exists()


def test_document_cli_fresh_subprocess_help_and_noncanonical_smoke(tmp_path: Path):
    env = {
        name: value
        for name, value in os.environ.items()
        if not re.search(
            r"credential|token|secret|password|api[_-]?key", name, re.IGNORECASE
        )
    }
    env.pop("DJANGO_DEBUG", None)
    help_result = subprocess.run(
        [sys.executable, "-m", "apps.chat.evals.run_offline_evidence", "--help"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "document-run" in help_result.stdout

    output = tmp_path / "noncanonical-smoke"
    smoke = subprocess.run(
        [
            sys.executable,
            "-m",
            "apps.chat.evals.run_offline_evidence",
            "document-run",
            "--noncanonical-smoke",
            "--synthetic-pages",
            "1",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert smoke.returncode == 0, smoke.stderr
    assert {path.name for path in output.iterdir()} == {"smoke-result.json"}
    payload = json.loads((output / "smoke-result.json").read_text(encoding="utf-8"))
    assert payload["mode"] == "noncanonical-smoke"
    assert payload["network_audit"]["total_attempts"] == 0

    rejected = subprocess.run(
        [
            sys.executable,
            "-m",
            "apps.chat.evals.run_offline_evidence",
            "document-validate",
            str(output),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert rejected.returncode == 1
