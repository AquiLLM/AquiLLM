"""Immutable artifacts and reproducibility checks for the document benchmark."""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath

from apps.chat.evals.offline.document_pipeline_runner import _sweep_record
from apps.chat.evals.offline.document_pipeline_schema import (
    _MEMORY_RECORD_KEYS,
    _STATIC_RECORD_KEYS,
    SCHEMA_VERSION,
    _validate_memory_record,
    _validate_static_record,
    _validate_timing_sweep,
    aggregate_document_results,
    validate_document_inventory,
)
from apps.chat.evals.offline.schema import canonical_json_bytes

DOCUMENT_ARTIFACTS = {
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

_JSON_SOURCES = {
    "manifest": "manifest.json",
    "inventory": "corpus-inventory.json",
    "aggregate": "aggregate.json",
}
_JSONL_SOURCES = {
    "real_records": "real-documents.jsonl",
    "synthetic_records": "synthetic-documents.jsonl",
    "timing_cases": "timing-cases.jsonl",
    "timing_sweeps": "timing-sweeps.jsonl",
    "memory_records": "memory.jsonl",
}
_EXPECTED_COUNTS = {
    "real_records": 17,
    "synthetic_records": 4,
    "timing_cases": 630,
    "timing_sweeps": 60,
    "memory_records": 63,
}
_STATIC_COLUMNS = (
    "schema_version",
    "arm",
    "case_id",
    "success",
    "diagnostic_code",
    "input_bytes",
    "input_mib",
    "page_count",
    "extracted_codepoints",
    "extracted_utf8_bytes",
    "word_count",
    "estimated_tokens",
    "chunk_count",
    "coverage_codepoints",
    "total_chunk_codepoints",
    "excess_overlap_codepoints",
    "overlap_ratio",
    "chunk_min_codepoints",
    "chunk_mean_codepoints",
    "chunk_median_codepoints",
    "chunk_max_codepoints",
    "output_sha256",
)
_TIMING_CASE_KEYS = {
    "schema_version",
    "arm",
    "case_id",
    "sweep_index",
    "order_index",
    "success",
    "diagnostic_code",
    "terminal_stage",
    "detect_ns",
    "extract_ns",
    "sanitize_ns",
    "chunk_plan_ns",
    "combined_ns",
    "input_bytes",
    "page_count",
    "extracted_codepoints",
    "estimated_tokens",
}
_STAGES = ("detect", "extract", "sanitize", "chunk_plan")
_TERMINALS = {*_STAGES, "complete"}
_TIMING_CASE_EXCLUSIONS = {
    "detect_ns",
    "extract_ns",
    "sanitize_ns",
    "chunk_plan_ns",
    "combined_ns",
}
_TIMING_SWEEP_EXCLUSIONS = {
    "case_combined_sum_ns",
    "successful_case_combined_sum_ns",
    "attempted_documents_per_second",
    "effective_successful_documents_per_second",
    "effective_pages_per_second",
    "effective_mib_per_second",
    "effective_codepoints_per_second",
    "effective_estimated_tokens_per_second",
    "success_conditioned_documents_per_second",
    "success_conditioned_pages_per_second",
    "success_conditioned_mib_per_second",
    "success_conditioned_codepoints_per_second",
    "success_conditioned_estimated_tokens_per_second",
    "milliseconds_per_attempted_document",
    "milliseconds_per_successful_document",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SENSITIVE_KEY_RE = re.compile(
    r"(?:filename|basename|content|text|(?:document_)?title|(?:raw_)?exception|"
    r"credential|"
    r"api[_-]?key|secret(?:[_-]?key)?|password|access[_-]?token|auth[_-]?token|"
    r"private[_-]?key|raw[_-]?(?:content|text)|document[_-]?(?:content|text))",
    re.IGNORECASE,
)
_SENSITIVE_VALUE_RE = re.compile(
    r"(?:api[_-]?key|secret|password|credential|token)\s*[=:]\s*[^\s\"']+|"
    r"AKIA[0-9A-Z]{16}",
    re.IGNORECASE,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) for row in rows)


def _read_json(path: Path) -> object:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if raw != canonical_json_bytes(value):
        raise ValueError(f"{path.name} is not canonical JSON")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise ValueError(f"{path.name} must contain canonical JSONL")
    rows = []
    for line in raw.splitlines(keepends=True):
        value = json.loads(line.decode("utf-8"))
        if not isinstance(value, dict) or line != canonical_json_bytes(value):
            raise ValueError(f"{path.name} contains a noncanonical JSONL row")
        rows.append(value)
    return rows


def _is_absolute_or_private_path(value: str) -> bool:
    return (
        PureWindowsPath(value).is_absolute()
        or PurePosixPath(value).is_absolute()
        or bool(re.search(r"(?i)(?:^|[/\\])(?:users|home)[/\\][^/\\]+", value))
    )


def _privacy_scan(value: object) -> None:
    private_values = {
        item.casefold()
        for item in (os.environ.get("USERNAME", ""), platform.node())
        if len(item) >= 4
    }
    inherited_secrets = {
        secret
        for name, secret in os.environ.items()
        if secret
        and re.search(
            r"credential|token|secret|password|(?:^|_)key(?:_|$)",
            name,
            re.IGNORECASE,
        )
    }

    def visit(item: object) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ValueError("privacy validation requires string field names")
                if _SENSITIVE_KEY_RE.fullmatch(key):
                    raise ValueError(f"privacy-prohibited field: {key}")
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif isinstance(item, str):
            folded = item.casefold()
            if _is_absolute_or_private_path(item) or _SENSITIVE_VALUE_RE.search(item):
                raise ValueError("privacy-prohibited path or credential value")
            if any(private in folded for private in private_values):
                raise ValueError("privacy-prohibited username or hostname")
            if any(secret in item for secret in inherited_secrets):
                raise ValueError("privacy-prohibited inherited credential value")

    visit(value)


def _sort_result(result: Mapping[str, object]) -> dict:
    payload = copy.deepcopy(dict(result))
    payload["real_records"] = sorted(
        payload["real_records"], key=lambda row: row["case_id"]
    )
    payload["synthetic_records"] = sorted(
        payload["synthetic_records"], key=lambda row: row["case_id"]
    )
    payload["timing_cases"] = sorted(
        payload["timing_cases"],
        key=lambda row: (
            0 if row["arm"] == "real" else 1,
            row["sweep_index"],
            row["order_index"],
        ),
    )
    payload["timing_sweeps"] = sorted(
        payload["timing_sweeps"],
        key=lambda row: (0 if row["arm"] == "real" else 1, row["sweep_index"]),
    )
    payload["memory_records"] = sorted(
        payload["memory_records"],
        key=lambda row: (
            0 if row["arm"] == "real" else 1,
            row["case_id"],
            row["memory_index"],
        ),
    )
    return payload


def _validate_manifest(manifest: object) -> None:
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be a mapping")
    required = {
        "schema_version",
        "timestamp_utc",
        "source",
        "hashes",
        "synthetic_inputs",
        "chunk_configuration",
        "token_estimator",
        "dependencies",
        "environment",
        "execution",
        "network_audit",
        "network_audit_statement",
    }
    if set(manifest) != required:
        raise ValueError("manifest keys do not match the required schema")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("manifest schema_version must be 1.0")
    source = manifest["source"]
    if not isinstance(source, Mapping) or set(source) != {"commit", "dirty"}:
        raise ValueError("manifest source keys are invalid")
    if not isinstance(source["commit"], str) or not re.fullmatch(
        r"[0-9a-f]{40,64}", source["commit"]
    ):
        raise ValueError("manifest source commit is invalid")
    if source["dirty"] is not False:
        raise ValueError("canonical document artifacts require a clean source")
    execution = manifest["execution"]
    if (
        not isinstance(execution, Mapping)
        or execution.get("timing_sweeps_per_arm") != 30
        or execution.get("memory_repeats_per_case") != 3
    ):
        raise ValueError(
            "canonical document artifacts require 30 sweeps and 3 memory passes"
        )
    audit = manifest["network_audit"]
    expected_audit = {
        "guard": "deny_network",
        "scope": "fixture_generation_through_artifact_validation",
        "total_attempts": 0,
        "details": [],
    }
    if audit != expected_audit:
        raise ValueError(
            "canonical document artifacts require an exact zero network audit"
        )
    if manifest["network_audit_statement"] != (
        "zero connection attempts observed through the configured process-local "
        "socket guard"
    ):
        raise ValueError("network audit statement is invalid")
    synthetic_inputs = manifest["synthetic_inputs"]
    if not isinstance(synthetic_inputs, list) or [
        row.get("case_id") for row in synthetic_inputs if isinstance(row, Mapping)
    ] != [f"synthetic-{index:03d}" for index in range(1, 5)]:
        raise ValueError("manifest synthetic inputs are invalid")
    synthetic_keys = {
        "case_id",
        "pdf_sha256",
        "pdf_hash_algorithm",
        "expected_output_sha256",
        "expected_output_hash_algorithm",
        "page_count",
    }
    for row in synthetic_inputs:
        if set(row) != synthetic_keys:
            raise ValueError("manifest synthetic input keys are invalid")
        if (
            row["pdf_hash_algorithm"] != "sha256-raw-bytes-v1"
            or row["expected_output_hash_algorithm"] != "sha256-utf8-bytes-v1"
        ):
            raise ValueError("manifest synthetic input hash algorithm is invalid")
        for field in ("pdf_sha256", "expected_output_sha256"):
            if not isinstance(row[field], str) or not _SHA256_RE.fullmatch(row[field]):
                raise ValueError("manifest synthetic input hash is invalid")


def _validate_timing_case(
    row: object,
    static_by_identity: Mapping[tuple[str, str], Mapping[str, object]],
) -> None:
    if not isinstance(row, Mapping) or set(row) != _TIMING_CASE_KEYS:
        raise ValueError("timing case keys do not match the required schema")
    if row["schema_version"] != SCHEMA_VERSION:
        raise ValueError("timing case schema_version must be 1.0")
    identity = (row["arm"], row["case_id"])
    static = static_by_identity.get(identity)
    if static is None:
        raise ValueError("timing case has no matching static record")
    for field in ("sweep_index", "order_index", "combined_ns"):
        if type(row[field]) is not int or row[field] < 0:
            raise ValueError(f"timing case {field} must be a nonnegative integer")
    if type(row["success"]) is not bool or row["success"] != static["success"]:
        raise ValueError("timing case success is invalid")
    if row["diagnostic_code"] != static["diagnostic_code"]:
        raise ValueError("timing case diagnostic does not match static record")
    terminal = row["terminal_stage"]
    if terminal not in _TERMINALS or (row["success"] and terminal != "complete"):
        raise ValueError("timing case terminal_stage is invalid")
    terminal_index = _STAGES.index(terminal) if terminal in _STAGES else len(_STAGES)
    for index, stage in enumerate(_STAGES):
        value = row[f"{stage}_ns"]
        if index <= terminal_index:
            if type(value) is not int or value < 0:
                raise ValueError("executed stage timing must be nonnegative integer ns")
        elif value is not None:
            raise ValueError("unexecuted stage timing must be null")
    for field in (
        "input_bytes",
        "page_count",
        "extracted_codepoints",
        "estimated_tokens",
    ):
        if row[field] != static[field]:
            raise ValueError(f"timing case {field} does not match static record")


def _validate_result(result: Mapping[str, object]) -> None:
    required = {
        "manifest",
        "inventory",
        "real_records",
        "synthetic_records",
        "timing_cases",
        "timing_sweeps",
        "memory_records",
        "aggregate",
    }
    if set(result) != required:
        raise ValueError("document result keys do not match the required schema")
    _privacy_scan(result)
    _validate_manifest(result["manifest"])
    validate_document_inventory(result["inventory"])
    for key, count in _EXPECTED_COUNTS.items():
        rows = result[key]
        if not isinstance(rows, list) or len(rows) != count:
            raise ValueError(f"canonical {key} cardinality must be exactly {count}")

    real = result["real_records"]
    synthetic = result["synthetic_records"]
    if [row.get("case_id") for row in real] != [
        f"real-{index:03d}" for index in range(1, 18)
    ] or any(row.get("arm") != "real" for row in real):
        raise ValueError("real static rows are not in canonical identity order")
    if [row.get("case_id") for row in synthetic] != [
        f"synthetic-{index:03d}" for index in range(1, 5)
    ] or any(row.get("arm") != "synthetic" for row in synthetic):
        raise ValueError("synthetic static rows are not in canonical identity order")
    static = [*real, *synthetic]
    identities: set[tuple[str, str]] = set()
    for row in static:
        if set(row) != _STATIC_RECORD_KEYS:
            raise ValueError("static row keys do not match required schema")
        identity = _validate_static_record(row)
        if identity in identities:
            raise ValueError("duplicate static identity")
        identities.add(identity)
    inventory_by_id = {row["case_id"]: row for row in result["inventory"]["cases"]}
    for row in real:
        if row["input_bytes"] != inventory_by_id[row["case_id"]]["input_bytes"]:
            raise ValueError("real static input_bytes does not match corpus inventory")

    static_by_identity = {(row["arm"], row["case_id"]): row for row in static}
    expected_timing_order = []
    grouped_cases: dict[tuple[str, int], list[dict]] = {}
    for row in result["timing_cases"]:
        _validate_timing_case(row, static_by_identity)
        expected_timing_order.append(
            (row["arm"], row["sweep_index"], row["order_index"])
        )
        grouped_cases.setdefault((row["arm"], row["sweep_index"]), []).append(row)
    if expected_timing_order != sorted(
        expected_timing_order,
        key=lambda item: (0 if item[0] == "real" else 1, item[1], item[2]),
    ):
        raise ValueError("timing cases are not in canonical order")

    sweep_identities = []
    for row in result["timing_sweeps"]:
        _validate_timing_sweep(row, identities)
        key = (row["arm"], row["sweep_index"])
        case_rows = grouped_cases.get(key, [])
        regenerated = _sweep_record(
            row["arm"],
            row["sweep_index"],
            [case["case_id"] for case in case_rows],
            case_rows,
            static_by_identity,
        )
        if dict(row) != regenerated:
            raise ValueError("timing sweep does not regenerate from timing cases")
        sweep_identities.append(key)
    expected_sweeps = [
        (arm, index) for arm in ("real", "synthetic") for index in range(30)
    ]
    if sweep_identities != expected_sweeps:
        raise ValueError("timing sweeps are not in canonical order")
    for arm, case_ids in (
        ("real", [f"real-{index:03d}" for index in range(1, 18)]),
        ("synthetic", [f"synthetic-{index:03d}" for index in range(1, 5)]),
    ):
        for sweep_index in range(30):
            rows = grouped_cases[(arm, sweep_index)]
            offset = sweep_index % len(case_ids)
            expected_ids = case_ids[offset:] + case_ids[:offset]
            if [row["order_index"] for row in rows] != list(range(len(case_ids))):
                raise ValueError("timing case order indexes are not contiguous")
            if [row["case_id"] for row in rows] != expected_ids:
                raise ValueError("timing case rotation order is invalid")

    memory_identities = []
    for row in result["memory_records"]:
        if set(row) != _MEMORY_RECORD_KEYS:
            raise ValueError("memory row keys do not match required schema")
        _validate_memory_record(row, identities)
        memory_identities.append((row["arm"], row["case_id"], row["memory_index"]))
    expected_memory = [
        (row["arm"], row["case_id"], repeat) for row in static for repeat in range(3)
    ]
    if memory_identities != expected_memory:
        raise ValueError("memory rows are not in canonical order")

    aggregate = aggregate_document_results(
        static,
        result["timing_sweeps"],
        result["memory_records"],
        network_audit=result["manifest"]["network_audit"],
    )
    if result["aggregate"] != aggregate:
        raise ValueError("aggregate does not regenerate from canonical source rows")


def _render_static_csv(rows: Sequence[Mapping[str, object]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(_STATIC_COLUMNS)
    for row in rows:
        writer.writerow(
            ["N/A" if row[key] is None else row[key] for key in _STATIC_COLUMNS]
        )
    return stream.getvalue()


def _display(value: object, precision: int = 3) -> str:
    if value is None:
        return "N/A"
    if type(value) is float:
        return f"{value:.{precision}f}"
    return str(value)


def _document_table_text(aggregate: Mapping[str, object]) -> str:
    rows = []
    for arm, label in (
        ("real", "Real fixed convenience corpus"),
        ("synthetic", "Synthetic scaling corpus"),
    ):
        totals = aggregate[arm]
        timing = aggregate["timing"][arm]["effective_pages_per_second"]
        memory = aggregate["memory"][arm]
        rows.append(
            "| "
            + " | ".join(
                (
                    label,
                    _display(totals["attempted_count"]),
                    _display(totals["success_count"]),
                    _display(totals["total_page_count"]),
                    _display(totals["total_estimated_tokens"]),
                    _display(timing["median"]),
                    _display(timing["nearest_rank_p95"]),
                    _display(memory["max_peak_python_traced_bytes_over_successes"]),
                )
            )
            + " |"
        )
    return "\n".join(
        [
            "# Local document preprocessing measurements",
            "",
            "| Corpus arm | Attempted documents | Successful documents | Pages | "
            "estimated tokens | Median effective pages/s | p95 effective pages/s | "
            "Max Python traced bytes |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "Scope: measurements use a fixed convenience corpus plus deterministic "
            "synthetic scaling inputs and cover warm single-process in-memory local "
            "preprocessing. The run recorded zero connection attempts observed through "
            "the configured process-local socket guard.",
            "",
            "Exclusions: these results are not measurements of full ingestion, "
            "indexing, or retrieval; concurrency; end-to-end response latency; RSS; "
            "or GPU memory.",
            "",
        ]
    )


def regenerate_document_table(aggregate_path: Path) -> str:
    """Regenerate the publication table only from canonical aggregate JSON."""

    aggregate = _read_json(Path(aggregate_path))
    if not isinstance(aggregate, Mapping):
        raise ValueError("aggregate must be a mapping")
    _privacy_scan(aggregate)
    return _document_table_text(aggregate)


def _report_text(
    aggregate: Mapping[str, object],
    real: Sequence[Mapping[str, object]],
    synthetic: Sequence[Mapping[str, object]],
) -> str:
    failures = [row for row in [*real, *synthetic] if row["success"] is False]
    failure_lines = [
        f"- {row['arm']}/{row['case_id']}: {row['diagnostic_code']} "
        "(output metrics: N/A)"
        for row in failures
    ] or ["- None"]
    return (
        "# Offline document preprocessing report\n\n"
        + _document_table_text(aggregate)
        + "\n## Failure diagnostics\n\n"
        + "\n".join(failure_lines)
        + "\n"
    )


def _materialize(temp_dir: Path, result: Mapping[str, object]) -> None:
    for key, name in _JSON_SOURCES.items():
        (temp_dir / name).write_bytes(canonical_json_bytes(result[key]))
    for key, name in _JSONL_SOURCES.items():
        (temp_dir / name).write_bytes(_jsonl_bytes(result[key]))
    (temp_dir / "real-documents.csv").write_text(
        _render_static_csv(result["real_records"]), encoding="utf-8", newline=""
    )
    (temp_dir / "synthetic-documents.csv").write_text(
        _render_static_csv(result["synthetic_records"]), encoding="utf-8", newline=""
    )
    (temp_dir / "paper-table.md").write_text(
        _document_table_text(result["aggregate"]), encoding="utf-8", newline="\n"
    )
    (temp_dir / "report.md").write_text(
        _report_text(
            result["aggregate"], result["real_records"], result["synthetic_records"]
        ),
        encoding="utf-8",
        newline="\n",
    )
    hashes = {
        name: _sha256(temp_dir / name)
        for name in sorted(DOCUMENT_ARTIFACTS - {"COMPLETE"})
    }
    (temp_dir / "COMPLETE").write_bytes(
        canonical_json_bytes({"schema_version": SCHEMA_VERSION, "sha256": hashes})
    )


def write_document_artifacts(result: dict, output_dir: Path) -> None:
    """Validate and atomically publish one immutable canonical result directory."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"document artifact output already exists: {output_dir}")
    result = _sort_result(result)
    _validate_result(result)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
    )
    try:
        _materialize(temp_dir, result)
        validate_document_artifacts(temp_dir)
        os.replace(temp_dir, output_dir)
    except Exception:
        if (
            temp_dir.exists()
            and temp_dir.parent.resolve() == output_dir.parent.resolve()
        ):
            shutil.rmtree(temp_dir)
        raise


def _load_sources(output_dir: Path) -> dict:
    result = {key: _read_json(output_dir / name) for key, name in _JSON_SOURCES.items()}
    result.update(
        {key: _read_jsonl(output_dir / name) for key, name in _JSONL_SOURCES.items()}
    )
    return result


def validate_document_artifacts(output_dir: Path) -> None:
    """Fail closed unless every canonical source and derived byte is reproducible."""

    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise ValueError("document artifact output must be a directory")
    actual = {path.name for path in output_dir.iterdir()}
    if actual != DOCUMENT_ARTIFACTS or any(
        not path.is_file() for path in output_dir.iterdir()
    ):
        raise ValueError("document artifact membership is not exact")
    complete = _read_json(output_dir / "COMPLETE")
    if not isinstance(complete, Mapping) or set(complete) != {
        "schema_version",
        "sha256",
    }:
        raise ValueError("COMPLETE keys are invalid")
    expected_hashes = complete["sha256"]
    if (
        complete["schema_version"] != SCHEMA_VERSION
        or not isinstance(expected_hashes, Mapping)
        or set(expected_hashes) != DOCUMENT_ARTIFACTS - {"COMPLETE"}
    ):
        raise ValueError("COMPLETE inventory is invalid")
    for name, expected in expected_hashes.items():
        if not isinstance(expected, str) or not _SHA256_RE.fullmatch(expected):
            raise ValueError("COMPLETE contains an invalid SHA-256")
        if _sha256(output_dir / name) != expected:
            raise ValueError(f"artifact hash mismatch: {name}")
    result = _load_sources(output_dir)
    _validate_result(result)
    if (output_dir / "real-documents.csv").read_text(
        encoding="utf-8"
    ) != _render_static_csv(result["real_records"]):
        raise ValueError("real-documents.csv does not regenerate from canonical JSONL")
    if (output_dir / "synthetic-documents.csv").read_text(
        encoding="utf-8"
    ) != _render_static_csv(result["synthetic_records"]):
        raise ValueError(
            "synthetic-documents.csv does not regenerate from canonical JSONL"
        )
    if (output_dir / "paper-table.md").read_text(
        encoding="utf-8"
    ) != _document_table_text(result["aggregate"]):
        raise ValueError("paper-table.md does not regenerate from aggregate JSON")
    if (output_dir / "report.md").read_text(encoding="utf-8") != _report_text(
        result["aggregate"], result["real_records"], result["synthetic_records"]
    ):
        raise ValueError("report.md does not regenerate from canonical sources")


def normalized_document_result(output_dir: Path) -> bytes:
    """Return parsed sources with only design timing/memory paths removed."""

    output_dir = Path(output_dir)
    validate_document_artifacts(output_dir)
    payload = _load_sources(output_dir)
    payload["manifest"].pop("timestamp_utc")
    for row in payload["timing_cases"]:
        for field in _TIMING_CASE_EXCLUSIONS:
            row.pop(field)
    for row in payload["timing_sweeps"]:
        for field in _TIMING_SWEEP_EXCLUSIONS:
            row.pop(field)
    for row in payload["memory_records"]:
        row.pop("peak_python_traced_bytes")
    payload["aggregate"].pop("timing")
    payload["aggregate"].pop("memory")
    return canonical_json_bytes(payload)


def compare_document_results(first: Path, second: Path) -> None:
    """Raise unless two validated runs match outside exact design exclusions."""

    if normalized_document_result(first) != normalized_document_result(second):
        raise ValueError("document benchmark results are not reproducible")


def _git_bytes(repository: Path, commit: str, relative_path: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repository), "show", f"{commit}:{relative_path}"],
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _git_repository_for_commit(repository: Path, commit: str) -> Path:
    candidates = (repository, Path.cwd(), Path(__file__).resolve().parent)
    for candidate in candidates:
        root = subprocess.run(
            ["git", "-C", str(candidate), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
        )
        if root.returncode:
            continue
        git_root = Path(root.stdout.strip()).resolve()
        exists = subprocess.run(
            ["git", "-C", str(git_root), "cat-file", "-e", f"{commit}^{{commit}}"],
            capture_output=True,
        )
        if exists.returncode == 0:
            return git_root
    raise ValueError("artifact commit is unavailable in the active Git repository")


def write_document_provenance(
    aggregate_path: Path,
    artifact_commit: str,
    output_path: Path,
    repository: Path | None = None,
) -> None:
    """Write non-self-referential source, artifact, hash, and algorithm lineage."""

    aggregate_path = Path(aggregate_path).resolve()
    output_path = Path(output_path)
    artifact_dir = aggregate_path.parent
    if repository is None:
        repository = Path(
            subprocess.run(
                ["git", "-C", str(artifact_dir), "rev-parse", "--show-toplevel"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    repository = Path(repository).resolve()
    git_repository = _git_repository_for_commit(repository, artifact_commit)
    validate_document_artifacts(artifact_dir)
    if output_path.exists():
        raise FileExistsError(f"provenance output already exists: {output_path}")
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", artifact_commit):
        raise ValueError("artifact commit must be a full hexadecimal SHA")
    relative_dir = artifact_dir.relative_to(repository).as_posix()
    complete = _read_json(artifact_dir / "COMPLETE")
    for name, digest in complete["sha256"].items():
        relative = f"{relative_dir}/{name}" if relative_dir else name
        blob = _git_bytes(git_repository, artifact_commit, relative)
        if (
            blob != (artifact_dir / name).read_bytes()
            or hashlib.sha256(blob).hexdigest() != digest
        ):
            raise ValueError(f"artifact bytes do not match Git blob: {name}")
    manifest = _read_json(artifact_dir / "manifest.json")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "evaluated_source_commit": manifest["source"]["commit"],
        "artifact_commit": artifact_commit.lower(),
        "artifact_hashes": dict(complete["sha256"]),
        "aggregate_sha256": complete["sha256"]["aggregate.json"],
        "corpus_inventory_hash": {
            "algorithm": "sha256-raw-bytes-v1",
            "sha256": complete["sha256"]["corpus-inventory.json"],
        },
        "source_code_hashes": manifest["hashes"]["source_code"],
        "inventory_source_hash": manifest["hashes"]["inventory_source"],
        "review_source_hash": manifest["hashes"]["review_source"],
        "synthetic_protocol_hash": manifest["hashes"]["synthetic_protocol"],
        "configuration_hash": {
            "algorithm": "sha256-canonical-json-v1",
            "sha256": hashlib.sha256(
                canonical_json_bytes(
                    {
                        "chunk_configuration": manifest["chunk_configuration"],
                        "execution": manifest["execution"],
                        "token_estimator": manifest["token_estimator"],
                    }
                )
            ).hexdigest(),
        },
    }
    _privacy_scan(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_name(f".{output_path.name}.tmp")
    try:
        temp.write_bytes(canonical_json_bytes(payload))
        os.replace(temp, output_path)
    finally:
        temp.unlink(missing_ok=True)


def validate_document_provenance(path: Path, repository: Path) -> None:
    """Validate provenance bytes, commit existence, artifact blobs, and lineage."""

    path = Path(path)
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError("provenance must be a mapping")
    _privacy_scan(payload)
    required = {
        "schema_version",
        "evaluated_source_commit",
        "artifact_commit",
        "artifact_hashes",
        "aggregate_sha256",
        "corpus_inventory_hash",
        "source_code_hashes",
        "inventory_source_hash",
        "review_source_hash",
        "synthetic_protocol_hash",
        "configuration_hash",
    }
    if set(payload) != required or payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("provenance keys do not match the required schema")
    repository = Path(repository).resolve()
    artifact_commit = payload["artifact_commit"]
    _git_repository_for_commit(repository, artifact_commit)
    candidates = []
    for complete_path in repository.rglob("COMPLETE"):
        try:
            complete = _read_json(complete_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if complete.get("sha256") == payload["artifact_hashes"]:
            candidates.append(complete_path.parent)
    if len(candidates) != 1:
        raise ValueError("provenance artifact directory is absent or ambiguous")
    artifact_dir = candidates[0]
    validate_document_artifacts(artifact_dir)
    manifest = _read_json(artifact_dir / "manifest.json")
    if payload["evaluated_source_commit"] != manifest["source"]["commit"]:
        raise ValueError("provenance evaluated source commit contradicts manifest")
    expected_path = path.with_name(f".{path.name}.validation")
    write_document_provenance(
        artifact_dir / "aggregate.json",
        artifact_commit,
        expected_path,
        repository,
    )
    try:
        if expected_path.read_bytes() != path.read_bytes():
            raise ValueError("provenance does not regenerate from artifact lineage")
    finally:
        expected_path.unlink(missing_ok=True)


__all__ = [
    "compare_document_results",
    "normalized_document_result",
    "regenerate_document_table",
    "validate_document_artifacts",
    "validate_document_provenance",
    "write_document_artifacts",
    "write_document_provenance",
]
