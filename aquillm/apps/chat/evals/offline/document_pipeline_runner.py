"""Warm, in-memory measurement runner for the document preprocessing pipeline."""

from __future__ import annotations

import gc
import hashlib
import io
import os
import platform
import stat
import struct
import subprocess
import sys
import time
import tracemalloc
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import psutil
from pypdf import PdfReader
from pypdf.errors import FileNotDecryptedError, PdfReadError

from apps.chat.evals.offline.document_pipeline_schema import (
    aggregate_document_results,
    build_document_record,
    generate_synthetic_pdf,
    load_document_inventory,
    load_document_review,
)
from apps.chat.evals.offline.schema import sha256_canonical_text
from apps.chat.services import rag_evidence
from apps.documents.services.text_chunk_plan import plan_text_chunks
from aquillm.ingestion import parsers
from aquillm.task_ingest_helpers import sanitize_db_text

SCHEMA_VERSION = "1.0"


class BenchmarkIntegrityError(RuntimeError):
    """Raised when a benchmark fixture or protocol invariant fails closed."""


def _integrity_error(code: str) -> ValueError:
    return ValueError(f"real corpus integrity error: {code}")


def _page_count(pdf_bytes: bytes) -> int | None:
    try:
        return len(PdfReader(io.BytesIO(pdf_bytes)).pages)
    except Exception:
        return None


def _is_link_or_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


def resolve_real_corpus(
    corpus_dir: Path,
    inventory: dict,
    *,
    allow_unlisted_pdfs: bool = False,
) -> list[dict]:
    """Preload direct-child PDFs and match them to a path-free hash inventory."""

    inventory_cases = inventory.get("cases")
    if not isinstance(inventory_cases, list):
        raise _integrity_error("invalid_inventory")
    by_hash: dict[str, dict] = {}
    for entry in inventory_cases:
        if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str):
            raise _integrity_error("invalid_inventory")
        raw_hash = entry["sha256"]
        if raw_hash in by_hash:
            raise _integrity_error("duplicate_inventory_hash")
        by_hash[raw_hash] = entry

    observed: dict[str, bytes] = {}
    for path in Path(corpus_dir).iterdir():
        if path.suffix.casefold() != ".pdf":
            continue
        if _is_link_or_reparse(path):
            raise _integrity_error("linked_pdf")
        if not path.is_file():
            continue
        pdf_bytes = path.read_bytes()
        raw_hash = hashlib.sha256(pdf_bytes).hexdigest()
        if raw_hash in observed:
            raise _integrity_error("duplicate_pdf_hash")
        if raw_hash not in by_hash:
            if allow_unlisted_pdfs:
                continue
            raise _integrity_error("unlisted_pdf")
        observed[raw_hash] = pdf_bytes

    if set(by_hash) - set(observed):
        raise _integrity_error("missing_inventory_pdf")

    resolved = []
    for raw_hash, entry in by_hash.items():
        pdf_bytes = observed[raw_hash]
        if entry.get("input_bytes") != len(pdf_bytes):
            raise _integrity_error("pdf_size_mismatch")
        resolved.append(
            {
                "arm": "real",
                "case_id": entry["case_id"],
                "pdf_bytes": pdf_bytes,
                "raw_sha256": raw_hash,
                "page_count": _page_count(pdf_bytes),
            }
        )
    return sorted(resolved, key=lambda case: case["case_id"])


def _diagnostic_for(exc: Exception, *, page_count: int | None) -> str:
    if isinstance(exc, _EmptyPrimaryTextError):
        return "empty_primary_text"
    if isinstance(exc, FileNotDecryptedError):
        return "encrypted_pdf"
    if isinstance(exc, PdfReadError) or page_count is None:
        return "invalid_pdf"
    return "parser_error"


class _EmptyPrimaryTextError(ValueError):
    pass


def _build_pipeline_execution(
    *,
    success: bool,
    terminal_stage: str,
    exception: Exception | None,
    sanitized_text: str | None,
    chunk_specs: object,
) -> dict:
    return {
        "success": success,
        "terminal_stage": terminal_stage,
        "_exception": exception,
        "_sanitized_text": sanitized_text,
        "_chunk_specs": chunk_specs,
    }


def _execute_pipeline_core(
    case: dict,
    *,
    chunk_size: int,
    overlap: int,
    stage_runner: Callable[[str, Callable[[], object]], object] | None = None,
    terminal_callback: Callable[[], None] | None = None,
) -> dict:
    """Execute the single production stage sequence and retain raw outcome state."""

    run_stage = stage_runner or (lambda _stage, operation: operation())
    mark_terminal = terminal_callback or (lambda: None)
    terminal_stage = "detect"
    try:
        ingest_type = run_stage(
            "detect",
            lambda: parsers.detect_ingest_type(
                f"{case['case_id']}.pdf", "application/pdf"
            ),
        )
        terminal_stage = "extract"
        payload = run_stage(
            "extract",
            lambda: parsers.extract_primary_text_payload(
                f"{case['case_id']}.pdf",
                case["pdf_bytes"],
                content_type="application/pdf",
                ingest_type=ingest_type,
            ),
        )

        terminal_stage = "sanitize"

        def sanitize() -> str:
            text = sanitize_db_text(payload.full_text or "").strip()
            if not text:
                raise _EmptyPrimaryTextError
            return text

        sanitized_text = run_stage("sanitize", sanitize)
        terminal_stage = "chunk_plan"
        chunk_specs = run_stage(
            "chunk_plan",
            lambda: plan_text_chunks(
                sanitized_text, chunk_size=chunk_size, overlap=overlap
            ),
        )
    except Exception as exc:
        mark_terminal()
        return _build_pipeline_execution(
            success=False,
            terminal_stage=terminal_stage,
            exception=exc,
            sanitized_text=None,
            chunk_specs=None,
        )
    mark_terminal()
    return _build_pipeline_execution(
        success=True,
        terminal_stage="complete",
        exception=None,
        sanitized_text=sanitized_text,
        chunk_specs=chunk_specs,
    )


def _finalize_pipeline_execution(execution: Mapping[str, object], case: dict) -> dict:
    success = execution["success"]
    diagnostic_code = (
        "ok"
        if success
        else _diagnostic_for(execution["_exception"], page_count=case["page_count"])
    )
    return {
        "success": success,
        "diagnostic_code": diagnostic_code,
        "terminal_stage": execution["terminal_stage"],
        "_sanitized_text": execution["_sanitized_text"],
        "_chunk_specs": execution["_chunk_specs"],
    }


def run_document_case(
    case: dict,
    *,
    chunk_size: int,
    overlap: int,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> dict:
    """Execute and directly time one detect/extract/sanitize/chunk-plan attempt."""

    observation = _observe_pipeline(
        case, chunk_size=chunk_size, overlap=overlap, clock=clock
    )
    sanitized_text = observation.pop("_sanitized_text")
    chunk_specs = observation.pop("_chunk_specs")
    success = observation["success"]
    diagnostic_code = observation["diagnostic_code"]

    if not success and case["arm"] == "synthetic":
        raise BenchmarkIntegrityError(
            f"synthetic benchmark integrity error: {diagnostic_code}"
        )

    static_record = build_document_record(
        arm=case["arm"],
        case_id=case["case_id"],
        input_bytes=len(case["pdf_bytes"]),
        page_count=case["page_count"],
        success=success,
        diagnostic_code=diagnostic_code,
        sanitized_text=sanitized_text,
        chunk_specs=chunk_specs,
    )
    return {"static_record": static_record, "timing": observation}


def _observe_pipeline(
    case: dict,
    *,
    chunk_size: int,
    overlap: int,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> dict:
    """Run only the measured pipeline and retain transient outputs for its caller."""

    timings: dict[str, int | None] = {
        "detect_ns": None,
        "extract_ns": None,
        "sanitize_ns": None,
        "chunk_plan_ns": None,
    }

    def time_stage(stage: str, operation: Callable[[], object]) -> object:
        stage_start = clock()
        try:
            return operation()
        finally:
            timings[f"{stage}_ns"] = clock() - stage_start

    combined_start = clock()
    combined_end = None

    def mark_terminal() -> None:
        nonlocal combined_end
        combined_end = clock()

    execution = _execute_pipeline_core(
        case,
        chunk_size=chunk_size,
        overlap=overlap,
        stage_runner=time_stage,
        terminal_callback=mark_terminal,
    )
    combined_ns = combined_end - combined_start
    outcome = _finalize_pipeline_execution(execution, case)

    return {
        **outcome,
        **timings,
        "combined_ns": combined_ns,
    }


def _rate(numerator: int, denominator_ns: int) -> float | None:
    return numerator * 1_000_000_000 / denominator_ns if denominator_ns else None


def _build_observed_static(case: dict, observation: Mapping[str, object]) -> dict:
    return build_document_record(
        arm=case["arm"],
        case_id=case["case_id"],
        input_bytes=len(case["pdf_bytes"]),
        page_count=case["page_count"],
        success=observation["success"],
        diagnostic_code=observation["diagnostic_code"],
        sanitized_text=observation.get("_sanitized_text"),
        chunk_specs=observation.get("_chunk_specs"),
    )


def _require_static_match(
    case: dict,
    observation: Mapping[str, object],
    frozen_static: Mapping[str, object],
    *,
    phase: str,
) -> None:
    observed_static = _build_observed_static(case, observation)
    if observed_static != frozen_static:
        raise BenchmarkIntegrityError(f"{phase} output drift from static record")


def _sweep_record(
    arm: str,
    sweep_index: int,
    ordered_case_ids: list[str],
    case_rows: list[dict],
    static_by_identity: dict[tuple[str, str], dict],
) -> dict:
    successful_rows = [row for row in case_rows if row["success"]]
    successful_static = [
        static_by_identity[(arm, row["case_id"])] for row in successful_rows
    ]
    combined_ns = sum(row["combined_ns"] for row in case_rows)
    successful_ns = sum(row["combined_ns"] for row in successful_rows)
    success_count = len(successful_rows)
    attempted_count = len(case_rows)
    input_bytes = sum(row["input_bytes"] for row in successful_static)
    page_count = sum(row["page_count"] for row in successful_static)
    codepoints = sum(row["extracted_codepoints"] for row in successful_static)
    estimated_tokens = sum(row["estimated_tokens"] for row in successful_static)
    return {
        "schema_version": SCHEMA_VERSION,
        "arm": arm,
        "sweep_index": sweep_index,
        "ordered_case_ids": ordered_case_ids,
        "attempted_count": attempted_count,
        "success_count": success_count,
        "failure_count": attempted_count - success_count,
        "successful_input_bytes": input_bytes,
        "successful_page_count": page_count,
        "successful_extracted_codepoints": codepoints,
        "successful_estimated_tokens": estimated_tokens,
        "case_combined_sum_ns": combined_ns,
        "successful_case_combined_sum_ns": successful_ns,
        "attempted_documents_per_second": _rate(attempted_count, combined_ns),
        "effective_successful_documents_per_second": _rate(success_count, combined_ns),
        "effective_pages_per_second": _rate(page_count, combined_ns),
        "effective_mib_per_second": _rate(input_bytes, combined_ns) / 1_048_576
        if combined_ns
        else None,
        "effective_codepoints_per_second": _rate(codepoints, combined_ns),
        "effective_estimated_tokens_per_second": _rate(estimated_tokens, combined_ns),
        "success_conditioned_documents_per_second": _rate(success_count, successful_ns),
        "success_conditioned_pages_per_second": _rate(page_count, successful_ns),
        "success_conditioned_mib_per_second": _rate(input_bytes, successful_ns)
        / 1_048_576
        if successful_ns
        else None,
        "success_conditioned_codepoints_per_second": _rate(codepoints, successful_ns),
        "success_conditioned_estimated_tokens_per_second": _rate(
            estimated_tokens, successful_ns
        ),
        "milliseconds_per_attempted_document": (
            combined_ns / attempted_count / 1_000_000 if attempted_count else None
        ),
        "milliseconds_per_successful_document": (
            successful_ns / success_count / 1_000_000 if success_count else None
        ),
    }


def _run_timing_sweeps(
    cases_by_arm: dict[str, list[dict]],
    static_records: list[dict],
    *,
    sweeps: int,
    chunk_size: int,
    overlap: int,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> tuple[list[dict], list[dict]]:
    """Warm each arm independently, then collect rotating sweep observations."""

    static_by_identity = {(row["arm"], row["case_id"]): row for row in static_records}
    timing_cases: list[dict] = []
    timing_sweeps: list[dict] = []
    for arm in ("real", "synthetic"):
        base = sorted(cases_by_arm.get(arm, []), key=lambda case: case["case_id"])
        if not base:
            continue
        for sweep_index in range(sweeps):
            offset = sweep_index % len(base)
            ordered = base[offset:] + base[:offset]
            sweep_case_rows = []
            for order_index, case in enumerate(ordered):
                observation = _observe_pipeline(
                    case, chunk_size=chunk_size, overlap=overlap, clock=clock
                )
                static = static_by_identity[(arm, case["case_id"])]
                _require_static_match(case, observation, static, phase="timing")
                observation = {
                    key: value
                    for key, value in observation.items()
                    if not key.startswith("_")
                }
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "arm": arm,
                    "case_id": case["case_id"],
                    "sweep_index": sweep_index,
                    "order_index": order_index,
                    **observation,
                    "input_bytes": static["input_bytes"],
                    "page_count": static["page_count"],
                    "extracted_codepoints": static["extracted_codepoints"],
                    "estimated_tokens": static["estimated_tokens"],
                }
                timing_cases.append(row)
                sweep_case_rows.append(row)
            timing_sweeps.append(
                _sweep_record(
                    arm,
                    sweep_index,
                    [case["case_id"] for case in ordered],
                    sweep_case_rows,
                    static_by_identity,
                )
            )
    return timing_cases, timing_sweeps


def _run_pipeline_unmeasured(
    case: dict,
    *,
    chunk_size: int,
    overlap: int,
) -> dict:
    """Execute the shared production core without timing or finalization."""

    return _execute_pipeline_core(case, chunk_size=chunk_size, overlap=overlap)


def _run_memory_passes(
    cases_by_arm: dict[str, list[dict]],
    static_records: list[dict],
    *,
    memory_repeats: int,
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    """Collect isolated incremental Python allocation peaks for every attempt."""

    if tracemalloc.is_tracing():
        raise BenchmarkIntegrityError("tracemalloc is already active")
    records = []
    static_by_identity = {(row["arm"], row["case_id"]): row for row in static_records}
    for arm in ("real", "synthetic"):
        cases = sorted(cases_by_arm.get(arm, []), key=lambda case: case["case_id"])
        for case in cases:
            for memory_index in range(memory_repeats):
                gc.collect()
                tracemalloc.start()
                try:
                    tracemalloc.reset_peak()
                    execution = _run_pipeline_unmeasured(
                        case, chunk_size=chunk_size, overlap=overlap
                    )
                    _, peak = tracemalloc.get_traced_memory()
                finally:
                    tracemalloc.stop()
                outcome = _finalize_pipeline_execution(execution, case)
                static = static_by_identity[(arm, case["case_id"])]
                _require_static_match(case, outcome, static, phase="memory")
                if arm == "synthetic" and outcome["success"] is not True:
                    raise BenchmarkIntegrityError("synthetic memory pass failed")
                records.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "arm": arm,
                        "case_id": case["case_id"],
                        "memory_index": memory_index,
                        "success": outcome["success"],
                        "diagnostic_code": outcome["diagnostic_code"],
                        "peak_python_traced_bytes": peak,
                    }
                )
    return records


def _prepare_synthetic_cases(review: Mapping[str, object]) -> list[dict]:
    """Generate and independently check all reviewed deterministic PDF cases."""

    protocol = review.get("protocol")
    entries = protocol.get("cases") if isinstance(protocol, Mapping) else None
    if not isinstance(entries, list):
        raise BenchmarkIntegrityError("synthetic protocol cases are missing")
    cases = []
    for index, entry in enumerate(entries, 1):
        if not isinstance(entry, Mapping):
            raise BenchmarkIntegrityError("synthetic protocol case is invalid")
        page_count = entry.get("page_count")
        pdf_bytes, _ = generate_synthetic_pdf(page_count)
        raw_hash = hashlib.sha256(pdf_bytes).hexdigest()
        if raw_hash != entry.get("pdf_sha256"):
            raise BenchmarkIntegrityError("synthetic PDF hash mismatch")
        observed_pages = _page_count(pdf_bytes)
        if observed_pages != entry.get("expected_page_count"):
            raise BenchmarkIntegrityError("synthetic page count mismatch")
        cases.append(
            {
                "arm": "synthetic",
                "case_id": f"synthetic-{index:03d}",
                "pdf_bytes": pdf_bytes,
                "raw_sha256": raw_hash,
                "page_count": observed_pages,
                "expected_output_sha256": entry.get("normalized_text_sha256"),
            }
        )
    return cases


def _initialize_dependencies() -> None:
    """Resolve lazy imports and initialize the production estimator before work."""

    _ = parsers.detect_ingest_type
    _ = parsers.extract_primary_text_payload
    _ = plan_text_chunks
    rag_evidence._estimate_tokens("")


def _source_state() -> dict[str, object]:
    project_dir = Path(__file__).resolve().parents[4]
    try:
        commit = subprocess.run(
            ["git", "-C", str(project_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(project_dir), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BenchmarkIntegrityError("source revision state is unavailable") from exc
    return {"commit": commit, "dirty": dirty}


def _source_hashes() -> dict[str, str]:
    paths = {
        "document_pipeline_artifacts": Path(__file__).with_name(
            "document_pipeline_artifacts.py"
        ),
        "document_pipeline_runner": Path(__file__),
        "document_pipeline_schema": Path(__file__).with_name(
            "document_pipeline_schema.py"
        ),
        "ingestion_parsers": Path(parsers.__file__),
        "network_guard": Path(__file__).with_name("network.py"),
        "run_offline_evidence": Path(__file__).parent.parent
        / "run_offline_evidence.py",
        "text_chunk_plan": Path(sys.modules[plan_text_chunks.__module__].__file__),
    }
    return {name: sha256_canonical_text(path) for name, path in sorted(paths.items())}


def _validate_network_audit(network_audit: Mapping[str, object]) -> dict:
    required_keys = {"guard", "scope", "total_attempts", "details"}
    if not isinstance(network_audit, Mapping) or set(network_audit) != required_keys:
        raise BenchmarkIntegrityError("network audit keys are invalid")
    if network_audit["guard"] != "deny_network":
        raise BenchmarkIntegrityError("network audit guard identity is invalid")
    if network_audit["scope"] != "fixture_generation_through_artifact_validation":
        raise BenchmarkIntegrityError("network audit scope identity is invalid")
    total_attempts = network_audit["total_attempts"]
    if type(total_attempts) is not int or total_attempts < 0:
        raise BenchmarkIntegrityError("network audit total type is invalid")
    details = network_audit["details"]
    if not isinstance(details, list):
        raise BenchmarkIntegrityError("network audit details type is invalid")
    allowed_operations = {
        "socket.socket.connect",
        "socket.socket.connect_ex",
        "socket.create_connection",
    }
    for detail in details:
        if (
            not isinstance(detail, Mapping)
            or set(detail) != {"operation"}
            or detail["operation"] not in allowed_operations
        ):
            raise BenchmarkIntegrityError("network audit detail is invalid or unsafe")
    if total_attempts != len(details):
        raise BenchmarkIntegrityError("network audit total does not match details")
    if total_attempts:
        raise BenchmarkIntegrityError("network audit observed a connection attempt")
    return {
        "guard": network_audit["guard"],
        "scope": network_audit["scope"],
        "total_attempts": total_attempts,
        "details": [dict(detail) for detail in details],
    }


def _validate_benchmark_config(
    *, sweeps: int, memory_repeats: int, chunk_size: int, overlap: int
) -> None:
    for field, value in (
        ("sweeps", sweeps),
        ("memory_repeats", memory_repeats),
        ("chunk_size", chunk_size),
    ):
        if type(value) is not int or value <= 0:
            raise BenchmarkIntegrityError(
                f"benchmark configuration {field} must be a positive integer"
            )
    if type(overlap) is not int or overlap < 0 or overlap >= chunk_size:
        raise BenchmarkIntegrityError(
            "benchmark configuration overlap must be an integer in [0, chunk_size)"
        )


def _load_benchmark_inputs(inventory_path: Path, review_path: Path) -> dict:
    inventory_path = Path(inventory_path)
    review_path = Path(review_path)
    inventory = load_document_inventory(inventory_path)
    review = load_document_review(review_path, inventory_path)
    return {
        "inventory": inventory,
        "review": review,
        "inventory_source_hash": sha256_canonical_text(inventory_path),
        "review_source_hash": sha256_canonical_text(review_path),
    }


def _build_manifest(
    *,
    inventory: Mapping[str, object],
    review: Mapping[str, object],
    inventory_source_hash: str,
    review_source_hash: str,
    synthetic_cases: list[dict],
    network_audit: Mapping[str, object],
    chunk_size: int,
    overlap: int,
    sweeps: int,
    memory_repeats: int,
) -> dict:
    dependency_versions = {}
    for distribution in ("Django", "pypdf", "psutil"):
        try:
            dependency_versions[distribution.casefold()] = metadata.version(
                distribution
            )
        except metadata.PackageNotFoundError:
            dependency_versions[distribution.casefold()] = "unavailable"
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source": _source_state(),
        "hashes": {
            "source_code": {
                "algorithm": "sha256-utf8-lf-v1",
                "values": _source_hashes(),
            },
            "inventory_source": {
                "algorithm": "sha256-utf8-lf-v1",
                "sha256": inventory_source_hash,
            },
            "review_source": {
                "algorithm": "sha256-utf8-lf-v1",
                "sha256": review_source_hash,
            },
            "synthetic_protocol": {
                "algorithm": review["protocol_hash_algorithm"],
                "sha256": review["protocol_hash"],
            },
        },
        "synthetic_inputs": [
            {
                "case_id": case["case_id"],
                "pdf_sha256": case["raw_sha256"],
                "pdf_hash_algorithm": "sha256-raw-bytes-v1",
                "expected_output_sha256": case["expected_output_sha256"],
                "expected_output_hash_algorithm": "sha256-utf8-bytes-v1",
                "page_count": case["page_count"],
            }
            for case in synthetic_cases
        ],
        "chunk_configuration": {
            "chunk_size_codepoints": chunk_size,
            "overlap_codepoints": overlap,
            "pitch_codepoints": chunk_size - overlap,
        },
        "token_estimator": {
            "name": "production_character_estimator",
            "algorithm": "max(1, len(text) // 4)",
        },
        "dependencies": dependency_versions,
        "environment": {
            "os": platform.system(),
            "os_release": platform.release(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "logical_cpu_count": os.cpu_count(),
            "cpu": platform.processor() or "unknown",
            "total_system_ram_bytes": psutil.virtual_memory().total,
            "process_bits": struct.calcsize("P") * 8,
            "timer": "time.perf_counter_ns",
            "timer_resolution_seconds": time.get_clock_info("perf_counter").resolution,
        },
        "execution": {
            "mode": "single_thread_sequential",
            "warmups_per_arm": 1,
            "warmup_role": "static_record_pass",
            "timing_sweeps_per_arm": sweeps,
            "memory_repeats_per_case": memory_repeats,
        },
        "network_audit": dict(network_audit),
        "network_audit_statement": (
            "zero connection attempts observed through the configured process-local "
            "socket guard"
        ),
    }


def run_document_benchmark(
    corpus_dir: Path,
    inventory_path: Path,
    review_path: Path,
    *,
    network_audit: Mapping[str, object],
    sweeps: int = 30,
    memory_repeats: int = 3,
    chunk_size: int = 2048,
    overlap: int = 384,
    allow_unlisted_pdfs: bool = False,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> dict:
    """Prepare, measure, aggregate, and return the canonical in-memory result."""

    _validate_benchmark_config(
        sweeps=sweeps,
        memory_repeats=memory_repeats,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    validated_audit = _validate_network_audit(network_audit)
    loaded_inputs = _load_benchmark_inputs(inventory_path, review_path)
    inventory = loaded_inputs["inventory"]
    review = loaded_inputs["review"]
    real_cases = resolve_real_corpus(
        corpus_dir, inventory, allow_unlisted_pdfs=allow_unlisted_pdfs
    )
    synthetic_cases = _prepare_synthetic_cases(review)
    cases_by_arm = {"real": real_cases, "synthetic": synthetic_cases}
    _initialize_dependencies()

    static_records = []
    timing_cases = []
    timing_sweeps = []
    for arm in ("real", "synthetic"):
        arm_static_records = []
        for case in sorted(cases_by_arm[arm], key=lambda value: value["case_id"]):
            result = run_document_case(
                case, chunk_size=chunk_size, overlap=overlap, clock=clock
            )
            static = result["static_record"]
            if arm == "synthetic" and (
                static["output_sha256"] != case["expected_output_sha256"]
            ):
                raise BenchmarkIntegrityError("synthetic extracted text hash mismatch")
            static_records.append(static)
            arm_static_records.append(static)
        arm_timing_cases, arm_timing_sweeps = _run_timing_sweeps(
            {arm: cases_by_arm[arm]},
            arm_static_records,
            sweeps=sweeps,
            chunk_size=chunk_size,
            overlap=overlap,
            clock=clock,
        )
        timing_cases.extend(arm_timing_cases)
        timing_sweeps.extend(arm_timing_sweeps)
    memory_records = _run_memory_passes(
        cases_by_arm,
        static_records,
        memory_repeats=memory_repeats,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    aggregate = aggregate_document_results(
        static_records,
        timing_sweeps,
        memory_records,
        network_audit=validated_audit,
    )
    manifest = _build_manifest(
        inventory=inventory,
        review=review,
        inventory_source_hash=loaded_inputs["inventory_source_hash"],
        review_source_hash=loaded_inputs["review_source_hash"],
        synthetic_cases=synthetic_cases,
        network_audit=validated_audit,
        chunk_size=chunk_size,
        overlap=overlap,
        sweeps=sweeps,
        memory_repeats=memory_repeats,
    )
    return {
        "manifest": manifest,
        "inventory": inventory,
        "review": review,
        "real_records": [row for row in static_records if row["arm"] == "real"],
        "synthetic_records": [
            row for row in static_records if row["arm"] == "synthetic"
        ],
        "timing_cases": timing_cases,
        "timing_sweeps": timing_sweeps,
        "memory_records": memory_records,
        "aggregate": aggregate,
    }


__all__ = [
    "BenchmarkIntegrityError",
    "resolve_real_corpus",
    "run_document_case",
    "run_document_benchmark",
]
