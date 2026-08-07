"""Warm, in-memory measurement runner for the document preprocessing pipeline."""

from __future__ import annotations

import gc
import hashlib
import io
import os
import platform
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
    validate_document_review,
)
from apps.chat.evals.offline.schema import canonical_json_bytes, sha256_canonical_text
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
        if not path.is_file() or path.suffix.casefold() != ".pdf":
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
    if isinstance(exc, FileNotDecryptedError):
        return "encrypted_pdf"
    if isinstance(exc, PdfReadError) or page_count is None:
        return "invalid_pdf"
    return "parser_error"


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
    sanitized_text = None
    chunk_specs = None
    success = False
    diagnostic_code = "parser_error"
    terminal_stage = "detect"
    combined_start = clock()
    try:
        stage_start = clock()
        try:
            ingest_type = parsers.detect_ingest_type(
                f"{case['case_id']}.pdf", "application/pdf"
            )
        finally:
            timings["detect_ns"] = clock() - stage_start

        terminal_stage = "extract"
        stage_start = clock()
        try:
            payload = parsers.extract_primary_text_payload(
                f"{case['case_id']}.pdf",
                case["pdf_bytes"],
                content_type="application/pdf",
                ingest_type=ingest_type,
            )
        finally:
            timings["extract_ns"] = clock() - stage_start

        terminal_stage = "sanitize"
        stage_start = clock()
        try:
            sanitized_text = sanitize_db_text(payload.full_text or "").strip()
            if not sanitized_text:
                diagnostic_code = "empty_primary_text"
                raise ValueError("empty primary text")
        finally:
            timings["sanitize_ns"] = clock() - stage_start

        terminal_stage = "chunk_plan"
        stage_start = clock()
        try:
            chunk_specs = plan_text_chunks(
                sanitized_text, chunk_size=chunk_size, overlap=overlap
            )
        finally:
            timings["chunk_plan_ns"] = clock() - stage_start
        combined_end = clock()
        success = True
        diagnostic_code = "ok"
        terminal_stage = "complete"
    except Exception as exc:
        combined_end = clock()
        if diagnostic_code != "empty_primary_text":
            diagnostic_code = _diagnostic_for(exc, page_count=case["page_count"])
        sanitized_text = None
        chunk_specs = None
    combined_ns = combined_end - combined_start

    return {
        "success": success,
        "diagnostic_code": diagnostic_code,
        "terminal_stage": terminal_stage,
        **timings,
        "combined_ns": combined_ns,
        "_sanitized_text": sanitized_text,
        "_chunk_specs": chunk_specs,
    }


def _rate(numerator: int, denominator_ns: int) -> float | None:
    return numerator * 1_000_000_000 / denominator_ns if denominator_ns else None


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
        for case in base:
            warmup = _observe_pipeline(
                case, chunk_size=chunk_size, overlap=overlap, clock=clock
            )
            if arm == "synthetic" and warmup["success"] is not True:
                raise BenchmarkIntegrityError("synthetic warm-up failed")
        for sweep_index in range(sweeps):
            offset = sweep_index % len(base)
            ordered = base[offset:] + base[:offset]
            sweep_case_rows = []
            for order_index, case in enumerate(ordered):
                observation = _observe_pipeline(
                    case, chunk_size=chunk_size, overlap=overlap, clock=clock
                )
                observation = {
                    key: value
                    for key, value in observation.items()
                    if not key.startswith("_")
                }
                static = static_by_identity[(arm, case["case_id"])]
                if (
                    observation["success"] != static["success"]
                    or observation["diagnostic_code"] != static["diagnostic_code"]
                ):
                    raise BenchmarkIntegrityError(
                        "timing outcome differs from validated static record"
                    )
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
    """Execute the production stages without timing or post-processing metrics."""

    diagnostic_code = "parser_error"
    try:
        ingest_type = parsers.detect_ingest_type(
            f"{case['case_id']}.pdf", "application/pdf"
        )
        payload = parsers.extract_primary_text_payload(
            f"{case['case_id']}.pdf",
            case["pdf_bytes"],
            content_type="application/pdf",
            ingest_type=ingest_type,
        )
        sanitized_text = sanitize_db_text(payload.full_text or "").strip()
        if not sanitized_text:
            return {"success": False, "diagnostic_code": "empty_primary_text"}
        plan_text_chunks(sanitized_text, chunk_size=chunk_size, overlap=overlap)
        return {"success": True, "diagnostic_code": "ok"}
    except Exception as exc:
        diagnostic_code = _diagnostic_for(exc, page_count=case["page_count"])
        return {"success": False, "diagnostic_code": diagnostic_code}


def _run_memory_passes(
    cases_by_arm: dict[str, list[dict]],
    *,
    memory_repeats: int,
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    """Collect isolated incremental Python allocation peaks for every attempt."""

    records = []
    for arm in ("real", "synthetic"):
        cases = sorted(cases_by_arm.get(arm, []), key=lambda case: case["case_id"])
        for case in cases:
            for memory_index in range(memory_repeats):
                gc.collect()
                tracemalloc.start()
                try:
                    tracemalloc.reset_peak()
                    outcome = _run_pipeline_unmeasured(
                        case, chunk_size=chunk_size, overlap=overlap
                    )
                    _, peak = tracemalloc.get_traced_memory()
                finally:
                    tracemalloc.stop()
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


def _mapping_hash(value: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _source_hashes() -> dict[str, str]:
    paths = {
        "document_pipeline_runner": Path(__file__),
        "document_pipeline_schema": Path(__file__).with_name(
            "document_pipeline_schema.py"
        ),
        "ingestion_parsers": Path(parsers.__file__),
        "text_chunk_plan": Path(sys.modules[plan_text_chunks.__module__].__file__),
    }
    return {name: sha256_canonical_text(path) for name, path in sorted(paths.items())}


def _normalize_network_audit(network_audit: Mapping[str, object]) -> dict:
    normalized = dict(network_audit)
    if normalized.get("total_attempts") == 0:
        normalized["scope"] = (
            "zero connection attempts observed through the configured process-local "
            "socket guard"
        )
    return normalized


def _build_manifest(
    *,
    inventory: Mapping[str, object],
    review: Mapping[str, object],
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
            "algorithm": "sha256-utf8-lf-v1",
            "source_code": _source_hashes(),
            "inventory_source": review.get("inventory_hash")
            or _mapping_hash(inventory),
            "review_config": _mapping_hash(review),
            "synthetic_protocol": review.get("protocol_hash")
            or _mapping_hash(review.get("protocol", {})),
        },
        "synthetic_inputs": [
            {
                "case_id": case["case_id"],
                "pdf_sha256": case["raw_sha256"],
                "expected_output_sha256": case["expected_output_sha256"],
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
            "timing_sweeps_per_arm": sweeps,
            "memory_repeats_per_case": memory_repeats,
        },
        "network_audit": dict(network_audit),
    }


def run_document_benchmark(
    corpus_dir: Path,
    inventory: dict,
    review: dict,
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

    validate_document_review(review, inventory)
    real_cases = resolve_real_corpus(
        corpus_dir, inventory, allow_unlisted_pdfs=allow_unlisted_pdfs
    )
    synthetic_cases = _prepare_synthetic_cases(review)
    cases_by_arm = {"real": real_cases, "synthetic": synthetic_cases}
    _initialize_dependencies()

    static_records = []
    for arm in ("real", "synthetic"):
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

    timing_cases, timing_sweeps = _run_timing_sweeps(
        cases_by_arm,
        static_records,
        sweeps=sweeps,
        chunk_size=chunk_size,
        overlap=overlap,
        clock=clock,
    )
    memory_records = _run_memory_passes(
        cases_by_arm,
        memory_repeats=memory_repeats,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    normalized_audit = _normalize_network_audit(network_audit)
    aggregate = aggregate_document_results(
        static_records,
        timing_sweeps,
        memory_records,
        network_audit=normalized_audit,
    )
    manifest = _build_manifest(
        inventory=inventory,
        review=review,
        synthetic_cases=synthetic_cases,
        network_audit=normalized_audit,
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
