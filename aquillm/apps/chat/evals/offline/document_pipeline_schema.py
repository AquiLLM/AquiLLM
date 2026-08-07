"""Frozen-input and absolute-metric contracts for the document benchmark."""

from __future__ import annotations

import hashlib
import math
import re
import statistics
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path, PurePosixPath, PureWindowsPath

import yaml

from apps.chat.evals.offline.schema import canonical_json_bytes, sha256_canonical_text
from apps.chat.services import rag_evidence

SCHEMA_VERSION = "1.0"
INVENTORY_ID = "astro-test-real-v1"
RAW_HASH_ALGORITHM = "sha256-raw-bytes-v1"
SOURCE_HASH_ALGORITHM = "sha256-utf8-lf-v1"
PROTOCOL_HASH_ALGORITHM = "sha256-canonical-json-v1"
REAL_CASE_COUNT = 17
REAL_TOTAL_BYTES = 97_006_698

_RATIONALE = "all PDF members of the fixed astro_test convenience set"
_LINEAGE = "existing Semantic Extraction Experiment astro_test set"
_SENSITIVITY = "public-paper-like local research corpus"
_LICENSE = "not redistributed; source license not asserted"
_INVENTORY_KEYS = {
    "schema_version",
    "inventory_id",
    "source_hash_algorithm",
    "case_count",
    "total_bytes",
    "cases",
}
_CASE_KEYS = {
    "case_id",
    "sha256",
    "input_bytes",
    "selection_rationale",
    "acquisition_lineage",
    "sensitivity",
    "redistribution_license_status",
}
_REVIEW_KEYS = {
    "schema_version",
    "review_id",
    "status",
    "source_hash_algorithm",
    "inventory_hash",
    "protocol_hash_algorithm",
    "protocol_hash",
    "protocol",
    "reviewer_identity",
    "reviewer_role",
    "review_date",
    "decisions",
}
_PROTOCOL_KEYS = {
    "generator_version",
    "page_counts",
    "page_string_template",
    "cases",
}
_PROTOCOL_CASE_KEYS = {
    "page_count",
    "pdf_sha256",
    "normalized_text_sha256",
    "expected_page_count",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_AGENT_ID_RE = re.compile(r"codex-agent:[a-z0-9][a-z0-9._:/-]*\Z")
_PROHIBITED_KEYS = {"filename", "basename", "path", "title", "content", "text"}
_SYNTHETIC_PAGE_TEMPLATE = (
    "AquiLLM synthetic preprocessing page NNNN. The quick brown fox jumps over "
    "the lazy dog. Value NNNN."
)
_ARMS = ("real", "synthetic")
_DIAGNOSTIC_CODES = (
    "ok",
    "invalid_pdf",
    "encrypted_pdf",
    "empty_primary_text",
    "parser_error",
)
_OUTPUT_FIELDS = (
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
_TIMING_UNITS = {
    "case_combined_sum_ns": "nanoseconds",
    "successful_case_combined_sum_ns": "nanoseconds",
    "attempted_documents_per_second": "documents_per_second",
    "effective_successful_documents_per_second": "documents_per_second",
    "effective_pages_per_second": "pages_per_second",
    "effective_mib_per_second": "mebibytes_per_second",
    "effective_codepoints_per_second": "codepoints_per_second",
    "effective_estimated_tokens_per_second": "estimated_tokens_per_second",
    "success_conditioned_documents_per_second": "documents_per_second",
    "success_conditioned_pages_per_second": "pages_per_second",
    "success_conditioned_mib_per_second": "mebibytes_per_second",
    "success_conditioned_codepoints_per_second": "codepoints_per_second",
    "success_conditioned_estimated_tokens_per_second": "estimated_tokens_per_second",
    "milliseconds_per_attempted_document": "milliseconds_per_document",
    "milliseconds_per_successful_document": "milliseconds_per_document",
}
_EXCLUDED_CLAIMS = [
    "No vector-embedding or vector-index insertion claim.",
    "No PostgreSQL persistence or full document-ingestion/indexing claim.",
    "No figure-extraction, OCR, model-inference, or retrieval claim.",
    "No concurrent-user, disk cold-cache, end-to-end response-latency, GPU, or "
    "RSS claim.",
    "No population-representativeness or cross-hardware generalization claim.",
]


def _require_exact_keys(
    value: Mapping[str, object], expected: set[str], label: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} keys do not match the required schema")


def _looks_like_private_path(value: str) -> bool:
    return (
        PureWindowsPath(value).is_absolute()
        or PurePosixPath(value).is_absolute()
        or bool(re.search(r"(?i)(?:^|[/\\])users[/\\][^/\\]+", value))
        or bool(re.search(r"(?i)(?:^|[/\\])home[/\\][^/\\]+", value))
    )


def _validate_privacy(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("privacy validation requires string field names")
            if key.casefold() in _PROHIBITED_KEYS:
                raise ValueError(f"privacy-prohibited field: {key}")
            _validate_privacy(child)
    elif isinstance(value, list):
        for child in value:
            _validate_privacy(child)
    elif isinstance(value, str) and _looks_like_private_path(value):
        raise ValueError("private or absolute path value is prohibited")


def _load_yaml_without_bom(path: Path, label: str) -> dict:
    raw = Path(path).read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"{label} must not begin with a UTF-8 BOM")
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} must be UTF-8") from exc
    data = yaml.safe_load(decoded)
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a mapping")
    return data


def load_document_inventory(path: Path) -> dict:
    """Load and validate a frozen, path-free real-corpus inventory."""

    data = _load_yaml_without_bom(Path(path), "document inventory")
    validate_document_inventory(data)
    return data


def validate_document_inventory(data: dict) -> None:
    """Validate the exact frozen inventory contract."""

    if not isinstance(data, dict):
        raise ValueError("document inventory must be a mapping")
    _require_exact_keys(data, _INVENTORY_KEYS, "document inventory")
    _validate_privacy(data)
    if data["schema_version"] != SCHEMA_VERSION:
        raise ValueError("schema_version must be 1.0")
    if data["inventory_id"] != INVENTORY_ID:
        raise ValueError(f"inventory_id must be {INVENTORY_ID}")
    if data["source_hash_algorithm"] != RAW_HASH_ALGORITHM:
        raise ValueError(f"source_hash_algorithm must be {RAW_HASH_ALGORITHM}")
    if data["case_count"] != REAL_CASE_COUNT:
        raise ValueError("case_count must be exactly 17")
    if data["total_bytes"] != REAL_TOTAL_BYTES:
        raise ValueError(f"total_bytes must be exactly {REAL_TOTAL_BYTES}")
    cases = data["cases"]
    if not isinstance(cases, list) or len(cases) != REAL_CASE_COUNT:
        raise ValueError("inventory must contain exactly 17 cases")

    seen_hashes: set[str] = set()
    hashes: list[str] = []
    for index, case in enumerate(cases, 1):
        if not isinstance(case, dict):
            raise ValueError(f"case {index} must be a mapping")
        _require_exact_keys(case, _CASE_KEYS, f"case {index}")
        if case["case_id"] != f"real-{index:03d}":
            raise ValueError("case_id sequence must be real-001 through real-017")
        raw_hash = case["sha256"]
        if not isinstance(raw_hash, str) or not _SHA256_RE.fullmatch(raw_hash):
            raise ValueError("sha256 must be lowercase raw-PDF SHA-256")
        if raw_hash in seen_hashes:
            raise ValueError("duplicate PDF sha256")
        seen_hashes.add(raw_hash)
        hashes.append(raw_hash)
        if type(case["input_bytes"]) is not int or case["input_bytes"] <= 0:
            raise ValueError("input_bytes must be a positive integer")
        for key, expected in (
            ("selection_rationale", _RATIONALE),
            ("acquisition_lineage", _LINEAGE),
            ("sensitivity", _SENSITIVITY),
            ("redistribution_license_status", _LICENSE),
        ):
            if case[key] != expected:
                raise ValueError(f"{key} must use the fixed inventory wording")
    if hashes != sorted(hashes):
        raise ValueError("cases must be ordered by ascending raw PDF sha256")
    if sum(case["input_bytes"] for case in cases) != data["total_bytes"]:
        raise ValueError("total_bytes does not equal the case byte-size sum")


def load_document_review(
    path: Path,
    inventory_path: Path,
    *,
    allow_pending: bool = False,
) -> dict:
    """Load review metadata and bind it to the exact committed inventory bytes."""

    inventory_path = Path(inventory_path)
    inventory = load_document_inventory(inventory_path)
    data = _load_yaml_without_bom(Path(path), "document review")
    validate_document_review(data, inventory, allow_pending=allow_pending)
    if data["inventory_hash"] != sha256_canonical_text(inventory_path):
        raise ValueError("inventory_hash does not match the inventory source")
    return data


def _validate_protocol(protocol: object) -> None:
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be a mapping")
    _require_exact_keys(protocol, _PROTOCOL_KEYS, "protocol")
    if (
        not isinstance(protocol["generator_version"], str)
        or not protocol["generator_version"]
    ):
        raise ValueError("generator_version must be a non-empty string")
    if protocol["page_counts"] != [1, 10, 50, 100]:
        raise ValueError("page_counts must be [1, 10, 50, 100]")
    if protocol["page_string_template"] != (
        "AquiLLM synthetic preprocessing page NNNN. The quick brown fox jumps "
        "over the lazy dog. Value NNNN."
    ):
        raise ValueError("page_string_template must use the frozen authored string")
    cases = protocol["cases"]
    if not isinstance(cases, list) or len(cases) != 4:
        raise ValueError("protocol cases must contain four entries")
    for case, count in zip(cases, (1, 10, 50, 100), strict=True):
        if not isinstance(case, dict):
            raise ValueError("protocol case must be a mapping")
        _require_exact_keys(case, _PROTOCOL_CASE_KEYS, "protocol case")
        if case["page_count"] != count or case["expected_page_count"] != count:
            raise ValueError("protocol page count does not match the frozen sequence")
        for key in ("pdf_sha256", "normalized_text_sha256"):
            if not isinstance(case[key], str) or not _SHA256_RE.fullmatch(case[key]):
                raise ValueError(f"{key} must be a lowercase SHA-256 digest")


def validate_document_review(
    data: dict,
    inventory: dict,
    *,
    allow_pending: bool = False,
) -> None:
    """Validate pending or independently approved frozen-input review metadata."""

    validate_document_inventory(inventory)
    if not isinstance(data, dict):
        raise ValueError("document review must be a mapping")
    _require_exact_keys(data, _REVIEW_KEYS, "document review")
    _validate_privacy(data)
    if data["schema_version"] != SCHEMA_VERSION:
        raise ValueError("schema_version must be 1.0")
    if data["review_id"] != "document-corpus-and-synthetic-v1":
        raise ValueError("review_id must be document-corpus-and-synthetic-v1")
    if data["source_hash_algorithm"] != SOURCE_HASH_ALGORITHM:
        raise ValueError(f"source_hash_algorithm must be {SOURCE_HASH_ALGORITHM}")
    if not isinstance(data["inventory_hash"], str) or not _SHA256_RE.fullmatch(
        data["inventory_hash"]
    ):
        raise ValueError("inventory_hash must be a lowercase SHA-256 digest")
    if data["protocol_hash_algorithm"] != PROTOCOL_HASH_ALGORITHM:
        raise ValueError(f"protocol_hash_algorithm must be {PROTOCOL_HASH_ALGORITHM}")
    _validate_protocol(data["protocol"])
    expected_protocol_hash = hashlib.sha256(
        canonical_json_bytes(data["protocol"])
    ).hexdigest()
    if data["protocol_hash"] != expected_protocol_hash:
        raise ValueError("protocol_hash does not match canonical protocol JSON")

    status = data["status"]
    if status == "pending_independent_review":
        if not allow_pending:
            raise ValueError("document inputs require approved independent review")
        for key in ("reviewer_identity", "reviewer_role", "review_date"):
            if data[key] is not None:
                raise ValueError(f"pending {key} must be null")
        if data["decisions"] != []:
            raise ValueError("pending decisions must be empty")
    elif status == "approved":
        identity = data["reviewer_identity"]
        if not isinstance(identity, str) or not _AGENT_ID_RE.fullmatch(identity):
            raise ValueError("reviewer_identity must be a deliberate stable agent id")
        if data["reviewer_role"] != "independent_reviewer":
            raise ValueError("reviewer_role must be independent_reviewer")
        review_date = data["review_date"]
        if not isinstance(review_date, str):
            raise ValueError("review_date must be an ISO-8601 date")
        try:
            date.fromisoformat(review_date)
        except ValueError as exc:
            raise ValueError("review_date must be an ISO-8601 date") from exc
        decisions = data["decisions"]
        if (
            not isinstance(decisions, list)
            or not decisions
            or not all(
                isinstance(decision, (str, dict)) and bool(decision)
                for decision in decisions
            )
        ):
            raise ValueError("decisions must record at least one deliberate decision")
    else:
        raise ValueError(f"invalid review status: {status!r}")


def freeze_document_inventory(
    corpus_dir: Path,
    *,
    expected_count: int,
    expected_total_bytes: int,
) -> dict:
    """Hash direct-child PDFs and return a path-free frozen inventory mapping."""

    corpus_dir = Path(corpus_dir)
    members: list[tuple[str, int]] = []
    for path in corpus_dir.iterdir():
        if path.is_file() and path.suffix.casefold() == ".pdf":
            raw = path.read_bytes()
            members.append((hashlib.sha256(raw).hexdigest(), len(raw)))
    if len(members) != expected_count:
        raise ValueError(
            f"PDF count mismatch: expected {expected_count}, observed {len(members)}"
        )
    total_bytes = sum(size for _, size in members)
    if total_bytes != expected_total_bytes:
        raise ValueError(
            "PDF byte total mismatch: "
            f"expected {expected_total_bytes}, observed {total_bytes}"
        )
    if len({raw_hash for raw_hash, _ in members}) != len(members):
        raise ValueError("duplicate PDF sha256")
    members.sort()
    result = {
        "schema_version": SCHEMA_VERSION,
        "inventory_id": INVENTORY_ID,
        "source_hash_algorithm": RAW_HASH_ALGORITHM,
        "case_count": len(members),
        "total_bytes": total_bytes,
        "cases": [
            {
                "case_id": f"real-{index:03d}",
                "sha256": raw_hash,
                "input_bytes": size,
                "selection_rationale": _RATIONALE,
                "acquisition_lineage": _LINEAGE,
                "sensitivity": _SENSITIVITY,
                "redistribution_license_status": _LICENSE,
            }
            for index, (raw_hash, size) in enumerate(members, 1)
        ],
    }
    validate_document_inventory(result)
    return result


def _pdf_object(object_number: int, body: bytes) -> bytes:
    return f"{object_number} 0 obj\n".encode("ascii") + body + b"\nendobj\n"


def _escape_pdf_literal(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def generate_synthetic_pdf(page_count: int) -> tuple[bytes, str]:
    """Return exact deterministic ASCII PDF bytes and independent normalized text."""

    if type(page_count) is not int or page_count <= 0:
        raise ValueError("page_count must be a positive integer")

    page_strings = [
        _SYNTHETIC_PAGE_TEMPLATE.replace("NNNN", f"{index:04d}")
        for index in range(1, page_count + 1)
    ]
    size = 4 + (2 * page_count)
    kids = " ".join(f"{4 + (2 * index)} 0 R" for index in range(page_count))
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: (f"<< /Type /Pages /Count {page_count} /Kids [{kids}] >>".encode("ascii")),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    for index, page_text in enumerate(page_strings):
        page_object = 4 + (2 * index)
        content_object = page_object + 1
        objects[page_object] = (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Resources << /Font << /F1 3 0 R >> >> "
            f"/Contents {content_object} 0 R >>"
        ).encode("ascii")
        content_line = (
            f"BT /F1 12 Tf 72 720 Td ({_escape_pdf_literal(page_text)}) Tj ET"
        ).encode("ascii")
        objects[content_object] = (
            f"<< /Length {len(content_line)} >>\nstream\n".encode("ascii")
            + content_line
            + b"\nendstream"
        )

    output = bytearray(b"%PDF-1.4\n%AquiLLM\n")
    offsets: list[int] = []
    for object_number in range(1, size):
        offsets.append(len(output))
        output.extend(_pdf_object(object_number, objects[object_number]))

    xref_offset = len(output)
    output.extend(f"xref\n0 {size}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        (
            f"trailer\n<< /Size {size} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(output), "\n\n".join(page_strings).strip()


def _chunk_bounds(chunk: object) -> tuple[int, int]:
    if isinstance(chunk, Mapping):
        start = chunk.get("start_position")
        end = chunk.get("end_position")
    else:
        start = getattr(chunk, "start_position", None)
        end = getattr(chunk, "end_position", None)
    if type(start) is not int or type(end) is not int:
        raise ValueError("chunk specs require integer half-open positions")
    return start, end


def _span_metrics(chunk_specs: Sequence[object], text_length: int) -> dict[str, object]:
    spans = sorted(_chunk_bounds(chunk) for chunk in chunk_specs)
    for start, end in spans:
        if start < 0 or end < start or end > text_length:
            raise ValueError("chunk specs contain an invalid half-open span")
    widths = [end - start for start, end in spans]
    coverage = 0
    if spans:
        union_start, union_end = spans[0]
        for start, end in spans[1:]:
            if start <= union_end:
                union_end = max(union_end, end)
            else:
                coverage += union_end - union_start
                union_start, union_end = start, end
        coverage += union_end - union_start
    total = sum(widths)
    excess = total - coverage
    return {
        "chunk_count": len(spans),
        "coverage_codepoints": coverage,
        "total_chunk_codepoints": total,
        "excess_overlap_codepoints": excess,
        "overlap_ratio": excess / coverage if coverage else 0.0,
        "chunk_min_codepoints": min(widths) if widths else None,
        "chunk_mean_codepoints": statistics.mean(widths) if widths else None,
        "chunk_median_codepoints": statistics.median(widths) if widths else None,
        "chunk_max_codepoints": max(widths) if widths else None,
    }


def build_document_record(
    *,
    arm: str,
    case_id: str,
    input_bytes: int,
    page_count: int | None,
    success: bool,
    diagnostic_code: str,
    sanitized_text: str | None,
    chunk_specs: Sequence[object] | None,
) -> dict[str, object]:
    """Build one exact static record using absolute, reproducible units."""

    if arm not in _ARMS:
        raise ValueError(f"invalid arm: {arm!r}")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("case_id must be a non-empty string")
    if type(input_bytes) is not int or input_bytes < 0:
        raise ValueError("input_bytes must be a nonnegative integer")
    if page_count is not None and (type(page_count) is not int or page_count <= 0):
        raise ValueError("page_count must be null or a positive integer")
    if diagnostic_code not in _DIAGNOSTIC_CODES:
        raise ValueError("diagnostic_code is not an allowed safe diagnostic")
    if success != (diagnostic_code == "ok"):
        raise ValueError("success and diagnostic_code are inconsistent")

    record: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "arm": arm,
        "case_id": case_id,
        "success": success,
        "diagnostic_code": diagnostic_code,
        "input_bytes": input_bytes,
        "input_mib": input_bytes / 1_048_576,
        "page_count": page_count,
    }
    if not success:
        if sanitized_text is not None or chunk_specs is not None:
            raise ValueError("failure records must not contain output data")
        record.update({key: None for key in _OUTPUT_FIELDS})
        return record

    if not isinstance(sanitized_text, str) or not sanitized_text:
        raise ValueError("successful records require nonempty sanitized_text")
    if chunk_specs is None:
        raise ValueError("successful records require chunk_specs")
    span_metrics = _span_metrics(chunk_specs, len(sanitized_text))
    if span_metrics["coverage_codepoints"] != len(sanitized_text):
        raise ValueError("successful chunks must cover all sanitized text codepoints")
    record.update(
        {
            "extracted_codepoints": len(sanitized_text),
            "extracted_utf8_bytes": len(sanitized_text.encode("utf-8")),
            "word_count": len(sanitized_text.split()),
            "estimated_tokens": rag_evidence._estimate_tokens(sanitized_text),
            **span_metrics,
            "output_sha256": hashlib.sha256(sanitized_text.encode("utf-8")).hexdigest(),
        }
    )
    return record


def _arm_totals(records: Sequence[Mapping[str, object]], arm: str) -> dict[str, object]:
    arm_records = [record for record in records if record.get("arm") == arm]
    successes = [record for record in arm_records if record.get("success") is True]
    attempted = len(arm_records)
    success_count = len(successes)
    return {
        "attempted_count": attempted,
        "success_count": success_count,
        "failure_count": attempted - success_count,
        "success_rate": success_count / attempted if attempted else None,
        "total_input_bytes": sum(int(record["input_bytes"]) for record in successes),
        "total_page_count": sum(int(record["page_count"]) for record in successes),
        "total_extracted_codepoints": sum(
            int(record["extracted_codepoints"]) for record in successes
        ),
        "total_estimated_tokens": sum(
            int(record["estimated_tokens"]) for record in successes
        ),
        "total_chunk_count": sum(int(record["chunk_count"]) for record in successes),
        "total_coverage_codepoints": sum(
            int(record["coverage_codepoints"]) for record in successes
        ),
        "total_excess_overlap_codepoints": sum(
            int(record["excess_overlap_codepoints"]) for record in successes
        ),
    }


def _nearest_rank_p95(values: Sequence[float | int]) -> float | int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def _timing_summary(
    timing_sweeps: Sequence[Mapping[str, object]], arm: str
) -> dict[str, object]:
    rows = [row for row in timing_sweeps if row.get("arm") == arm]
    result = {}
    for metric, unit in _TIMING_UNITS.items():
        values = [row[metric] for row in rows if row.get(metric) is not None]
        result[metric] = {
            "support_sweeps": len(rows),
            "median": statistics.median(values) if values else None,
            "nearest_rank_p95": _nearest_rank_p95(values),
            "unit": unit,
        }
    return result


def _memory_summary(
    memory_records: Sequence[Mapping[str, object]], arm: str
) -> dict[str, object]:
    rows = [row for row in memory_records if row.get("arm") == arm]
    maxima: dict[str, int] = {}
    for row in rows:
        case_id = str(row["case_id"])
        peak = int(row["peak_python_traced_bytes"])
        maxima[case_id] = max(maxima.get(case_id, peak), peak)
    successful_cases = {
        str(row["case_id"]) for row in rows if row.get("success") is True
    }
    successful_maxima = [maxima[case_id] for case_id in successful_cases]
    return {
        "successful_case_count": len(successful_cases),
        "max_peak_python_traced_bytes_per_case": dict(sorted(maxima.items())),
        "max_peak_python_traced_bytes_over_successes": (
            max(successful_maxima) if successful_maxima else None
        ),
        "max_peak_python_traced_bytes_over_all_attempts": (
            max(maxima.values()) if maxima else None
        ),
    }


def _failure_summary(
    static_records: Sequence[Mapping[str, object]], arm: str
) -> dict[str, object]:
    rows = [row for row in static_records if row.get("arm") == arm]
    return {
        "attempted_count": len(rows),
        "failed_count": sum(row.get("success") is False for row in rows),
        "diagnostic_counts": {
            code: sum(row.get("diagnostic_code") == code for row in rows)
            for code in _DIAGNOSTIC_CODES
        },
    }


def aggregate_document_results(
    static_records: Sequence[Mapping[str, object]],
    timing_sweeps: Sequence[Mapping[str, object]],
    memory_records: Sequence[Mapping[str, object]],
    *,
    network_audit: Mapping[str, object],
) -> dict[str, object]:
    """Deterministically aggregate already-created static, timing, and memory rows."""

    return {
        "schema_version": SCHEMA_VERSION,
        "real": _arm_totals(static_records, "real"),
        "synthetic": _arm_totals(static_records, "synthetic"),
        "timing": {arm: _timing_summary(timing_sweeps, arm) for arm in _ARMS},
        "memory": {arm: _memory_summary(memory_records, arm) for arm in _ARMS},
        "failures": {arm: _failure_summary(static_records, arm) for arm in _ARMS},
        "network_audit": dict(network_audit),
        "excluded_claims": list(_EXCLUDED_CLAIMS),
    }


def build_pending_document_review(inventory_path: Path) -> dict:
    """Build the exact path-free pending review for frozen benchmark inputs."""

    inventory_path = Path(inventory_path)
    inventory = load_document_inventory(inventory_path)
    protocol_cases = []
    for page_count in (1, 10, 50, 100):
        pdf_bytes, normalized_text = generate_synthetic_pdf(page_count)
        protocol_cases.append(
            {
                "page_count": page_count,
                "pdf_sha256": hashlib.sha256(pdf_bytes).hexdigest(),
                "normalized_text_sha256": hashlib.sha256(
                    normalized_text.encode("utf-8")
                ).hexdigest(),
                "expected_page_count": page_count,
            }
        )
    protocol = {
        "generator_version": "aquillm-ascii-pdf-v1",
        "page_counts": [1, 10, 50, 100],
        "page_string_template": _SYNTHETIC_PAGE_TEMPLATE,
        "cases": protocol_cases,
    }
    review = {
        "schema_version": SCHEMA_VERSION,
        "review_id": "document-corpus-and-synthetic-v1",
        "status": "pending_independent_review",
        "source_hash_algorithm": SOURCE_HASH_ALGORITHM,
        "inventory_hash": sha256_canonical_text(inventory_path),
        "protocol_hash_algorithm": PROTOCOL_HASH_ALGORITHM,
        "protocol_hash": hashlib.sha256(canonical_json_bytes(protocol)).hexdigest(),
        "protocol": protocol,
        "reviewer_identity": None,
        "reviewer_role": None,
        "review_date": None,
        "decisions": [],
    }
    validate_document_review(review, inventory, allow_pending=True)
    return review


__all__ = [
    "aggregate_document_results",
    "build_document_record",
    "build_pending_document_review",
    "freeze_document_inventory",
    "generate_synthetic_pdf",
    "load_document_inventory",
    "load_document_review",
    "validate_document_inventory",
    "validate_document_review",
]
