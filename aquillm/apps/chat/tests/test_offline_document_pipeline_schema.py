"""Contracts for frozen offline document-pipeline benchmark inputs and metrics."""

from __future__ import annotations

import copy
import hashlib
import io
from pathlib import Path

import pytest
import yaml
from pypdf import PdfReader

from apps.chat.evals.offline.document_pipeline_schema import (
    aggregate_document_results,
    build_document_record,
    build_pending_document_review,
    generate_synthetic_pdf,
    load_document_inventory,
    load_document_review,
    validate_document_inventory,
    validate_document_review,
)
from apps.chat.evals.offline.schema import canonical_json_bytes, sha256_canonical_text
from apps.chat.services import rag_evidence
from apps.documents.services.text_chunk_plan import TextChunkSpec
from aquillm.ingestion import parsers
from aquillm.task_ingest_helpers import sanitize_db_text

CASE_FIELDS = {
    "case_id",
    "sha256",
    "input_bytes",
    "selection_rationale",
    "acquisition_lineage",
    "sensitivity",
    "redistribution_license_status",
}
EXPECTED_PROTOCOL_HASHES = {
    1: (
        "0b304609233fd7b533791266f7b710a78e544c5609ce8cf20c445256f9a0366c",
        "d85906ee650bc6e4f0cbcf9886de0fabd2b1e5009a272a5d7332d818c273cd4b",
    ),
    10: (
        "04db9b687f0db956d89bd00aa1e995cd6963523aca94b95aa1b34d916ee18f23",
        "0813edeff259151ebee235f36e08a0069652e39ad00ce5d2b38a842efbb4038b",
    ),
    50: (
        "355b101287598dba241370980e150ae32f015ff751119be1244e490e15663ac5",
        "b74c7693f8bfd08d3070bd3e3bd4081f1c57c42a737fd2930b40f54d46b6daad",
    ),
    100: (
        "25c3560f0e505ceb5c5f88f9c4b8b9f5ab39ebe66b5e70770770808b56f406ec",
        "1fb1ad1e78be21e42ba3a0058e0633cfd0a01f6637eed87a49aff058ba120a0c",
    ),
}


def _valid_inventory() -> dict:
    cases = []
    for index in range(17):
        cases.append(
            {
                "case_id": f"real-{index + 1:03d}",
                "sha256": hashlib.sha256(f"pdf-{index}".encode()).hexdigest(),
                "input_bytes": 1,
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
        )
    cases.sort(key=lambda case: case["sha256"])
    cases[-1]["input_bytes"] = 97_006_698 - 16
    for index, case in enumerate(cases, 1):
        case["case_id"] = f"real-{index:03d}"
    return {
        "schema_version": "1.0",
        "inventory_id": "astro-test-real-v1",
        "source_hash_algorithm": "sha256-raw-bytes-v1",
        "case_count": 17,
        "total_bytes": sum(case["input_bytes"] for case in cases),
        "cases": cases,
    }


def _valid_protocol() -> dict:
    return {
        "generator_version": "aquillm-ascii-pdf-v1",
        "page_counts": [1, 10, 50, 100],
        "page_string_template": (
            "AquiLLM synthetic preprocessing page NNNN. The quick brown fox jumps "
            "over the lazy dog. Value NNNN."
        ),
        "cases": [
            {
                "page_count": count,
                "pdf_sha256": EXPECTED_PROTOCOL_HASHES[count][0],
                "normalized_text_sha256": EXPECTED_PROTOCOL_HASHES[count][1],
                "expected_page_count": count,
            }
            for count in (1, 10, 50, 100)
        ],
    }


def _valid_review(inventory_path: Path, *, status: str = "approved") -> dict:
    protocol = _valid_protocol()
    pending = status == "pending_independent_review"
    return {
        "schema_version": "1.0",
        "review_id": "document-corpus-and-synthetic-v1",
        "status": status,
        "source_hash_algorithm": "sha256-utf8-lf-v1",
        "inventory_hash": sha256_canonical_text(inventory_path),
        "protocol_hash_algorithm": "sha256-canonical-json-v1",
        "protocol_hash": hashlib.sha256(canonical_json_bytes(protocol)).hexdigest(),
        "protocol": protocol,
        "reviewer_identity": None
        if pending
        else "codex-agent:independent-input-reviewer",
        "reviewer_role": None if pending else "independent_reviewer",
        "review_date": None if pending else "2026-08-06",
        "decisions": [] if pending else ["approved inventory and synthetic protocol"],
    }


def _write_yaml(path: Path, value: dict) -> None:
    path.write_text(
        yaml.safe_dump(value, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )


def test_inventory_requires_exact_schema_and_frozen_case_contract():
    inventory = _valid_inventory()
    validate_document_inventory(inventory)

    assert set(inventory) == {
        "schema_version",
        "inventory_id",
        "source_hash_algorithm",
        "case_count",
        "total_bytes",
        "cases",
    }
    assert [case["case_id"] for case in inventory["cases"]] == [
        f"real-{index:03d}" for index in range(1, 18)
    ]
    assert all(set(case) == CASE_FIELDS for case in inventory["cases"])

    for mutation, message in (
        (lambda data: data.update(schema_version="2.0"), "schema_version"),
        (lambda data: data["cases"].pop(), "17"),
        (
            lambda data: data["cases"][1].update(sha256=data["cases"][0]["sha256"]),
            "duplicate",
        ),
        (lambda data: data["cases"][0].update(input_bytes=0), "positive"),
        (
            lambda data: data["cases"][0].update(selection_rationale="hand selected"),
            "selection_rationale",
        ),
    ):
        changed = copy.deepcopy(inventory)
        mutation(changed)
        with pytest.raises(ValueError, match=message):
            validate_document_inventory(changed)


@pytest.mark.parametrize(
    "private_key", ["filename", "basename", "path", "title", "content"]
)
def test_inventory_rejects_private_or_content_fields(private_key):
    inventory = _valid_inventory()
    inventory["cases"][0][private_key] = "private-value"
    with pytest.raises(ValueError, match="privacy|field|keys"):
        validate_document_inventory(inventory)


def test_inventory_rejects_absolute_or_private_path_values():
    inventory = _valid_inventory()
    inventory["cases"][0]["acquisition_lineage"] = "C:\\Users\\person\\paper.pdf"
    with pytest.raises(ValueError, match="private|path"):
        validate_document_inventory(inventory)


def test_inventory_loader_rejects_bom(tmp_path):
    path = tmp_path / "inventory.yaml"
    body = yaml.safe_dump(_valid_inventory(), sort_keys=True).encode("utf-8")
    path.write_bytes(b"\xef\xbb\xbf" + body)
    with pytest.raises(ValueError, match="BOM"):
        load_document_inventory(path)


def test_canonical_source_hash_fixed_vectors(tmp_path):
    expected = hashlib.sha256(b"alpha\nbeta\n").hexdigest()
    for index, raw in enumerate(
        (b"alpha\nbeta\n", b"alpha\r\nbeta\r\n", b"alpha\rbeta\r")
    ):
        path = tmp_path / f"vector-{index}.txt"
        path.write_bytes(raw)
        assert sha256_canonical_text(path) == expected

    no_final_lf = tmp_path / "no-final-lf.txt"
    no_final_lf.write_bytes(b"alpha\r\nbeta")
    assert (
        sha256_canonical_text(no_final_lf) == hashlib.sha256(b"alpha\nbeta").hexdigest()
    )

    bom = tmp_path / "bom.txt"
    bom.write_bytes(b"\xef\xbb\xbfalpha\r\n")
    assert (
        sha256_canonical_text(bom)
        == hashlib.sha256("\ufeffalpha\n".encode()).hexdigest()
    )


def test_review_hash_fixed_vectors():
    protocol = {"a": "caf\u00e9", "z": [2, 1]}
    assert canonical_json_bytes(protocol) == b'{"a":"caf\xc3\xa9","z":[2,1]}\n'
    assert (
        hashlib.sha256(canonical_json_bytes(protocol)).hexdigest()
        == "0086e80837682f3775eb34cf95f344e97dde178bb94dee50b9ee724f69ec54f8"
    )


def test_review_requires_exact_keys_hashes_and_independent_approval(tmp_path):
    inventory_path = tmp_path / "inventory.yaml"
    inventory = _valid_inventory()
    _write_yaml(inventory_path, inventory)
    review = _valid_review(inventory_path)

    validate_document_review(review, inventory)
    assert set(review) == {
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

    for key, value, message in (
        ("source_hash_algorithm", "sha256", "source_hash_algorithm"),
        ("protocol_hash_algorithm", "sha256", "protocol_hash_algorithm"),
        ("protocol_hash", "0" * 64, "protocol_hash"),
        ("reviewer_identity", "author", "reviewer_identity"),
        ("reviewer_role", "fixture_author", "reviewer_role"),
        ("review_date", "today", "review_date"),
        ("decisions", [], "decisions"),
    ):
        changed = copy.deepcopy(review)
        changed[key] = value
        with pytest.raises(ValueError, match=message):
            validate_document_review(changed, inventory)

    review_path = tmp_path / "review.yaml"
    review["inventory_hash"] = "0" * 64
    _write_yaml(review_path, review)
    with pytest.raises(ValueError, match="inventory_hash"):
        load_document_review(review_path, inventory_path)


@pytest.mark.parametrize(
    "case_index,field",
    [
        (None, "generator_version"),
        *((index, "page_count") for index in range(4)),
        *((index, "expected_page_count") for index in range(4)),
        *((index, "pdf_sha256") for index in range(4)),
        *((index, "normalized_text_sha256") for index in range(4)),
    ],
)
def test_review_protocol_rejects_any_frozen_generator_drift(
    tmp_path, case_index, field
):
    inventory_path = tmp_path / "inventory.yaml"
    inventory = _valid_inventory()
    _write_yaml(inventory_path, inventory)
    review = _valid_review(inventory_path)

    if field == "generator_version":
        review["protocol"][field] = "aquillm-ascii-pdf-v2"
    elif field in {"page_count", "expected_page_count"}:
        review["protocol"]["cases"][case_index][field] += 1
    else:
        review["protocol"]["cases"][case_index][field] = "f" * 64
    review["protocol_hash"] = hashlib.sha256(
        canonical_json_bytes(review["protocol"])
    ).hexdigest()

    with pytest.raises(ValueError, match="protocol|generator_version|page count"):
        validate_document_review(review, inventory)


def test_approved_review_accepts_slash_rooted_stable_task_identity(tmp_path):
    inventory_path = tmp_path / "inventory.yaml"
    inventory = _valid_inventory()
    _write_yaml(inventory_path, inventory)
    review = _valid_review(inventory_path)
    review["reviewer_identity"] = "codex-agent:/root/document_task2_input_review"

    validate_document_review(review, inventory)


@pytest.mark.parametrize(
    "identity",
    [
        "codex-agent:self",
        "codex-agent:ambient",
        "codex-agent:/",
        "codex-agent://root/reviewer",
        "codex-agent:/root/../reviewer",
    ],
)
def test_approved_review_rejects_placeholder_or_invalid_task_identity(
    tmp_path, identity
):
    inventory_path = tmp_path / "inventory.yaml"
    inventory = _valid_inventory()
    _write_yaml(inventory_path, inventory)
    review = _valid_review(inventory_path)
    review["reviewer_identity"] = identity

    with pytest.raises(ValueError, match="reviewer_identity"):
        validate_document_review(review, inventory)


def test_pending_review_requires_opt_in_and_is_structurally_strict(tmp_path):
    inventory_path = tmp_path / "inventory.yaml"
    review_path = tmp_path / "review.yaml"
    inventory = _valid_inventory()
    _write_yaml(inventory_path, inventory)
    pending = _valid_review(inventory_path, status="pending_independent_review")
    _write_yaml(review_path, pending)

    with pytest.raises(ValueError, match="approved independent review"):
        load_document_review(review_path, inventory_path)
    assert (
        load_document_review(review_path, inventory_path, allow_pending=True) == pending
    )

    for key, value in (
        ("reviewer_identity", "codex-agent:self"),
        ("reviewer_role", "independent_reviewer"),
        ("review_date", "2026-08-06"),
        ("decisions", ["self approved"]),
    ):
        changed = copy.deepcopy(pending)
        changed[key] = value
        with pytest.raises(ValueError, match=key):
            validate_document_review(changed, inventory, allow_pending=True)


def test_review_loader_rejects_bom_and_private_fields(tmp_path):
    inventory_path = tmp_path / "inventory.yaml"
    review_path = tmp_path / "review.yaml"
    inventory = _valid_inventory()
    _write_yaml(inventory_path, inventory)
    review = _valid_review(inventory_path)
    review["protocol"]["path"] = "/Users/person/private.pdf"
    review["protocol_hash"] = hashlib.sha256(
        canonical_json_bytes(review["protocol"])
    ).hexdigest()
    _write_yaml(review_path, review)
    with pytest.raises(ValueError, match="privacy|path"):
        load_document_review(review_path, inventory_path)

    review_path.write_bytes(b"\xef\xbb\xbf" + review_path.read_bytes())
    with pytest.raises(ValueError, match="BOM"):
        load_document_review(review_path, inventory_path)


@pytest.mark.parametrize("page_count", [1, 10, 50, 100])
def test_synthetic_pdf_is_byte_deterministic(page_count):
    first_bytes, first_text = generate_synthetic_pdf(page_count)
    second_bytes, second_text = generate_synthetic_pdf(page_count)

    assert first_bytes == second_bytes
    assert first_text == second_text
    assert first_bytes.startswith(b"%PDF-1.4\n%AquiLLM\n")
    assert first_bytes.endswith(b"%%EOF\n")
    assert not first_bytes.endswith(b"\n\n")
    assert b"\r" not in first_bytes


@pytest.mark.parametrize(
    "page_count,expected_pdf_sha256,expected_text_sha256",
    [
        (
            1,
            "0b304609233fd7b533791266f7b710a78e544c5609ce8cf20c445256f9a0366c",
            "d85906ee650bc6e4f0cbcf9886de0fabd2b1e5009a272a5d7332d818c273cd4b",
        ),
        (
            10,
            "04db9b687f0db956d89bd00aa1e995cd6963523aca94b95aa1b34d916ee18f23",
            "0813edeff259151ebee235f36e08a0069652e39ad00ce5d2b38a842efbb4038b",
        ),
        (
            50,
            "355b101287598dba241370980e150ae32f015ff751119be1244e490e15663ac5",
            "b74c7693f8bfd08d3070bd3e3bd4081f1c57c42a737fd2930b40f54d46b6daad",
        ),
        (
            100,
            "25c3560f0e505ceb5c5f88f9c4b8b9f5ab39ebe66b5e70770770808b56f406ec",
            "1fb1ad1e78be21e42ba3a0058e0633cfd0a01f6637eed87a49aff058ba120a0c",
        ),
    ],
)
def test_synthetic_pdf_page_count_and_expected_text(
    monkeypatch, page_count, expected_pdf_sha256, expected_text_sha256
):
    pdf_bytes, normalized_text = generate_synthetic_pdf(page_count)
    page_strings = [
        (
            f"AquiLLM synthetic preprocessing page {index:04d}. The quick brown "
            f"fox jumps over the lazy dog. Value {index:04d}."
        )
        for index in range(1, page_count + 1)
    ]
    independently_expected = "\n\n".join(page_strings).strip()

    reader = PdfReader(io.BytesIO(pdf_bytes))
    assert len(reader.pages) == page_count

    figure_calls = []

    def fail_figure_hook(*args, **kwargs):
        figure_calls.append((args, kwargs))
        raise AssertionError("primary-text extraction must exclude figures")

    monkeypatch.setattr(
        parsers, "extract_figure_payloads_for_format", fail_figure_hook
    )
    payload = parsers.extract_primary_text_payload(
        "synthetic.pdf",
        pdf_bytes,
        content_type="application/pdf",
        ingest_type="document",
    )
    production_text = sanitize_db_text(payload.full_text).strip()

    assert figure_calls == []
    assert payload.normalized_type == "pdf"
    assert hashlib.sha256(pdf_bytes).hexdigest() == expected_pdf_sha256
    assert production_text == independently_expected
    assert hashlib.sha256(production_text.encode("utf-8")).hexdigest() == (
        expected_text_sha256
    )
    assert normalized_text == independently_expected

    size = 4 + (2 * page_count)
    kids = " ".join(f"{4 + (2 * index)} 0 R" for index in range(page_count))
    assert (
        f"2 0 obj\n<< /Type /Pages /Count {page_count} /Kids [{kids}] >>\nendobj\n"
    ).encode("ascii") in pdf_bytes
    assert f"xref\n0 {size}\n".encode("ascii") in pdf_bytes


@pytest.mark.parametrize("page_count", [0, -1])
def test_synthetic_pdf_rejects_nonpositive_page_count(page_count):
    with pytest.raises(ValueError, match="positive"):
        generate_synthetic_pdf(page_count)


def test_truncated_synthetic_pdf_fails_validation():
    pdf_bytes, _ = generate_synthetic_pdf(10)
    truncated = pdf_bytes[: len(pdf_bytes) // 2]
    with pytest.raises(Exception):
        PdfReader(io.BytesIO(truncated))


STATIC_RECORD_FIELDS = {
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
}


def test_build_document_record_absolute_units():
    text = "na\u00efve caf\u00e9\nthree words"
    chunks = [
        TextChunkSpec(text[:8], 0, 8, 0),
        TextChunkSpec(text[6:], 6, len(text), 1),
    ]

    record = build_document_record(
        arm="real",
        case_id="real-001",
        input_bytes=1_572_864,
        page_count=2,
        success=True,
        diagnostic_code="ok",
        sanitized_text=text,
        chunk_specs=chunks,
    )

    assert set(record) == STATIC_RECORD_FIELDS
    assert record == {
        "schema_version": "1.0",
        "arm": "real",
        "case_id": "real-001",
        "success": True,
        "diagnostic_code": "ok",
        "input_bytes": 1_572_864,
        "input_mib": 1.5,
        "page_count": 2,
        "extracted_codepoints": len(text),
        "extracted_utf8_bytes": len(text.encode("utf-8")),
        "word_count": 4,
        "estimated_tokens": rag_evidence._estimate_tokens(text),
        "chunk_count": 2,
        "coverage_codepoints": len(text),
        "total_chunk_codepoints": 8 + (len(text) - 6),
        "excess_overlap_codepoints": 2,
        "overlap_ratio": 2 / len(text),
        "chunk_min_codepoints": min(8, len(text) - 6),
        "chunk_mean_codepoints": (8 + len(text) - 6) / 2,
        "chunk_median_codepoints": (8 + len(text) - 6) / 2,
        "chunk_max_codepoints": max(8, len(text) - 6),
        "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def test_build_document_record_uses_production_token_estimator(monkeypatch):
    calls = []

    def estimate(text):
        calls.append(text)
        return 777

    monkeypatch.setattr(rag_evidence, "_estimate_tokens", estimate)
    record = build_document_record(
        arm="synthetic",
        case_id="synthetic-001",
        input_bytes=10,
        page_count=1,
        success=True,
        diagnostic_code="ok",
        sanitized_text="token source",
        chunk_specs=[TextChunkSpec("token source", 0, 12, 0)],
    )
    assert calls == ["token source"]
    assert record["estimated_tokens"] == 777


def _valid_document_record_kwargs() -> dict:
    return {
        "arm": "real",
        "case_id": "real-001",
        "input_bytes": 10,
        "page_count": 1,
        "success": True,
        "diagnostic_code": "ok",
        "sanitized_text": "text",
        "chunk_specs": [TextChunkSpec("text", 0, 4, 0)],
    }


def test_build_document_record_requires_exact_bool_success():
    kwargs = _valid_document_record_kwargs()
    kwargs["success"] = 1

    with pytest.raises(ValueError, match="success"):
        build_document_record(**kwargs)


def test_build_document_record_success_requires_known_page_count():
    kwargs = _valid_document_record_kwargs()
    kwargs["page_count"] = None

    with pytest.raises(ValueError, match="page_count"):
        build_document_record(**kwargs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("arm", []),
        ("case_id", True),
        ("input_bytes", True),
        ("input_bytes", 10.0),
        ("page_count", True),
        ("page_count", 1.0),
        ("diagnostic_code", []),
    ],
)
def test_build_document_record_rejects_wrong_scalar_types(field, value):
    kwargs = _valid_document_record_kwargs()
    kwargs[field] = value

    with pytest.raises(ValueError, match=field):
        build_document_record(**kwargs)


@pytest.mark.parametrize(
    "diagnostic_code,page_count",
    [
        ("invalid_pdf", None),
        ("encrypted_pdf", 3),
        ("empty_primary_text", 2),
        ("parser_error", 1),
    ],
)
def test_build_document_record_failure_conventions(diagnostic_code, page_count):
    record = build_document_record(
        arm="real",
        case_id="real-001",
        input_bytes=1_048_576,
        page_count=page_count,
        success=False,
        diagnostic_code=diagnostic_code,
        sanitized_text=None,
        chunk_specs=None,
    )
    assert record["input_bytes"] == 1_048_576
    assert record["input_mib"] == 1.0
    assert record["page_count"] == page_count
    for key in STATIC_RECORD_FIELDS - {
        "schema_version",
        "arm",
        "case_id",
        "success",
        "diagnostic_code",
        "input_bytes",
        "input_mib",
        "page_count",
    }:
        assert record[key] is None

    with pytest.raises(ValueError, match="diagnostic"):
        build_document_record(
            arm="real",
            case_id="real-001",
            input_bytes=1,
            page_count=None,
            success=False,
            diagnostic_code="raw exception text",
            sanitized_text=None,
            chunk_specs=None,
        )


TIMING_METRICS = (
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
)
TIMING_SWEEP_FIELDS = {
    "schema_version",
    "arm",
    "sweep_index",
    "ordered_case_ids",
    "attempted_count",
    "success_count",
    "failure_count",
    "successful_input_bytes",
    "successful_page_count",
    "successful_extracted_codepoints",
    "successful_estimated_tokens",
    *TIMING_METRICS,
}
MEMORY_FIELDS = {
    "schema_version",
    "arm",
    "case_id",
    "memory_index",
    "success",
    "diagnostic_code",
    "peak_python_traced_bytes",
}


def _timing_sweep_record(
    arm, sweep_index, ordered_case_ids, *, success_count=None, multiplier=1
):
    attempted_count = len(ordered_case_ids)
    if success_count is None:
        success_count = attempted_count
    row = {
        "schema_version": "1.0",
        "arm": arm,
        "sweep_index": sweep_index,
        "ordered_case_ids": ordered_case_ids,
        "attempted_count": attempted_count,
        "success_count": success_count,
        "failure_count": attempted_count - success_count,
        "successful_input_bytes": success_count,
        "successful_page_count": success_count,
        "successful_extracted_codepoints": success_count,
        "successful_estimated_tokens": success_count,
    }
    row.update(
        {
            metric: (
                2 * multiplier * (index + 1)
                if metric.endswith("_ns")
                else 2.0 * multiplier * (index + 1)
            )
            for index, metric in enumerate(TIMING_METRICS)
        }
    )
    return row


def _memory_record(
    arm, case_id, memory_index, success, diagnostic_code, peak_bytes
):
    return {
        "schema_version": "1.0",
        "arm": arm,
        "case_id": case_id,
        "memory_index": memory_index,
        "success": success,
        "diagnostic_code": diagnostic_code,
        "peak_python_traced_bytes": peak_bytes,
    }


def _success_record(arm, case_id, input_bytes, pages, text):
    return build_document_record(
        arm=arm,
        case_id=case_id,
        input_bytes=input_bytes,
        page_count=pages,
        success=True,
        diagnostic_code="ok",
        sanitized_text=text,
        chunk_specs=[TextChunkSpec(text, 0, len(text), 0)],
    )


def test_aggregate_document_results_ratio_denominators():
    static_records = [
        _success_record("real", "real-001", 1_048_576, 2, "abcdefgh"),
        build_document_record(
            arm="real",
            case_id="real-002",
            input_bytes=99,
            page_count=None,
            success=False,
            diagnostic_code="invalid_pdf",
            sanitized_text=None,
            chunk_specs=None,
        ),
        _success_record("synthetic", "synthetic-001", 100, 1, "abcd"),
    ]
    timing_sweeps = []
    for arm, multiplier in (("real", 1), ("synthetic", 10)):
        for sweep_index, scale in enumerate((1, 2)):
            timing_sweeps.append(
                _timing_sweep_record(
                    arm,
                    sweep_index,
                    ["real-001", "real-002"]
                    if arm == "real"
                    else ["synthetic-001"],
                    success_count=1,
                    multiplier=multiplier * scale,
                )
            )
    memory_records = [
        _memory_record("real", "real-001", 0, True, "ok", 100),
        _memory_record("real", "real-001", 1, True, "ok", 125),
        _memory_record("real", "real-001", 2, False, "parser_error", 400),
        _memory_record("real", "real-002", 0, False, "invalid_pdf", 300),
        _memory_record("synthetic", "synthetic-001", 0, True, "ok", 55),
    ]
    network_audit = {
        "guard": "process_local_socket_guard",
        "scope": "benchmark_process",
        "total_attempts": 0,
        "details": [],
    }

    aggregate = aggregate_document_results(
        static_records,
        timing_sweeps,
        memory_records,
        network_audit=network_audit,
    )

    assert set(aggregate) == {
        "schema_version",
        "real",
        "synthetic",
        "timing",
        "memory",
        "failures",
        "network_audit",
        "excluded_claims",
    }
    assert aggregate["real"] == {
        "attempted_count": 2,
        "success_count": 1,
        "failure_count": 1,
        "success_rate": 0.5,
        "total_input_bytes": 1_048_576,
        "total_page_count": 2,
        "total_extracted_codepoints": 8,
        "total_estimated_tokens": 2,
        "total_chunk_count": 1,
        "total_coverage_codepoints": 8,
        "total_excess_overlap_codepoints": 0,
    }
    assert aggregate["synthetic"]["success_rate"] == 1.0
    real_first_timing = aggregate["timing"]["real"][TIMING_METRICS[0]]
    assert real_first_timing == {
        "support_sweeps": 2,
        "median": 3.0,
        "nearest_rank_p95": 4.0,
        "unit": "nanoseconds",
    }
    assert aggregate["memory"]["real"] == {
        "successful_case_count": 1,
        "max_peak_python_traced_bytes_per_case": {
            "real-001": 125,
        },
        "max_peak_python_traced_bytes_over_successes": 125,
        "max_peak_python_traced_bytes_over_all_attempts": 400,
    }
    assert aggregate["failures"]["real"]["attempted_count"] == 2
    assert aggregate["failures"]["real"]["failed_count"] == 1
    assert aggregate["failures"]["real"]["diagnostic_counts"] == {
        "ok": 1,
        "invalid_pdf": 1,
        "encrypted_pdf": 0,
        "empty_primary_text": 0,
        "parser_error": 0,
    }
    assert aggregate["network_audit"] == network_audit
    assert aggregate["excluded_claims"]


def _minimal_aggregate_inputs():
    static_records = [
        _success_record("real", "real-001", 10, 1, "real"),
        _success_record("synthetic", "synthetic-001", 20, 1, "synthetic"),
    ]
    timing_sweeps = [
        _timing_sweep_record("real", 0, ["real-001"]),
        _timing_sweep_record("synthetic", 0, ["synthetic-001"]),
    ]
    memory_records = [
        _memory_record("real", "real-001", 0, True, "ok", 10),
        _memory_record("synthetic", "synthetic-001", 0, True, "ok", 20),
    ]
    return static_records, timing_sweeps, memory_records


def _aggregate_minimal(static_records, timing_sweeps, memory_records):
    return aggregate_document_results(
        static_records,
        timing_sweeps,
        memory_records,
        network_audit={
            "guard": "process_local_socket_guard",
            "scope": "benchmark_process",
            "total_attempts": 0,
            "details": [],
        },
    )


@pytest.mark.parametrize("collection_index", [0, 1, 2])
@pytest.mark.parametrize("invalid_arm", ["unknown", True])
def test_aggregate_rejects_unknown_or_mistyped_arm(collection_index, invalid_arm):
    inputs = list(_minimal_aggregate_inputs())
    inputs[collection_index][0]["arm"] = invalid_arm

    with pytest.raises(ValueError, match="arm"):
        _aggregate_minimal(*inputs)


@pytest.mark.parametrize(
    "collection_index,field,value",
    [
        (0, "input_bytes", True),
        (0, "input_bytes", "10"),
        (1, "case_combined_sum_ns", True),
        (1, "case_combined_sum_ns", "10"),
        (2, "peak_python_traced_bytes", True),
        (2, "peak_python_traced_bytes", "10"),
    ],
)
def test_aggregate_rejects_bool_or_string_numeric_fields(
    collection_index, field, value
):
    inputs = list(_minimal_aggregate_inputs())
    inputs[collection_index][0][field] = value

    with pytest.raises(ValueError, match=field):
        _aggregate_minimal(*inputs)


@pytest.mark.parametrize("collection_index", [0, 1, 2])
@pytest.mark.parametrize("change", ["missing", "extra"])
def test_aggregate_rejects_missing_or_extra_row_keys(collection_index, change):
    inputs = list(_minimal_aggregate_inputs())
    row = inputs[collection_index][0]
    if change == "missing":
        del row["schema_version"]
    else:
        row["unexpected"] = "value"

    with pytest.raises(ValueError, match="keys"):
        _aggregate_minimal(*inputs)


@pytest.mark.parametrize("collection_index", [0, 2])
@pytest.mark.parametrize(
    "field,value",
    [("success", 1), ("diagnostic_code", "unsafe_raw_exception")],
)
def test_aggregate_rejects_invalid_success_or_diagnostic(
    collection_index, field, value
):
    inputs = list(_minimal_aggregate_inputs())
    inputs[collection_index][0][field] = value

    with pytest.raises(ValueError, match=field):
        _aggregate_minimal(*inputs)


@pytest.mark.parametrize("collection_index", [1, 2])
def test_aggregate_rejects_unknown_or_cross_arm_case_identity(collection_index):
    inputs = list(_minimal_aggregate_inputs())
    row = inputs[collection_index][0]
    if collection_index == 1:
        row["ordered_case_ids"] = ["synthetic-001"]
    else:
        row["case_id"] = "synthetic-001"

    with pytest.raises(ValueError, match="case_id|identity|static"):
        _aggregate_minimal(*inputs)


INVENTORY_PATH = (
    Path(__file__).parents[1] / "evals" / "offline" / "document_corpus_inventory.yaml"
)
REVIEW_PATH = INVENTORY_PATH.with_name("document_corpus_review.yaml")


def test_build_pending_review_exact_shape_and_hashes():
    review = build_pending_document_review(INVENTORY_PATH)

    assert set(review) == {
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
    assert review["schema_version"] == "1.0"
    assert review["review_id"] == "document-corpus-and-synthetic-v1"
    assert review["status"] == "pending_independent_review"
    assert review["source_hash_algorithm"] == "sha256-utf8-lf-v1"
    assert review["inventory_hash"] == (
        "17a89c2e4fb74dda4be49e83b1a4afb27dd05ed0bf03ea1849a7f14961c3c131"
    )
    assert review["protocol_hash_algorithm"] == "sha256-canonical-json-v1"
    assert review["protocol_hash"] == (
        "1125d1212f33f92c2d0529a998fc42fa0579f04153653c522fd1fdd0b658a6cd"
    )
    assert review["protocol"] == {
        "generator_version": "aquillm-ascii-pdf-v1",
        "page_counts": [1, 10, 50, 100],
        "page_string_template": (
            "AquiLLM synthetic preprocessing page NNNN. The quick brown fox jumps "
            "over the lazy dog. Value NNNN."
        ),
        "cases": [
            {
                "page_count": 1,
                "pdf_sha256": (
                    "0b304609233fd7b533791266f7b710a78e544c5609ce8cf20c445256f9a0366c"
                ),
                "normalized_text_sha256": (
                    "d85906ee650bc6e4f0cbcf9886de0fabd2b1e5009a272a5d7332d818c273cd4b"
                ),
                "expected_page_count": 1,
            },
            {
                "page_count": 10,
                "pdf_sha256": (
                    "04db9b687f0db956d89bd00aa1e995cd6963523aca94b95aa1b34d916ee18f23"
                ),
                "normalized_text_sha256": (
                    "0813edeff259151ebee235f36e08a0069652e39ad00ce5d2b38a842efbb4038b"
                ),
                "expected_page_count": 10,
            },
            {
                "page_count": 50,
                "pdf_sha256": (
                    "355b101287598dba241370980e150ae32f015ff751119be1244e490e15663ac5"
                ),
                "normalized_text_sha256": (
                    "b74c7693f8bfd08d3070bd3e3bd4081f1c57c42a737fd2930b40f54d46b6daad"
                ),
                "expected_page_count": 50,
            },
            {
                "page_count": 100,
                "pdf_sha256": (
                    "25c3560f0e505ceb5c5f88f9c4b8b9f5ab39ebe66b5e70770770808b56f406ec"
                ),
                "normalized_text_sha256": (
                    "1fb1ad1e78be21e42ba3a0058e0633cfd0a01f6637eed87a49aff058ba120a0c"
                ),
                "expected_page_count": 100,
            },
        ],
    }
    assert review["reviewer_identity"] is None
    assert review["reviewer_role"] is None
    assert review["review_date"] is None
    assert review["decisions"] == []
    validate_document_review(
        review, load_document_inventory(INVENTORY_PATH), allow_pending=True
    )


def test_pending_review_requires_opt_in(tmp_path):
    pending_path = tmp_path / "document_corpus_review.yaml"
    pending = build_pending_document_review(INVENTORY_PATH)
    _write_yaml(pending_path, pending)

    with pytest.raises(ValueError, match="approved independent review"):
        load_document_review(pending_path, INVENTORY_PATH)

    loaded = load_document_review(pending_path, INVENTORY_PATH, allow_pending=True)
    assert loaded == pending


def test_committed_inventory_and_review_are_path_free():
    inventory = load_document_inventory(INVENTORY_PATH)
    validate_document_inventory(inventory)
    load_document_review(REVIEW_PATH, INVENTORY_PATH, allow_pending=True)

    combined = INVENTORY_PATH.read_text("utf-8") + REVIEW_PATH.read_text("utf-8")
    lowered = combined.casefold()
    for prohibited in (
        "filename:",
        "basename:",
        "path:",
        "title:",
        "content:",
        "c:\\users\\",
        "/users/",
        "/home/",
    ):
        assert prohibited not in lowered
