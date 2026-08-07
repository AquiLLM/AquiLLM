from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pypdf.errors import FileNotDecryptedError, PdfReadError

from apps.chat.evals.offline.document_pipeline_schema import (
    build_document_record,
    generate_synthetic_pdf,
)
from apps.documents.services.text_chunk_plan import plan_text_chunks


def _inventory_for(*members: tuple[str, bytes]) -> dict:
    return {
        "cases": [
            {
                "case_id": case_id,
                "sha256": hashlib.sha256(pdf_bytes).hexdigest(),
                "input_bytes": len(pdf_bytes),
            }
            for case_id, pdf_bytes in members
        ]
    }


def test_resolve_real_corpus_matches_direct_case_insensitive_pdfs(tmp_path: Path):
    from apps.chat.evals.offline.document_pipeline_runner import resolve_real_corpus

    first, _ = generate_synthetic_pdf(1)
    second, _ = generate_synthetic_pdf(10)
    (tmp_path / "opaque-a.PDF").write_bytes(first)
    (tmp_path / "opaque-b.pdf").write_bytes(second)
    (tmp_path / "notes.json").write_text("sidecar", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "ignored.pdf").write_bytes(first)

    cases = resolve_real_corpus(
        tmp_path,
        _inventory_for(("real-002", second), ("real-001", first)),
    )

    assert [case["case_id"] for case in cases] == ["real-001", "real-002"]
    assert cases[0] == {
        "arm": "real",
        "case_id": "real-001",
        "pdf_bytes": first,
        "raw_sha256": hashlib.sha256(first).hexdigest(),
        "page_count": 1,
    }
    assert not (
        {"path", "name", "filename", "title", "content", "text"} & cases[0].keys()
    )


@pytest.mark.parametrize(
    ("mutation", "error_code"),
    [
        ("unlisted", "unlisted_pdf"),
        ("missing", "missing_inventory_pdf"),
        ("duplicate", "duplicate_pdf_hash"),
        ("size", "pdf_size_mismatch"),
    ],
)
def test_resolve_real_corpus_fails_closed_with_path_free_errors(
    tmp_path: Path, mutation: str, error_code: str
):
    from apps.chat.evals.offline.document_pipeline_runner import resolve_real_corpus

    first, _ = generate_synthetic_pdf(1)
    second, _ = generate_synthetic_pdf(10)
    inventory = _inventory_for(("real-001", first))
    (tmp_path / "private-paper.pdf").write_bytes(first)
    if mutation == "unlisted":
        (tmp_path / "secret-extra.pdf").write_bytes(second)
    elif mutation == "missing":
        inventory = _inventory_for(("real-001", first), ("real-002", second))
    elif mutation == "duplicate":
        (tmp_path / "private-copy.PDF").write_bytes(first)
    else:
        inventory["cases"][0]["input_bytes"] += 1

    with pytest.raises(ValueError, match=error_code) as raised:
        resolve_real_corpus(tmp_path, inventory)

    message = str(raised.value)
    assert "private-paper" not in message
    assert str(tmp_path) not in message


def test_resolve_real_corpus_can_explicitly_ignore_unlisted_pdfs(tmp_path: Path):
    from apps.chat.evals.offline.document_pipeline_runner import resolve_real_corpus

    first, _ = generate_synthetic_pdf(1)
    second, _ = generate_synthetic_pdf(10)
    (tmp_path / "listed.pdf").write_bytes(first)
    (tmp_path / "extra.pdf").write_bytes(second)

    cases = resolve_real_corpus(
        tmp_path,
        _inventory_for(("real-001", first)),
        allow_unlisted_pdfs=True,
    )

    assert [case["case_id"] for case in cases] == ["real-001"]


def test_run_document_case_uses_exact_production_pipeline(monkeypatch):
    from apps.chat.evals.offline import document_pipeline_runner as runner
    from apps.documents.services.text_chunk_plan import plan_text_chunks as real_plan
    from aquillm.ingestion import parsers
    from aquillm.task_ingest_helpers import sanitize_db_text as real_sanitize

    pdf_bytes, _ = generate_synthetic_pdf(1)
    case = {
        "arm": "real",
        "case_id": "real-001",
        "pdf_bytes": pdf_bytes,
        "raw_sha256": hashlib.sha256(pdf_bytes).hexdigest(),
        "page_count": 1,
    }
    calls: list[object] = []
    real_detect = parsers.detect_ingest_type
    real_extract = parsers.extract_primary_text_payload

    def detect(filename, content_type=None):
        calls.append(("detect", filename, content_type))
        return real_detect(filename, content_type)

    def extract(filename, data, *, content_type=None, ingest_type=None):
        calls.append(("extract", filename, content_type, ingest_type))
        return real_extract(
            filename, data, content_type=content_type, ingest_type=ingest_type
        )

    def sanitize(value):
        calls.append(("sanitize", value))
        return real_sanitize(value)

    def plan(text, *, chunk_size, overlap):
        calls.append(("plan", text, chunk_size, overlap))
        return real_plan(text, chunk_size=chunk_size, overlap=overlap)

    monkeypatch.setattr(runner.parsers, "detect_ingest_type", detect)
    monkeypatch.setattr(runner.parsers, "extract_primary_text_payload", extract)
    monkeypatch.setattr(runner, "sanitize_db_text", sanitize)
    monkeypatch.setattr(runner, "plan_text_chunks", plan)

    result = runner.run_document_case(case, chunk_size=2048, overlap=384)

    assert [call[0] for call in calls] == ["detect", "extract", "sanitize", "plan"]
    assert calls[1][3] == "document"
    assert calls[2][1]
    assert calls[3][1] == real_sanitize(calls[2][1]).strip()
    assert result["static_record"]["success"] is True
    assert result["static_record"]["page_count"] == 1
    assert result["timing"]["terminal_stage"] == "complete"


class _TickClock:
    def __init__(self, step: int = 10):
        self.value = -step
        self.step = step

    def __call__(self) -> int:
        self.value += self.step
        return self.value


def _one_page_case(arm: str = "real") -> dict:
    pdf_bytes, _ = generate_synthetic_pdf(1)
    return {
        "arm": arm,
        "case_id": f"{arm}-001",
        "pdf_bytes": pdf_bytes,
        "raw_sha256": hashlib.sha256(pdf_bytes).hexdigest(),
        "page_count": 1,
    }


def test_run_document_case_uses_nested_timers_and_direct_combined_interval():
    from apps.chat.evals.offline.document_pipeline_runner import run_document_case

    result = run_document_case(
        _one_page_case(), chunk_size=2048, overlap=384, clock=_TickClock()
    )

    assert result["timing"] == {
        "success": True,
        "diagnostic_code": "ok",
        "terminal_stage": "complete",
        "detect_ns": 10,
        "extract_ns": 10,
        "sanitize_ns": 10,
        "chunk_plan_ns": 10,
        "combined_ns": 90,
    }
    assert result["timing"]["combined_ns"] != sum(
        result["timing"][field]
        for field in ("detect_ns", "extract_ns", "sanitize_ns", "chunk_plan_ns")
    )


def test_combined_timer_closes_before_failure_diagnostic_mapping(monkeypatch):
    from apps.chat.evals.offline import document_pipeline_runner as runner

    events = []
    ticks = iter((0, 1, 2, 3))

    def clock():
        events.append("clock")
        return next(ticks)

    monkeypatch.setattr(
        runner.parsers,
        "detect_ingest_type",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fail")),
    )
    monkeypatch.setattr(
        runner,
        "_diagnostic_for",
        lambda *args, **kwargs: (events.append("diagnostic") or "parser_error"),
    )

    runner.run_document_case(
        _one_page_case(), chunk_size=2048, overlap=384, clock=clock
    )

    assert events == ["clock", "clock", "clock", "clock", "diagnostic"]


@pytest.mark.parametrize(
    ("stage", "exception", "diagnostic", "expected_timings"),
    [
        ("detect", RuntimeError("detect"), "parser_error", [10, None, None, None]),
        (
            "extract",
            PdfReadError("corrupt and private detail"),
            "invalid_pdf",
            [10, 10, None, None],
        ),
        (
            "extract",
            FileNotDecryptedError("secret"),
            "encrypted_pdf",
            [10, 10, None, None],
        ),
        ("sanitize", RuntimeError("sanitize"), "parser_error", [10, 10, 10, None]),
        ("chunk_plan", RuntimeError("chunk"), "parser_error", [10, 10, 10, 10]),
    ],
)
def test_run_document_case_retains_safe_real_failures(
    monkeypatch, stage, exception, diagnostic, expected_timings
):
    from apps.chat.evals.offline import document_pipeline_runner as runner

    case = _one_page_case()
    target = {
        "detect": (runner.parsers, "detect_ingest_type"),
        "extract": (runner.parsers, "extract_primary_text_payload"),
        "sanitize": (runner, "sanitize_db_text"),
        "chunk_plan": (runner, "plan_text_chunks"),
    }[stage]

    def fail(*args, **kwargs):
        raise exception

    monkeypatch.setattr(*target, fail)
    result = runner.run_document_case(
        case, chunk_size=2048, overlap=384, clock=_TickClock()
    )

    assert result["static_record"]["success"] is False
    assert result["static_record"]["diagnostic_code"] == diagnostic
    assert result["static_record"]["input_bytes"] == len(case["pdf_bytes"])
    assert result["static_record"]["page_count"] == 1
    assert result["static_record"]["extracted_codepoints"] is None
    assert result["timing"]["terminal_stage"] == stage
    assert [
        result["timing"][field]
        for field in ("detect_ns", "extract_ns", "sanitize_ns", "chunk_plan_ns")
    ] == expected_timings
    assert "private" not in repr(result)
    assert "secret" not in repr(result)


def test_run_document_case_classifies_empty_primary_text(monkeypatch):
    from apps.chat.evals.offline import document_pipeline_runner as runner

    class EmptyPayload:
        full_text = " \x00\n "

    monkeypatch.setattr(
        runner.parsers,
        "extract_primary_text_payload",
        lambda *args, **kwargs: EmptyPayload(),
    )

    result = runner.run_document_case(
        _one_page_case(), chunk_size=2048, overlap=384, clock=_TickClock()
    )

    assert result["static_record"]["diagnostic_code"] == "empty_primary_text"
    assert result["timing"]["terminal_stage"] == "sanitize"
    assert result["timing"]["chunk_plan_ns"] is None


def test_run_document_case_synthetic_failure_fails_closed(monkeypatch):
    from apps.chat.evals.offline import document_pipeline_runner as runner

    monkeypatch.setattr(
        runner.parsers,
        "extract_primary_text_payload",
        lambda *args, **kwargs: (_ for _ in ()).throw(PdfReadError("bad")),
    )

    with pytest.raises(
        runner.BenchmarkIntegrityError, match="synthetic benchmark integrity"
    ):
        runner.run_document_case(
            _one_page_case("synthetic"),
            chunk_size=2048,
            overlap=384,
            clock=_TickClock(),
        )


def _static_record(arm: str, case_id: str, *, success: bool) -> dict:
    if not success:
        return build_document_record(
            arm=arm,
            case_id=case_id,
            input_bytes=300,
            page_count=3,
            success=False,
            diagnostic_code="parser_error",
            sanitized_text=None,
            chunk_specs=None,
        )
    text = "abcdefgh"
    return build_document_record(
        arm=arm,
        case_id=case_id,
        input_bytes=100,
        page_count=2,
        success=True,
        diagnostic_code="ok",
        sanitized_text=text,
        chunk_specs=plan_text_chunks(text, chunk_size=5, overlap=1),
    )


def test_timing_sweeps_warm_each_arm_and_rotate_case_order(monkeypatch):
    from apps.chat.evals.offline import document_pipeline_runner as runner

    cases_by_arm = {
        "real": [
            {"arm": "real", "case_id": "real-b"},
            {"arm": "real", "case_id": "real-a"},
        ],
        "synthetic": [
            {"arm": "synthetic", "case_id": "synthetic-b"},
            {"arm": "synthetic", "case_id": "synthetic-a"},
        ],
    }
    static_records = [
        _static_record("real", "real-a", success=True),
        _static_record("real", "real-b", success=False),
        _static_record("synthetic", "synthetic-a", success=True),
        _static_record("synthetic", "synthetic-b", success=True),
    ]
    events = []

    def observe(case, *, chunk_size, overlap, clock):
        events.append((case["arm"], case["case_id"]))
        static = next(
            row
            for row in static_records
            if (row["arm"], row["case_id"]) == (case["arm"], case["case_id"])
        )
        combined = 100 if case["case_id"].endswith("a") else 300
        return {
            "success": static["success"],
            "diagnostic_code": static["diagnostic_code"],
            "terminal_stage": "complete" if static["success"] else "extract",
            "detect_ns": 10,
            "extract_ns": 20,
            "sanitize_ns": 30 if static["success"] else None,
            "chunk_plan_ns": 40 if static["success"] else None,
            "combined_ns": combined,
        }

    monkeypatch.setattr(runner, "_observe_pipeline", observe)

    timing_cases, timing_sweeps = runner._run_timing_sweeps(
        cases_by_arm,
        static_records,
        sweeps=3,
        chunk_size=2048,
        overlap=384,
        clock=_TickClock(),
    )

    assert events[:2] == [("real", "real-a"), ("real", "real-b")]
    assert events[8:10] == [
        ("synthetic", "synthetic-a"),
        ("synthetic", "synthetic-b"),
    ]
    assert len(events) == 16  # one two-case warm-up and three sweeps per arm
    assert len(timing_cases) == 12
    assert len(timing_sweeps) == 6
    assert [
        row["ordered_case_ids"] for row in timing_sweeps if row["arm"] == "real"
    ] == [
        ["real-a", "real-b"],
        ["real-b", "real-a"],
        ["real-a", "real-b"],
    ]


def test_timing_case_rows_copy_static_work_units_exactly(monkeypatch):
    from apps.chat.evals.offline import document_pipeline_runner as runner

    static = _static_record("real", "real-a", success=True)
    monkeypatch.setattr(
        runner,
        "_observe_pipeline",
        lambda *args, **kwargs: {
            "success": True,
            "diagnostic_code": "ok",
            "terminal_stage": "complete",
            "detect_ns": 1,
            "extract_ns": 2,
            "sanitize_ns": 3,
            "chunk_plan_ns": 4,
            "combined_ns": 20,
        },
    )

    case_rows, _ = runner._run_timing_sweeps(
        {"real": [{"arm": "real", "case_id": "real-a"}], "synthetic": []},
        [static],
        sweeps=1,
        chunk_size=2048,
        overlap=384,
        clock=_TickClock(),
    )

    assert case_rows == [
        {
            "schema_version": "1.0",
            "arm": "real",
            "case_id": "real-a",
            "sweep_index": 0,
            "order_index": 0,
            "success": True,
            "diagnostic_code": "ok",
            "terminal_stage": "complete",
            "detect_ns": 1,
            "extract_ns": 2,
            "sanitize_ns": 3,
            "chunk_plan_ns": 4,
            "combined_ns": 20,
            "input_bytes": static["input_bytes"],
            "page_count": static["page_count"],
            "extracted_codepoints": static["extracted_codepoints"],
            "estimated_tokens": static["estimated_tokens"],
        }
    ]


def test_timing_sweep_rates_are_ratio_of_sums_with_success_denominator(monkeypatch):
    from apps.chat.evals.offline import document_pipeline_runner as runner

    static_records = [
        _static_record("real", "real-a", success=True),
        _static_record("real", "real-b", success=False),
    ]

    def observe(case, **kwargs):
        success = case["case_id"] == "real-a"
        return {
            "success": success,
            "diagnostic_code": "ok" if success else "parser_error",
            "terminal_stage": "complete" if success else "extract",
            "detect_ns": 1,
            "extract_ns": 1,
            "sanitize_ns": 1 if success else None,
            "chunk_plan_ns": 1 if success else None,
            "combined_ns": 100 if success else 300,
        }

    monkeypatch.setattr(runner, "_observe_pipeline", observe)
    _, sweeps = runner._run_timing_sweeps(
        {
            "real": [
                {"arm": "real", "case_id": "real-a"},
                {"arm": "real", "case_id": "real-b"},
            ],
            "synthetic": [],
        },
        static_records,
        sweeps=1,
        chunk_size=2048,
        overlap=384,
        clock=_TickClock(),
    )

    row = sweeps[0]
    assert row["case_combined_sum_ns"] == 400
    assert row["successful_case_combined_sum_ns"] == 100
    assert row["attempted_documents_per_second"] == pytest.approx(5_000_000)
    assert row["effective_successful_documents_per_second"] == pytest.approx(2_500_000)
    assert row["success_conditioned_documents_per_second"] == pytest.approx(10_000_000)
    assert row["effective_pages_per_second"] == pytest.approx(5_000_000)
    assert row["success_conditioned_pages_per_second"] == pytest.approx(20_000_000)
    assert row["milliseconds_per_attempted_document"] == pytest.approx(0.0002)
    assert row["milliseconds_per_successful_document"] == pytest.approx(0.0001)


def test_memory_passes_use_fresh_tracing_lifecycle_three_times(monkeypatch):
    from apps.chat.evals.offline import document_pipeline_runner as runner

    cases = {
        "real": [{"arm": "real", "case_id": "real-a"}],
        "synthetic": [{"arm": "synthetic", "case_id": "synthetic-a"}],
    }
    events = []
    peaks = iter((101, 102, 103, 201, 202, 203))

    monkeypatch.setattr(runner.gc, "collect", lambda: events.append("gc"))
    monkeypatch.setattr(runner.tracemalloc, "start", lambda: events.append("start"))
    monkeypatch.setattr(
        runner.tracemalloc, "reset_peak", lambda: events.append("reset_peak")
    )
    monkeypatch.setattr(
        runner.tracemalloc,
        "get_traced_memory",
        lambda: events.append("get_peak") or (0, next(peaks)),
    )
    monkeypatch.setattr(runner.tracemalloc, "stop", lambda: events.append("stop"))

    def execute(case, *, chunk_size, overlap):
        events.append(("pipeline", case["case_id"], chunk_size, overlap))
        return {"success": True, "diagnostic_code": "ok"}

    monkeypatch.setattr(runner, "_run_pipeline_unmeasured", execute)

    records = runner._run_memory_passes(
        cases, memory_repeats=3, chunk_size=2048, overlap=384
    )

    assert len(records) == 6
    assert [row["peak_python_traced_bytes"] for row in records] == [
        101,
        102,
        103,
        201,
        202,
        203,
    ]
    assert [row["memory_index"] for row in records] == [0, 1, 2, 0, 1, 2]
    for offset in range(0, len(events), 6):
        assert events[offset : offset + 6][0:3] == ["gc", "start", "reset_peak"]
        assert events[offset + 3][0] == "pipeline"
        assert events[offset + 4 : offset + 6] == ["get_peak", "stop"]


def test_memory_pass_stops_tracing_when_pipeline_raises(monkeypatch):
    from apps.chat.evals.offline import document_pipeline_runner as runner

    events = []
    monkeypatch.setattr(runner.gc, "collect", lambda: events.append("gc"))
    monkeypatch.setattr(runner.tracemalloc, "start", lambda: events.append("start"))
    monkeypatch.setattr(
        runner.tracemalloc, "reset_peak", lambda: events.append("reset_peak")
    )
    monkeypatch.setattr(runner.tracemalloc, "stop", lambda: events.append("stop"))
    monkeypatch.setattr(
        runner,
        "_run_pipeline_unmeasured",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError, match="boom"):
        runner._run_memory_passes(
            {"real": [_one_page_case()], "synthetic": []},
            memory_repeats=1,
            chunk_size=2048,
            overlap=384,
        )

    assert events == ["gc", "start", "reset_peak", "stop"]


def test_memory_pass_retains_failed_real_peak_and_rejects_synthetic(monkeypatch):
    from apps.chat.evals.offline import document_pipeline_runner as runner

    monkeypatch.setattr(runner.gc, "collect", lambda: None)
    monkeypatch.setattr(runner.tracemalloc, "start", lambda: None)
    monkeypatch.setattr(runner.tracemalloc, "reset_peak", lambda: None)
    monkeypatch.setattr(runner.tracemalloc, "get_traced_memory", lambda: (0, 999))
    monkeypatch.setattr(runner.tracemalloc, "stop", lambda: None)
    monkeypatch.setattr(
        runner,
        "_run_pipeline_unmeasured",
        lambda *args, **kwargs: {
            "success": False,
            "diagnostic_code": "invalid_pdf",
        },
    )

    records = runner._run_memory_passes(
        {"real": [_one_page_case()], "synthetic": []},
        memory_repeats=1,
        chunk_size=2048,
        overlap=384,
    )
    assert records[0]["success"] is False
    assert records[0]["peak_python_traced_bytes"] == 999

    with pytest.raises(runner.BenchmarkIntegrityError, match="synthetic memory"):
        runner._run_memory_passes(
            {"real": [], "synthetic": [_one_page_case("synthetic")]},
            memory_repeats=1,
            chunk_size=2048,
            overlap=384,
        )


def test_prepare_synthetic_cases_verifies_reviewed_hashes_and_pages():
    from apps.chat.evals.offline.document_pipeline_runner import (
        BenchmarkIntegrityError,
        _prepare_synthetic_cases,
    )

    pdf_bytes, normalized_text = generate_synthetic_pdf(1)
    review = {
        "protocol": {
            "cases": [
                {
                    "page_count": 1,
                    "expected_page_count": 1,
                    "pdf_sha256": hashlib.sha256(pdf_bytes).hexdigest(),
                    "normalized_text_sha256": hashlib.sha256(
                        normalized_text.encode("utf-8")
                    ).hexdigest(),
                }
            ]
        }
    }

    cases = _prepare_synthetic_cases(review)

    assert cases == [
        {
            "arm": "synthetic",
            "case_id": "synthetic-001",
            "pdf_bytes": pdf_bytes,
            "raw_sha256": hashlib.sha256(pdf_bytes).hexdigest(),
            "page_count": 1,
            "expected_output_sha256": hashlib.sha256(
                normalized_text.encode("utf-8")
            ).hexdigest(),
        }
    ]

    review["protocol"]["cases"][0]["pdf_sha256"] = "0" * 64
    with pytest.raises(BenchmarkIntegrityError, match="synthetic PDF hash"):
        _prepare_synthetic_cases(review)


def test_run_document_benchmark_gates_production_and_returns_canonical_result(
    tmp_path: Path, monkeypatch
):
    from apps.chat.evals.offline import document_pipeline_runner as runner
    from apps.chat.services import rag_evidence

    pdf_bytes, normalized_text = generate_synthetic_pdf(1)
    (tmp_path / "real-input.pdf").write_bytes(pdf_bytes)
    inventory = _inventory_for(("real-001", pdf_bytes))
    review = {
        "status": "approved",
        "inventory_hash": "1" * 64,
        "protocol_hash": "2" * 64,
        "protocol": {
            "generator_version": "aquillm-ascii-pdf-v1",
            "cases": [
                {
                    "page_count": 1,
                    "expected_page_count": 1,
                    "pdf_sha256": hashlib.sha256(pdf_bytes).hexdigest(),
                    "normalized_text_sha256": hashlib.sha256(
                        normalized_text.encode("utf-8")
                    ).hexdigest(),
                }
            ],
        },
    }
    events = []
    real_estimate = rag_evidence._estimate_tokens
    real_detect = runner.parsers.detect_ingest_type

    monkeypatch.setattr(
        runner,
        "validate_document_review",
        lambda supplied_review, supplied_inventory: events.append("review"),
    )

    def estimate(text):
        events.append("estimate")
        return real_estimate(text)

    def detect(*args, **kwargs):
        events.append("detect")
        return real_detect(*args, **kwargs)

    monkeypatch.setattr(rag_evidence, "_estimate_tokens", estimate)
    monkeypatch.setattr(runner.parsers, "detect_ingest_type", detect)

    result = runner.run_document_benchmark(
        tmp_path,
        inventory,
        review,
        network_audit={"guard": "socket", "total_attempts": 0, "details": []},
        sweeps=2,
        memory_repeats=1,
        clock=_TickClock(),
    )

    assert events[0:2] == ["review", "estimate"]
    assert events.index("estimate") < events.index("detect")
    assert set(result) == {
        "manifest",
        "inventory",
        "review",
        "real_records",
        "synthetic_records",
        "timing_cases",
        "timing_sweeps",
        "memory_records",
        "aggregate",
    }
    assert len(result["real_records"]) == 1
    assert len(result["synthetic_records"]) == 1
    assert len(result["timing_cases"]) == 4
    assert len(result["timing_sweeps"]) == 4
    assert len(result["memory_records"]) == 2
    assert result["aggregate"]["real"]["success_count"] == 1
    assert result["aggregate"]["synthetic"]["success_count"] == 1
    assert result["aggregate"]["network_audit"]["scope"] == (
        "zero connection attempts observed through the configured process-local "
        "socket guard"
    )
    manifest = result["manifest"]
    assert manifest["chunk_configuration"] == {
        "chunk_size_codepoints": 2048,
        "overlap_codepoints": 384,
        "pitch_codepoints": 1664,
    }
    assert manifest["token_estimator"] == {
        "name": "production_character_estimator",
        "algorithm": "max(1, len(text) // 4)",
    }
    assert manifest["execution"] == {
        "mode": "single_thread_sequential",
        "warmups_per_arm": 1,
        "timing_sweeps_per_arm": 2,
        "memory_repeats_per_case": 1,
    }
    assert "hostname" not in repr(manifest).lower()
    assert "username" not in repr(manifest).lower()
    assert str(tmp_path) not in repr(manifest)
