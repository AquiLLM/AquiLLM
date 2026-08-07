from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

import pytest
from pypdf.errors import FileNotDecryptedError, PdfReadError

from apps.chat.evals.offline.document_pipeline_schema import (
    build_document_record,
    generate_synthetic_pdf,
)
from apps.chat.evals.offline.schema import sha256_canonical_text
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


def test_resolve_real_corpus_rejects_pdf_symlink_or_reparse_point_before_read(
    tmp_path: Path,
):
    from apps.chat.evals.offline.document_pipeline_runner import resolve_real_corpus

    pdf_bytes, _ = generate_synthetic_pdf(1)
    target = tmp_path / "private-target.bin"
    target.write_bytes(pdf_bytes)
    linked_pdf = tmp_path / "linked-input.pdf"
    try:
        linked_pdf.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {type(exc).__name__}")

    if os.name == "nt":
        assert linked_pdf.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT

    with pytest.raises(ValueError, match="linked_pdf") as raised:
        resolve_real_corpus(
            tmp_path,
            _inventory_for(("real-001", pdf_bytes)),
        )

    assert str(target) not in str(raised.value)
    assert str(linked_pdf) not in str(raised.value)


def test_resolve_real_corpus_rejects_link_signal_before_read(
    tmp_path: Path, monkeypatch
):
    from apps.chat.evals.offline.document_pipeline_runner import resolve_real_corpus

    pdf_bytes, _ = generate_synthetic_pdf(1)
    candidate = tmp_path / "candidate.pdf"
    candidate.write_bytes(pdf_bytes)
    real_is_symlink = Path.is_symlink
    real_read_bytes = Path.read_bytes

    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda self: self == candidate or real_is_symlink(self),
    )

    def guarded_read(self):
        if self == candidate:
            raise AssertionError("linked PDF must be rejected before read")
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)

    with pytest.raises(ValueError, match="linked_pdf"):
        resolve_real_corpus(
            tmp_path,
            _inventory_for(("real-001", pdf_bytes)),
        )


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


def test_timed_and_unmeasured_paths_share_one_execution_core(monkeypatch):
    from apps.chat.evals.offline import document_pipeline_runner as runner
    from apps.documents.services.text_chunk_plan import plan_text_chunks as real_plan

    case = _one_page_case()
    core_calls = []
    stage_calls = []
    real_core = runner._execute_pipeline_core

    def core(*args, **kwargs):
        core_calls.append(True)
        return real_core(*args, **kwargs)

    class Payload:
        full_text = "parser output"

    monkeypatch.setattr(runner, "_execute_pipeline_core", core)
    monkeypatch.setattr(
        runner.parsers,
        "detect_ingest_type",
        lambda *args, **kwargs: (stage_calls.append("detect") or "document"),
    )
    monkeypatch.setattr(
        runner.parsers,
        "extract_primary_text_payload",
        lambda *args, **kwargs: (stage_calls.append("extract") or Payload()),
    )
    monkeypatch.setattr(
        runner,
        "sanitize_db_text",
        lambda value: (stage_calls.append("sanitize") or "changed\x00output"),
    )

    def plan(text, *, chunk_size, overlap):
        stage_calls.append("chunk_plan")
        return real_plan(text, chunk_size=chunk_size, overlap=overlap)

    monkeypatch.setattr(runner, "plan_text_chunks", plan)

    timed = runner._observe_pipeline(
        case, chunk_size=2048, overlap=384, clock=_TickClock()
    )
    execution = runner._run_pipeline_unmeasured(
        case, chunk_size=2048, overlap=384
    )
    unmeasured = runner._finalize_pipeline_execution(execution, case)

    assert len(core_calls) == 2
    assert stage_calls == [
        "detect",
        "extract",
        "sanitize",
        "chunk_plan",
        "detect",
        "extract",
        "sanitize",
        "chunk_plan",
    ]
    for field in (
        "success",
        "diagnostic_code",
        "_sanitized_text",
        "_chunk_specs",
    ):
        assert timed[field] == unmeasured[field]


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
        lambda *args, **kwargs: events.append("diagnostic") or "parser_error",
    )

    runner.run_document_case(
        _one_page_case(), chunk_size=2048, overlap=384, clock=clock
    )

    assert events == ["clock", "clock", "clock", "clock", "diagnostic"]


@pytest.mark.parametrize("failure", [False, True])
def test_combined_timer_closes_at_pipeline_terminal_boundary(monkeypatch, failure):
    from apps.chat.evals.offline import document_pipeline_runner as runner

    events = []
    tick = iter(range(20))
    real_core = runner._execute_pipeline_core
    real_finalize = runner._finalize_pipeline_execution
    real_plan = runner.plan_text_chunks

    def clock():
        events.append("clock")
        return next(tick)

    def build_execution(
        *, success, terminal_stage, exception, sanitized_text, chunk_specs
    ):
        events.append("result")
        return {
            "success": success,
            "terminal_stage": terminal_stage,
            "_exception": exception,
            "_sanitized_text": sanitized_text,
            "_chunk_specs": chunk_specs,
        }

    def core(*args, **kwargs):
        result = real_core(*args, **kwargs)
        events.append("core_return")
        return result

    def finalize(execution, case):
        events.append("finalize")
        return real_finalize(execution, case)

    if failure:
        def fail_detect(*args, **kwargs):
            events.append("detect")
            raise RuntimeError("fail")

        monkeypatch.setattr(runner.parsers, "detect_ingest_type", fail_detect)
    else:
        def plan(*args, **kwargs):
            events.append("chunk")
            return real_plan(*args, **kwargs)

        monkeypatch.setattr(runner, "plan_text_chunks", plan)
    monkeypatch.setattr(
        runner, "_build_pipeline_execution", build_execution, raising=False
    )
    monkeypatch.setattr(runner, "_execute_pipeline_core", core)
    monkeypatch.setattr(runner, "_finalize_pipeline_execution", finalize)
    monkeypatch.setattr(
        runner,
        "_diagnostic_for",
        lambda *args, **kwargs: events.append("diagnostic") or "parser_error",
    )

    runner._observe_pipeline(
        _one_page_case(), chunk_size=2048, overlap=384, clock=clock
    )

    expected_terminal = [
        "detect" if failure else "chunk",
        "clock",
        "clock",
        "result",
        "core_return",
        "finalize",
    ]
    if failure:
        expected_terminal.append("diagnostic")
    assert events[-len(expected_terminal) :] == expected_terminal


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


def _case_for_static(static: dict) -> dict:
    return {
        "arm": static["arm"],
        "case_id": static["case_id"],
        "pdf_bytes": b"x" * static["input_bytes"],
        "page_count": static["page_count"],
    }


def _observed_outputs(static: dict) -> dict:
    text = "abcdefgh" if static["success"] else None
    return {
        "_sanitized_text": text,
        "_chunk_specs": (
            plan_text_chunks(text, chunk_size=5, overlap=1) if text else None
        ),
    }


def test_timing_sweeps_do_not_add_a_second_warmup_and_rotate_case_order(monkeypatch):
    from apps.chat.evals.offline import document_pipeline_runner as runner

    static_records = [
        _static_record("real", "real-a", success=True),
        _static_record("real", "real-b", success=False),
        _static_record("synthetic", "synthetic-a", success=True),
        _static_record("synthetic", "synthetic-b", success=True),
    ]
    cases_by_arm = {
        "real": [
            _case_for_static(static_records[1]),
            _case_for_static(static_records[0]),
        ],
        "synthetic": [
            _case_for_static(static_records[3]),
            _case_for_static(static_records[2]),
        ],
    }
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
            **_observed_outputs(static),
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
    assert events[6:8] == [
        ("synthetic", "synthetic-a"),
        ("synthetic", "synthetic-b"),
    ]
    assert len(events) == 12
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
            **_observed_outputs(static),
        },
    )

    case_rows, _ = runner._run_timing_sweeps(
        {"real": [_case_for_static(static)], "synthetic": []},
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


def test_timing_sweep_fails_closed_when_successful_output_drifts(monkeypatch):
    from apps.chat.evals.offline import document_pipeline_runner as runner

    static = _static_record("real", "real-a", success=True)
    changed_text = "changed successful output"
    case = {
        "arm": "real",
        "case_id": "real-a",
        "pdf_bytes": b"x" * static["input_bytes"],
        "page_count": static["page_count"],
    }
    def observe(*args, **kwargs):
        text = changed_text
        return {
            "success": True,
            "diagnostic_code": "ok",
            "terminal_stage": "complete",
            "detect_ns": 1,
            "extract_ns": 2,
            "sanitize_ns": 3,
            "chunk_plan_ns": 4,
            "combined_ns": 20,
            "_sanitized_text": text,
            "_chunk_specs": plan_text_chunks(text, chunk_size=5, overlap=1),
        }

    monkeypatch.setattr(runner, "_observe_pipeline", observe)

    with pytest.raises(runner.BenchmarkIntegrityError, match="timing output drift"):
        runner._run_timing_sweeps(
            {"real": [case], "synthetic": []},
            [static],
            sweeps=1,
            chunk_size=2048,
            overlap=384,
            clock=_TickClock(),
        )


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
            **_observed_outputs(
                next(row for row in static_records if row["case_id"] == case["case_id"])
            ),
        }

    monkeypatch.setattr(runner, "_observe_pipeline", observe)
    _, sweeps = runner._run_timing_sweeps(
        {
            "real": [
                _case_for_static(static_records[0]),
                _case_for_static(static_records[1]),
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

    static_records = [
        _static_record("real", "real-a", success=True),
        _static_record("synthetic", "synthetic-a", success=True),
    ]
    cases = {
        "real": [_case_for_static(static_records[0])],
        "synthetic": [_case_for_static(static_records[1])],
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
        static = next(row for row in static_records if row["arm"] == case["arm"])
        return {
            "success": True,
            "terminal_stage": "complete",
            "_exception": None,
            **_observed_outputs(static),
        }

    monkeypatch.setattr(runner, "_run_pipeline_unmeasured", execute)

    records = runner._run_memory_passes(
        cases,
        static_records,
        memory_repeats=3,
        chunk_size=2048,
        overlap=384,
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


def test_memory_pass_rejects_caller_owned_active_tracing_before_gc(monkeypatch):
    from apps.chat.evals.offline import document_pipeline_runner as runner

    static = _static_record("real", "real-a", success=True)
    events = []
    monkeypatch.setattr(runner.tracemalloc, "is_tracing", lambda: True)
    monkeypatch.setattr(runner.gc, "collect", lambda: events.append("gc"))
    monkeypatch.setattr(runner.tracemalloc, "start", lambda: events.append("start"))
    monkeypatch.setattr(runner.tracemalloc, "stop", lambda: events.append("stop"))
    monkeypatch.setattr(
        runner,
        "_run_pipeline_unmeasured",
        lambda *args, **kwargs: events.append("pipeline"),
    )

    with pytest.raises(runner.BenchmarkIntegrityError, match="already active"):
        runner._run_memory_passes(
            {"real": [_case_for_static(static)], "synthetic": []},
            [static],
            memory_repeats=1,
            chunk_size=2048,
            overlap=384,
        )

    assert events == []


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
            [],
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
            "terminal_stage": "extract",
            "_exception": PdfReadError("invalid PDF"),
            "_sanitized_text": None,
            "_chunk_specs": None,
        },
    )

    real_case = _one_page_case()
    real_static = build_document_record(
        arm="real",
        case_id=real_case["case_id"],
        input_bytes=len(real_case["pdf_bytes"]),
        page_count=real_case["page_count"],
        success=False,
        diagnostic_code="invalid_pdf",
        sanitized_text=None,
        chunk_specs=None,
    )

    records = runner._run_memory_passes(
        {"real": [real_case], "synthetic": []},
        [real_static],
        memory_repeats=1,
        chunk_size=2048,
        overlap=384,
    )
    assert records[0]["success"] is False
    assert records[0]["peak_python_traced_bytes"] == 999

    synthetic_case = _one_page_case("synthetic")
    synthetic_static = build_document_record(
        arm="synthetic",
        case_id=synthetic_case["case_id"],
        input_bytes=len(synthetic_case["pdf_bytes"]),
        page_count=synthetic_case["page_count"],
        success=False,
        diagnostic_code="invalid_pdf",
        sanitized_text=None,
        chunk_specs=None,
    )
    with pytest.raises(runner.BenchmarkIntegrityError, match="synthetic memory"):
        runner._run_memory_passes(
            {"real": [], "synthetic": [synthetic_case]},
            [synthetic_static],
            memory_repeats=1,
            chunk_size=2048,
            overlap=384,
        )


@pytest.mark.parametrize(
    ("frozen_success", "observed_success", "observed_text"),
    [
        (True, False, None),
        (False, True, "abcdefgh"),
        (True, True, "changed successful output"),
    ],
)
def test_memory_pass_fails_closed_on_static_outcome_or_output_drift(
    monkeypatch, frozen_success, observed_success, observed_text
):
    from apps.chat.evals.offline import document_pipeline_runner as runner

    static = _static_record("real", "real-a", success=frozen_success)
    case = {
        "arm": "real",
        "case_id": "real-a",
        "pdf_bytes": b"x" * static["input_bytes"],
        "page_count": static["page_count"],
    }
    monkeypatch.setattr(runner.gc, "collect", lambda: None)
    monkeypatch.setattr(runner.tracemalloc, "start", lambda: None)
    monkeypatch.setattr(runner.tracemalloc, "reset_peak", lambda: None)
    monkeypatch.setattr(runner.tracemalloc, "get_traced_memory", lambda: (0, 123))
    monkeypatch.setattr(runner.tracemalloc, "stop", lambda: None)
    monkeypatch.setattr(
        runner,
        "_run_pipeline_unmeasured",
        lambda *args, **kwargs: {
            "success": observed_success,
            "terminal_stage": "complete" if observed_success else "extract",
            "_exception": None if observed_success else PdfReadError("invalid PDF"),
            "_sanitized_text": observed_text,
            "_chunk_specs": (
                plan_text_chunks(observed_text, chunk_size=5, overlap=1)
                if observed_text
                else None
            ),
        },
    )

    with pytest.raises(runner.BenchmarkIntegrityError, match="memory output drift"):
        runner._run_memory_passes(
            {"real": [case], "synthetic": []},
            [static],
            memory_repeats=1,
            chunk_size=2048,
            overlap=384,
        )


def test_memory_metrics_are_built_only_after_tracing_stops(monkeypatch):
    from apps.chat.evals.offline import document_pipeline_runner as runner

    static = _static_record("real", "real-a", success=True)
    case = {
        "arm": "real",
        "case_id": "real-a",
        "pdf_bytes": b"x" * static["input_bytes"],
        "page_count": static["page_count"],
    }
    tracing = {"active": False}
    real_build = runner.build_document_record
    real_finalize = runner._finalize_pipeline_execution
    monkeypatch.setattr(runner.gc, "collect", lambda: None)
    monkeypatch.setattr(
        runner.tracemalloc, "start", lambda: tracing.__setitem__("active", True)
    )
    monkeypatch.setattr(runner.tracemalloc, "reset_peak", lambda: None)
    monkeypatch.setattr(runner.tracemalloc, "get_traced_memory", lambda: (0, 123))
    monkeypatch.setattr(
        runner.tracemalloc, "stop", lambda: tracing.__setitem__("active", False)
    )
    monkeypatch.setattr(
        runner,
        "_run_pipeline_unmeasured",
        lambda *args, **kwargs: {
            "success": True,
            "terminal_stage": "complete",
            "_exception": None,
            "_sanitized_text": "abcdefgh",
            "_chunk_specs": plan_text_chunks("abcdefgh", chunk_size=5, overlap=1),
        },
    )

    def finalize(execution, current_case):
        assert tracing["active"] is False
        return real_finalize(execution, current_case)

    def build(**kwargs):
        assert tracing["active"] is False
        return real_build(**kwargs)

    monkeypatch.setattr(runner, "build_document_record", build)
    monkeypatch.setattr(runner, "_finalize_pipeline_execution", finalize)

    runner._run_memory_passes(
        {"real": [case], "synthetic": []},
        [static],
        memory_repeats=1,
        chunk_size=2048,
        overlap=384,
    )


def _zero_network_audit() -> dict:
    return {
        "guard": "deny_network",
        "scope": "fixture_generation_through_artifact_validation",
        "total_attempts": 0,
        "details": [],
    }


@pytest.mark.parametrize(
    "audit",
    [
        {},
        {**_zero_network_audit(), "extra": True},
        {**_zero_network_audit(), "guard": "socket"},
        {**_zero_network_audit(), "scope": "timing_only"},
        {**_zero_network_audit(), "total_attempts": False},
        {**_zero_network_audit(), "details": "none"},
        {
            **_zero_network_audit(),
            "total_attempts": 1,
            "details": [{"operation": "socket.socket.connect", "address": "secret"}],
        },
        {
            **_zero_network_audit(),
            "total_attempts": 1,
            "details": [{"operation": "urllib.request.urlopen"}],
        },
        {
            **_zero_network_audit(),
            "total_attempts": 0,
            "details": [{"operation": "socket.create_connection"}],
        },
    ],
)
def test_network_audit_rejects_shape_identity_type_count_and_privacy_mutations(audit):
    from apps.chat.evals.offline.document_pipeline_runner import (
        BenchmarkIntegrityError,
        _validate_network_audit,
    )

    with pytest.raises(BenchmarkIntegrityError, match="network audit"):
        _validate_network_audit(audit)


def test_network_audit_accepts_exact_zero_attempt_contract_without_rewriting_scope():
    from apps.chat.evals.offline.document_pipeline_runner import _validate_network_audit

    audit = _zero_network_audit()

    assert _validate_network_audit(audit) == audit


@pytest.mark.parametrize(
    "operation",
    [
        "socket.socket.connect",
        "socket.socket.connect_ex",
        "socket.create_connection",
    ],
)
def test_network_audit_fails_closed_on_each_allowed_observed_operation(operation):
    from apps.chat.evals.offline.document_pipeline_runner import (
        BenchmarkIntegrityError,
        _validate_network_audit,
    )

    with pytest.raises(BenchmarkIntegrityError, match="connection attempt"):
        _validate_network_audit(
            {
                "guard": "deny_network",
                "scope": "fixture_generation_through_artifact_validation",
                "total_attempts": 1,
                "details": [{"operation": operation}],
            }
        )


def test_load_benchmark_inputs_binds_review_to_exact_inventory_and_review_sources(
    tmp_path: Path,
):
    from apps.chat.evals.offline.document_pipeline_runner import (
        _load_benchmark_inputs,
    )

    fixture_dir = Path(__file__).parents[1] / "evals" / "offline"
    inventory_path = tmp_path / "inventory.yaml"
    review_path = tmp_path / "review.yaml"
    inventory_path.write_bytes(
        (fixture_dir / "document_corpus_inventory.yaml").read_bytes()
    )
    review_path.write_bytes((fixture_dir / "document_corpus_review.yaml").read_bytes())

    loaded = _load_benchmark_inputs(inventory_path, review_path)

    assert loaded["inventory_source_hash"] == sha256_canonical_text(inventory_path)
    assert loaded["review_source_hash"] == sha256_canonical_text(review_path)
    assert loaded["inventory"]["inventory_id"] == "astro-test-real-v1"
    assert loaded["review"]["status"] == "approved"
    assert not ({"path", "filename", "content", "text"} & loaded["inventory"].keys())


@pytest.mark.parametrize("mutated_source", ["inventory", "review"])
def test_load_benchmark_inputs_fails_closed_on_source_byte_mutation(
    tmp_path: Path, mutated_source: str
):
    from apps.chat.evals.offline.document_pipeline_runner import (
        _load_benchmark_inputs,
    )

    fixture_dir = Path(__file__).parents[1] / "evals" / "offline"
    inventory_path = tmp_path / "inventory.yaml"
    review_path = tmp_path / "review.yaml"
    inventory_path.write_bytes(
        (fixture_dir / "document_corpus_inventory.yaml").read_bytes()
    )
    review_path.write_bytes((fixture_dir / "document_corpus_review.yaml").read_bytes())
    if mutated_source == "inventory":
        inventory_path.write_bytes(inventory_path.read_bytes() + b"\n# mutation\n")
    else:
        review_path.write_bytes(
            review_path.read_bytes().replace(
                b"status: approved", b"status: pending_independent_review"
            )
        )

    with pytest.raises(ValueError):
        _load_benchmark_inputs(inventory_path, review_path)


def test_manifest_labels_each_hash_family_with_its_actual_algorithm(monkeypatch):
    from apps.chat.evals.offline import document_pipeline_runner as runner

    monkeypatch.setattr(
        runner, "_source_state", lambda: {"commit": "a" * 40, "dirty": False}
    )
    monkeypatch.setattr(runner, "_source_hashes", lambda: {"runner": "b" * 64})
    manifest = runner._build_manifest(
        inventory={"cases": []},
        review={
            "inventory_hash": "c" * 64,
            "protocol_hash_algorithm": "sha256-canonical-json-v1",
            "protocol_hash": "d" * 64,
        },
        inventory_source_hash="c" * 64,
        review_source_hash="e" * 64,
        synthetic_cases=[
            {
                "case_id": "synthetic-001",
                "raw_sha256": "f" * 64,
                "expected_output_sha256": "1" * 64,
                "page_count": 1,
            }
        ],
        network_audit=_zero_network_audit(),
        chunk_size=2048,
        overlap=384,
        sweeps=30,
        memory_repeats=3,
    )

    assert manifest["hashes"] == {
        "source_code": {
            "algorithm": "sha256-utf8-lf-v1",
            "values": {"runner": "b" * 64},
        },
        "inventory_source": {
            "algorithm": "sha256-utf8-lf-v1",
            "sha256": "c" * 64,
        },
        "review_source": {
            "algorithm": "sha256-utf8-lf-v1",
            "sha256": "e" * 64,
        },
        "synthetic_protocol": {
            "algorithm": "sha256-canonical-json-v1",
            "sha256": "d" * 64,
        },
    }
    assert manifest["synthetic_inputs"][0]["pdf_hash_algorithm"] == (
        "sha256-raw-bytes-v1"
    )
    cr_text = "left\rright"
    exact_utf8_hash = hashlib.sha256(cr_text.encode("utf-8")).hexdigest()
    lf_canonical_hash = hashlib.sha256(
        cr_text.replace("\r", "\n").encode("utf-8")
    ).hexdigest()
    assert exact_utf8_hash != lf_canonical_hash
    assert manifest["synthetic_inputs"][0]["expected_output_hash_algorithm"] == (
        "sha256-utf8-bytes-v1"
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
        "protocol_hash_algorithm": "sha256-canonical-json-v1",
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

    def load_inputs(inventory_path, review_path):
        events.append("review")
        return {
            "inventory": inventory,
            "review": review,
            "inventory_source_hash": "1" * 64,
            "review_source_hash": "3" * 64,
        }

    monkeypatch.setattr(runner, "_load_benchmark_inputs", load_inputs)

    def estimate(text):
        events.append("estimate")
        return real_estimate(text)

    def detect(*args, **kwargs):
        events.append(("detect", args[0]))
        return real_detect(*args, **kwargs)

    monkeypatch.setattr(rag_evidence, "_estimate_tokens", estimate)
    monkeypatch.setattr(runner.parsers, "detect_ingest_type", detect)

    result = runner.run_document_benchmark(
        tmp_path,
        Path("inventory.yaml"),
        Path("review.yaml"),
        network_audit=_zero_network_audit(),
        sweeps=2,
        memory_repeats=1,
        clock=_TickClock(),
    )

    assert events[0:2] == ["review", "estimate"]
    detect_order = [event[1] for event in events if isinstance(event, tuple)]
    assert detect_order == [
        "real-001.pdf",
        "real-001.pdf",
        "real-001.pdf",
        "synthetic-001.pdf",
        "synthetic-001.pdf",
        "synthetic-001.pdf",
        "real-001.pdf",
        "synthetic-001.pdf",
    ]
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
        "fixture_generation_through_artifact_validation"
    )
    manifest = result["manifest"]
    assert manifest["network_audit_statement"] == (
        "zero connection attempts observed through the configured process-local "
        "socket guard"
    )
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
        "warmup_role": "static_record_pass",
        "timing_sweeps_per_arm": 2,
        "memory_repeats_per_case": 1,
    }
    assert "hostname" not in repr(manifest).lower()
    assert "username" not in repr(manifest).lower()
    assert str(tmp_path) not in repr(manifest)


@pytest.mark.parametrize(
    "invalid_config",
    [
        {"sweeps": 0},
        {"sweeps": -1},
        {"sweeps": True},
        {"sweeps": 1.0},
        {"memory_repeats": 0},
        {"memory_repeats": -1},
        {"memory_repeats": True},
        {"memory_repeats": 1.0},
        {"chunk_size": 0},
        {"chunk_size": -1},
        {"chunk_size": True},
        {"chunk_size": 8.0},
        {"overlap": -1},
        {"overlap": True},
        {"overlap": 1.0},
        {"chunk_size": 8, "overlap": 8},
        {"chunk_size": 8, "overlap": 9},
    ],
)
def test_run_document_benchmark_rejects_invalid_config_before_source_access(
    tmp_path: Path, monkeypatch, invalid_config
):
    from apps.chat.evals.offline import document_pipeline_runner as runner

    source_access = []

    def fail_source_access(*args, **kwargs):
        source_access.append(True)
        raise AssertionError("invalid config must fail before source access")

    monkeypatch.setattr(runner, "_load_benchmark_inputs", fail_source_access)

    with pytest.raises(runner.BenchmarkIntegrityError, match="configuration"):
        runner.run_document_benchmark(
            tmp_path,
            Path("inventory.yaml"),
            Path("review.yaml"),
            network_audit=_zero_network_audit(),
            **invalid_config,
        )

    assert source_access == []
