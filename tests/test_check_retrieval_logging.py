# ruff: noqa: E501
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "check_retrieval_logging.py"
EXPECTED_LANE_PATHS = (
    "aquillm/lib/knowledge_graph/query_extractor/client.py",
    "aquillm/lib/knowledge_graph/query_extractor/service.py",
    "aquillm/apps/knowledge_graph/retrieval/direct_seed_repository.py",
    "aquillm/apps/knowledge_graph/retrieval/direct_seed_resolution.py",
    "aquillm/apps/knowledge_graph/retrieval/query_embedding.py",
)


def _module():
    spec = importlib.util.spec_from_file_location("check_retrieval_logging", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scan(source: str):
    return _module().scan_source(path=Path("lane.py"), source=source)


def test_checker_has_an_explicit_lane_allowlist_and_current_lane_is_clean() -> None:
    module = _module()
    assert module.LANE_PATHS == EXPECTED_LANE_PATHS
    assert module.find_violations(REPO) == ()


def test_checker_allows_only_fixed_event_and_redacted_structured_fields() -> None:
    source = """
import structlog
from lib.retrieval_redaction import RetrievalLogReason, retrieval_log_fields
logger = structlog.stdlib.get_logger(__name__)
result_count = len(())
elapsed_ms = 2.5
logger.info(
    "obs.rag.extract_failed",
    **retrieval_log_fields(
        reason=RetrievalLogReason.UPSTREAM_UNAVAILABLE,
        count=result_count,
        elapsed_ms=elapsed_ms,
    ),
)
logger.info(
    "obs.rag.extract_completed",
    **retrieval_log_fields(
        reason=RetrievalLogReason.COMPLETED,
        count=len(()),
        elapsed_ms=2.5,
    ),
)
"""
    assert _scan(source) == ()


@pytest.mark.parametrize(
    "source",
    (
        "from lib.retrieval_redaction import retrieval_log_fields\ncount = query\n"
        'logger.info("obs.rag.search", **retrieval_log_fields('
        "reason=RetrievalLogReason.COMPLETED, count=count, elapsed_ms=1))",
        "from lib.retrieval_redaction import retrieval_log_fields\n"
        "count, other = (query, safe)\n"
        'logger.info("obs.rag.search", **retrieval_log_fields('
        "reason=RetrievalLogReason.COMPLETED, count=count, elapsed_ms=1))",
        "from lib.retrieval_redaction import retrieval_log_fields\n"
        "for count in query: pass\n"
        'logger.info("obs.rag.search", **retrieval_log_fields('
        "reason=RetrievalLogReason.COMPLETED, count=count, elapsed_ms=1))",
        'log.info("obs.rag.search", query=query)',
        'self.audit.info("obs.rag.search", body=body)',
        'logger.info("obs.rag.search", reason=RetrievalLogReason.COMPLETED, '
        "count=count, elapsed_ms=1)",
        "def retrieval_log_fields(**fields): return fields\n"
        'log.info("obs.rag.search", **retrieval_log_fields('
        "reason=RetrievalLogReason.COMPLETED, count=count, elapsed_ms=1))",
        "from lib.retrieval_redaction import retrieval_log_fields\n"
        "retrieval_log_fields = fake\n"
        'log.info("obs.rag.search", **retrieval_log_fields('
        "reason=RetrievalLogReason.COMPLETED, count=count, elapsed_ms=1))",
        'emit = logger.info\nemit("obs.rag.search", query=query)',
    ),
)
def test_checker_rejects_taint_aliases_dynamic_receivers_and_nonhelper_shape(
    source: str,
) -> None:
    assert len(_scan(source)) == 1


@pytest.mark.parametrize(
    "binding",
    (
        "emit, other = (logger.info, safe)",
        "for emit in (logger.info,): pass",
    ),
)
def test_checker_rejects_destructured_and_loop_logger_aliases(binding: str) -> None:
    source = f'{binding}\nemit("obs.rag.search", query=query)'
    violations = _scan(source)
    assert len(violations) == 1
    assert violations[0].line == 2


def test_checker_does_not_taint_safe_destructuring_sibling() -> None:
    source = (
        "emit, callback = (logger.info, worker)\n"
        'callback("obs.rag.search", query=query)'
    )
    assert _scan(source) == ()


def test_review_regressions_live_only_in_planned_test_modules() -> None:
    extra = (
        REPO
        / "aquillm/apps/knowledge_graph/tests/test_direct_seed_review_regressions.py"
    )
    assert not extra.exists()


@pytest.mark.parametrize(
    "call",
    (
        'logger.info("obs.rag.search", query=query)',
        'logger.info("obs.rag.search", body=body)',
        'logger.info("obs.rag.search", exact_terms=exact_terms)',
        'logger.warning("obs.rag.failed", reason=response.text)',
        'logger.warning("obs.rag.failed", reason=str(exc))',
        'logger.info(f"obs.rag.{query}")',
        'logger.info("query=%s", query)',
        "logger.info(event, reason=RetrievalLogReason.COMPLETED)",
        'logger.info("obs.rag.search", **fields)',
        'logger.info("obs.rag.search", **make_fields())',
        'logger.exception("obs.rag.failed")',
        'audit_logger.info("obs.rag.search", query=query)',
        'self.logger.info("obs.rag.search", body=body)',
        'logging.info("obs.rag.search", query=query)',
        'logging.getLogger(__name__).warning("obs.rag.failed", body=body)',
        'logging.LoggerAdapter(logger, {}).info("obs.rag.search", exact_terms=terms)',
        "structlog.stdlib.get_logger(__name__).error("
        '"obs.rag.failed", reason=str(exc))',
    ),
)
def test_checker_rejects_payloads_exception_strings_and_dynamic_shapes(
    call: str,
) -> None:
    source = (
        f"import structlog\nlogger = structlog.stdlib.get_logger(__name__)\n{call}\n"
    )
    violations = _scan(source)
    assert len(violations) == 1
    assert violations[0].line == 3
    assert violations[0].reason


def test_checker_fails_closed_on_missing_or_invalid_lane_source(tmp_path: Path) -> None:
    module = _module()
    module.LANE_PATHS = ("missing.py",)
    assert module.find_violations(tmp_path)[0].reason == "missing_lane_path"

    invalid = tmp_path / "broken.py"
    invalid.write_text("logger.info(", encoding="utf-8")
    module.LANE_PATHS = ("broken.py",)
    assert module.find_violations(tmp_path)[0].reason == "invalid_python_source"


def test_checker_rejects_direct_functions_imported_from_standard_logging() -> None:
    violations = _scan(
        "from logging import info as audit_info\n"
        'audit_info("obs.rag.search", query=query)\n'
    )
    assert len(violations) == 1
    assert violations[0].line == 2


@pytest.mark.parametrize(
    "source",
    (
        'getattr(logger, "info")("obs.rag.search", query=query)',
        'emit = getattr(logger, "info")\nemit("obs.rag.search", query=query)',
        'level = "info"\ngetattr(logger, level)("obs.rag.search", query=query)',
    ),
)
def test_checker_rejects_static_and_dynamic_getattr_log_dispatch(source: str) -> None:
    assert len(_scan(source)) == 1


def test_checker_propagates_payload_taint_through_transformations() -> None:
    source = """
from lib.retrieval_redaction import RetrievalLogReason, retrieval_log_fields
prompt = query
count = ord(prompt[0]) + 1
logger.info(
    "obs.rag.search",
    **retrieval_log_fields(
        reason=RetrievalLogReason.COMPLETED, count=count, elapsed_ms=1.0
    ),
)
"""
    assert len(_scan(source)) == 1


def test_checker_allows_unrelated_info_methods() -> None:
    assert _scan('catalog.info("record", query=query)') == ()


def _structured_count_source(assignment: str) -> str:
    return f"""from lib.retrieval_redaction import RetrievalLogReason, retrieval_log_fields
def record(prompt):
    {assignment}
    logger.info(
        "obs.rag.search",
        **retrieval_log_fields(
            reason=RetrievalLogReason.COMPLETED,
            count=count,
            elapsed_ms=1.0,
        ),
    )
"""


def test_checker_resolves_transformed_count_assignment_before_allowing_name() -> None:
    assert len(_scan(_structured_count_source("count = ord(prompt[0])"))) == 1


@pytest.mark.parametrize("assignment", ("count:int=0", "count = len(())", "count = +2"))
def test_checker_allows_positively_proven_count_sources(assignment: str) -> None:
    assert _scan(_structured_count_source(assignment)) == ()


@pytest.mark.parametrize(
    "assignment",
    (
        "count = unknown",
        "count = True",
        "count = -1",
        "count = 1.5",
        "count = len([0]*ord(prompt[0]))",
    ),
)
def test_checker_rejects_unresolved_or_imprecise_count_sources(assignment: str) -> None:
    assert len(_scan(_structured_count_source(assignment))) == 1


def test_checker_does_not_resolve_count_across_function_or_parameter_scope() -> None:
    source = """
from lib.retrieval_redaction import RetrievalLogReason, retrieval_log_fields
def unrelated():
    count = 0
def record(count):
    logger.info("obs.rag.search", **retrieval_log_fields(reason=RetrievalLogReason.COMPLETED, count=count, elapsed_ms=1.0))
"""
    assert len(_scan(source)) == 1


def test_checker_accepts_independent_lexical_count_definitions() -> None:
    source = """
from lib.retrieval_redaction import RetrievalLogReason, retrieval_log_fields
def first(rows):
    count = 0
    logger.info("obs.rag.first", **retrieval_log_fields(reason=RetrievalLogReason.COMPLETED, count=count, elapsed_ms=1.0))
def second(rows):
    count = 0
    logger.info("obs.rag.second", **retrieval_log_fields(reason=RetrievalLogReason.COMPLETED, count=count, elapsed_ms=1.0))
"""
    assert _scan(source) == ()


def test_checker_requires_one_unconditional_assignment_before_each_call() -> None:
    before = """
from lib.retrieval_redaction import RetrievalLogReason, retrieval_log_fields
def record():
    count = 0
    logger.info("obs.rag.search", **retrieval_log_fields(reason=RetrievalLogReason.COMPLETED, count=count, elapsed_ms=1.0))
    count = unknown
"""
    branch = before.replace("count = 0", "count = 0 if enabled else 1")
    ambiguous = before.replace(
        "count = 0", "count = 0\n    if enabled:\n        count = 1"
    )
    assert _scan(before) == ()
    assert len(_scan(branch)) == 1
    assert len(_scan(ambiguous)) == 1


# fmt: off
@pytest.mark.parametrize("statement", ("enabled and (count := 0)", "None if enabled else (count := 0)", "pending = (count := 0 for _ in ())", "count = 0; count += ord(prompt[0])", "assert (count := 0) == 0"))
# fmt: on
def test_checker_rejects_unsafe_count_bindings(statement: str) -> None:
    source = _structured_count_source(statement).replace("record(prompt)", "record(prompt, count)")
    assert len(_scan(source)) == 1
