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
    reason=RetrievalLogReason.COMPLETED,
    count=len(rows),
    elapsed_ms=2.5,
)
"""
    assert _scan(source) == ()


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
