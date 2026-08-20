#!/usr/bin/env python3
"""Fail closed on payload-bearing log calls in the retrieval feature lane."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import NamedTuple

REPO = Path(__file__).resolve().parents[1]
LANE_PATHS = (
    "aquillm/lib/knowledge_graph/query_extractor/client.py",
    "aquillm/lib/knowledge_graph/query_extractor/service.py",
    "aquillm/apps/knowledge_graph/retrieval/direct_seed_repository.py",
    "aquillm/apps/knowledge_graph/retrieval/direct_seed_resolution.py",
    "aquillm/apps/knowledge_graph/retrieval/query_embedding.py",
)

_EVENT = re.compile(r"^obs\.[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
_LOG_LEVELS = frozenset(
    {"debug", "info", "warning", "error", "critical", "exception", "log"}
)
_FIELDS = frozenset({"reason", "count", "elapsed_ms"})
_REASONS = frozenset(
    {
        "completed",
        "invalid_request",
        "authentication_failed",
        "payload_too_large",
        "upstream_unavailable",
        "provenance_mismatch",
        "mixed_ontology",
        "embedding_unavailable",
        "no_seeds",
        "ambiguous",
        "internal_failure",
    }
)


class LoggingViolation(NamedTuple):
    path: Path
    line: int
    reason: str


def _logger_names(tree: ast.AST) -> frozenset[str]:
    names = {"logger", "_logger"}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Attribute):
            continue
        if value.func.attr not in {"getLogger", "get_logger"}:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        names.update(target.id for target in targets if isinstance(target, ast.Name))
    return frozenset(names)


def _log_call(node: ast.Call, logger_names: frozenset[str]) -> str | None:
    function = node.func
    if not isinstance(function, ast.Attribute) or function.attr not in _LOG_LEVELS:
        return None
    receiver = function.value
    if isinstance(receiver, ast.Name) and (
        receiver.id in logger_names or receiver.id.casefold().endswith("logger")
    ):
        return function.attr
    if isinstance(receiver, ast.Attribute) and receiver.attr.casefold().endswith(
        "logger"
    ):
        return function.attr
    return None


def _fixed_reason(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return type(node.value) is str and node.value in _REASONS
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "RetrievalLogReason"
        and node.attr.isupper()
    )


def _safe_count(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return type(node.value) is int and node.value >= 0
    if isinstance(node, ast.Name):
        return node.id == "count" or node.id.endswith("_count")
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "len"
        and len(node.args) == 1
        and not node.keywords
    )


def _safe_elapsed(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return (
            type(node.value) in {int, float}
            and not isinstance(node.value, bool)
            and node.value >= 0
        )
    if isinstance(node, ast.Name):
        return node.id in {"elapsed_ms", "duration_ms"} or node.id.endswith("_ms")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _safe_elapsed(node.operand)
    if isinstance(node, ast.BinOp):
        return _safe_elapsed(node.left) and _safe_elapsed(node.right)
    return False


def _safe_fields(keywords: list[ast.keyword]) -> bool:
    if len(keywords) == 1 and keywords[0].arg is None:
        value = keywords[0].value
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "retrieval_log_fields"
        ):
            return False
        if value.args or any(keyword.arg is None for keyword in value.keywords):
            return False
        keywords = value.keywords
    elif any(keyword.arg is None for keyword in keywords):
        return False
    values = {keyword.arg: keyword.value for keyword in keywords}
    return (
        len(values) == len(keywords)
        and values.keys() == _FIELDS
        and _fixed_reason(values["reason"])
        and _safe_count(values["count"])
        and _safe_elapsed(values["elapsed_ms"])
    )


def scan_source(*, path: Path, source: str) -> tuple[LoggingViolation, ...]:
    """Return content-free findings for retrieval logger calls in one source."""

    try:
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, ValueError):
        return (LoggingViolation(path, 0, "invalid_python_source"),)
    names = _logger_names(tree)
    findings: list[LoggingViolation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        level = _log_call(node, names)
        if level is None:
            continue
        if level == "exception":
            reason = "exception_logging_forbidden"
        elif (
            len(node.args) != 1
            or not isinstance(node.args[0], ast.Constant)
            or type(node.args[0].value) is not str
            or _EVENT.fullmatch(node.args[0].value) is None
        ):
            reason = "dynamic_or_invalid_event"
        elif not _safe_fields(node.keywords):
            reason = "unknown_or_payload_field_shape"
        else:
            continue
        findings.append(LoggingViolation(path, node.lineno, reason))
    return tuple(sorted(findings, key=lambda item: (item.line, item.reason)))


def find_violations(repo: Path = REPO) -> tuple[LoggingViolation, ...]:
    """Scan the exact reviewed lane; missing or unreadable paths are violations."""

    findings: list[LoggingViolation] = []
    for relative in LANE_PATHS:
        path = repo / relative
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            findings.append(LoggingViolation(path, 0, "missing_lane_path"))
            continue
        findings.extend(scan_source(path=path, source=source))
    return tuple(findings)


def _invalid_calls(path: Path) -> tuple[LoggingViolation, ...]:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return (LoggingViolation(path, 0, "missing_lane_path"),)
    return scan_source(path=path, source=source)


_find_violations = find_violations


def main() -> int:
    findings = find_violations()
    if findings:
        print("Retrieval logging violations:", file=sys.stderr)
        for finding in findings:
            relative = finding.path.relative_to(REPO).as_posix()
            print(f"  {relative}:{finding.line}: {finding.reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
