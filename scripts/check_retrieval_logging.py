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
_VALUE_ASSIGNMENTS = (ast.AnnAssign, ast.NamedExpr)
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


def _assignments(tree: ast.AST) -> tuple[tuple[str, ast.AST], ...]:
    assignments: list[tuple[str, ast.AST]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, _VALUE_ASSIGNMENTS) and node.value is not None:
            targets, value = (node.target,), node.value
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            targets, value = (node.target,), node.iter
        else:
            continue
        assignments.extend(
            (name.id, value)
            for target in targets
            for name in ast.walk(target)
            if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store)
        )
    return tuple(assignments)


def _logging_functions(tree: ast.AST) -> frozenset[str]:
    names = set(
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "logging"
        for alias in node.names
        if alias.name in _LOG_LEVELS
    )
    aliases = _assignments(tree)
    changed = True
    while changed:
        before = len(names)
        names.update(
            name
            for name, value in aliases
            if (isinstance(value, ast.Attribute) and value.attr in _LOG_LEVELS)
            or (isinstance(value, ast.Name) and value.id in names)
        )
        changed = len(names) != before
    return frozenset(names)


def _redaction_helpers(tree: ast.AST) -> frozenset[str]:
    imported = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "lib.retrieval_redaction"
        for alias in node.names
        if alias.name == "retrieval_log_fields"
    }
    rebound = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    rebound.update(
        node.arg if isinstance(node, ast.arg) else node.name
        for node in ast.walk(tree)
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.arg)
        )
    )
    return frozenset(imported - rebound)


def _log_call(
    node: ast.Call,
    logging_functions: frozenset[str],
) -> str | None:
    function = node.func
    if isinstance(function, ast.Name) and function.id in logging_functions:
        return function.id
    if not isinstance(function, ast.Attribute) or function.attr not in _LOG_LEVELS:
        return None
    return function.attr


def _fixed_reason(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return type(node.value) is str and node.value in _REASONS
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "RetrievalLogReason"
        and node.attr.isupper()
    )


def _expression_is_tainted(node: ast.AST, tainted: frozenset[str]) -> bool:
    if isinstance(node, ast.Name):
        return node.id in tainted or node.id in {"query", "body", "exact_terms"}
    if isinstance(node, ast.Attribute) and node.attr in {"body", "text"}:
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "str" and node.args:
            return True
    return any(
        _expression_is_tainted(child, tainted)
        for child in ast.iter_child_nodes(node)
        if isinstance(child, ast.expr)
    )


def _tainted_names(tree: ast.AST) -> frozenset[str]:
    tainted: set[str] = set()
    assignments = _assignments(tree)
    changed = True
    while changed:
        changed = False
        frozen = frozenset(tainted)
        for name, value in assignments:
            if name not in tainted and _expression_is_tainted(value, frozen):
                tainted.add(name)
                changed = True
    return frozenset(tainted)


def _safe_count(node: ast.AST, tainted: frozenset[str]) -> bool:
    if _expression_is_tainted(node, tainted):
        return False
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


def _safe_elapsed(node: ast.AST, tainted: frozenset[str]) -> bool:
    if _expression_is_tainted(node, tainted):
        return False
    if isinstance(node, ast.Constant):
        return (
            type(node.value) in {int, float}
            and not isinstance(node.value, bool)
            and node.value >= 0
        )
    if isinstance(node, ast.Name):
        return node.id in {"elapsed_ms", "duration_ms"} or node.id.endswith("_ms")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _safe_elapsed(node.operand, tainted)
    if isinstance(node, ast.BinOp):
        return _safe_elapsed(node.left, tainted) and _safe_elapsed(node.right, tainted)
    return False


def _safe_fields(
    keywords: list[ast.keyword],
    tainted: frozenset[str],
    helpers: frozenset[str],
) -> bool:
    if len(keywords) != 1 or keywords[0].arg is not None:
        return False
    value = keywords[0].value
    if not (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id in helpers
    ):
        return False
    if value.args or any(keyword.arg is None for keyword in value.keywords):
        return False
    keywords = value.keywords
    values = {keyword.arg: keyword.value for keyword in keywords}
    return (
        len(values) == len(keywords)
        and values.keys() == _FIELDS
        and _fixed_reason(values["reason"])
        and _safe_count(values["count"], tainted)
        and _safe_elapsed(values["elapsed_ms"], tainted)
    )


def scan_source(*, path: Path, source: str) -> tuple[LoggingViolation, ...]:
    """Return content-free findings for retrieval logger calls in one source."""

    try:
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, ValueError):
        return (LoggingViolation(path, 0, "invalid_python_source"),)
    functions = _logging_functions(tree)
    helpers = _redaction_helpers(tree)
    tainted = _tainted_names(tree)
    findings: list[LoggingViolation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        level = _log_call(node, functions)
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
        elif not _safe_fields(node.keywords, tainted, helpers):
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
