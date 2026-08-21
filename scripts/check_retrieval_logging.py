#!/usr/bin/env python3
# ruff: noqa: E501,E701
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
_LOGGER_NAMES = frozenset({"logger", "log", "audit_logger", "logging", "structlog"})
_LOGGER_ATTRIBUTES = frozenset({"logger", "log", "audit", "audit_logger"})
_LOGGER_FACTORIES = frozenset({"getLogger", "get_logger", "LoggerAdapter"})
_FIELDS = frozenset({"reason", "count", "elapsed_ms"})
_VALUE_ASSIGNMENTS = (ast.AnnAssign, ast.NamedExpr)
# fmt: off
_REASONS = frozenset({"completed", "invalid_request", "authentication_failed", "payload_too_large", "upstream_unavailable", "provenance_mismatch", "mixed_ontology", "embedding_unavailable", "no_seeds", "ambiguous", "internal_failure"})
# fmt: on


class LoggingViolation(NamedTuple):
    path: Path
    line: int
    reason: str


# fmt: off
def _bindings(target: ast.AST, value: ast.AST) -> tuple[tuple[str, ast.AST], ...]:
    if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List)) and len(target.elts) == len(value.elts):
        return tuple(binding for target_item, value_item in zip(target.elts, value.elts, strict=True) for binding in _bindings(target_item, value_item))
    return tuple((name.id, value) for name in ast.walk(target) if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store))
# fmt: on


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
        for target in targets:
            assignments.extend(_bindings(target, value))
    return tuple(assignments)


# fmt: off
def _logger_expression(node: ast.AST, names: set[str]) -> bool:
    if isinstance(node, ast.Name):
        return node.id in names
    if isinstance(node, ast.Attribute):
        return node.attr in _LOGGER_ATTRIBUTES and isinstance(node.value, ast.Name) and node.value.id == "self"
    if not isinstance(node, ast.Call): return False
    function = node.func
    return (isinstance(function, ast.Name) and function.id in _LOGGER_FACTORIES) or (isinstance(function, ast.Attribute) and function.attr in _LOGGER_FACTORIES)

def _getattr_level(node: ast.AST, logger_names: set[str] | frozenset[str]) -> str | None:
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr" and len(node.args) >= 2 and _logger_expression(node.args[0], set(logger_names))): return None
    level = node.args[1]
    if isinstance(level, ast.Constant) and type(level.value) is str: return level.value if level.value in _LOG_LEVELS else None
    return "dynamic"

def _logging_function_expression(node: ast.AST, logger_names: set[str], functions: set[str]) -> bool:
    return any((isinstance(candidate, ast.Name) and candidate.id in functions) or (isinstance(candidate, ast.Attribute) and candidate.attr in _LOG_LEVELS and _logger_expression(candidate.value, logger_names)) or _getattr_level(candidate, logger_names) is not None for candidate in ast.walk(node))

def _logging_bindings(tree: ast.AST) -> tuple[frozenset[str], frozenset[str]]:
    functions = set(alias.asname or alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module == "logging" for alias in node.names if alias.name in _LOG_LEVELS)
    logger_names = set(_LOGGER_NAMES)
    logger_names.update(alias.asname or alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names if alias.name in {"logging", "structlog"})
    aliases = _assignments(tree)
    changed = True
    while changed:
        before = len(functions), len(logger_names)
        logger_names.update(name for name, value in aliases if _logger_expression(value, logger_names))
        functions.update(name for name, value in aliases if _logging_function_expression(value, logger_names, functions))
        changed = before != (len(functions), len(logger_names))
    return frozenset(functions), frozenset(logger_names)

def _redaction_helpers(tree: ast.AST) -> frozenset[str]:
    imported = {alias.asname or alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module == "lib.retrieval_redaction" for alias in node.names if alias.name == "retrieval_log_fields"}
    rebound = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
    rebound.update(node.arg if isinstance(node, ast.arg) else node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.arg)))
    return frozenset(imported - rebound)
# fmt: on


def _log_call(
    node: ast.Call,
    logging_functions: frozenset[str],
    logger_names: frozenset[str],
) -> str | None:
    function = node.func
    if isinstance(function, ast.Name) and function.id in logging_functions:
        return function.id
    if (
        isinstance(function, ast.Attribute)
        and function.attr in _LOG_LEVELS
        and _logger_expression(function.value, set(logger_names))
    ):
        return function.attr
    return _getattr_level(function, logger_names)


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


# fmt: off
def _scope(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> ast.AST:
    while not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)): node = parents[node]
    return node
def _binds(node: ast.AST, name: str) -> bool:
    return (isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, (ast.Store, ast.Del))) or (isinstance(node, ast.arg) and node.arg == name) or (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name) or (isinstance(node, ast.alias) and (node.asname or node.name).split(".", 1)[0] == name) or (isinstance(node, ast.ExceptHandler) and node.name == name) or (isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name == name) or (isinstance(node, ast.MatchMapping) and node.rest == name) or (isinstance(node, (ast.Global, ast.Nonlocal)) and name in node.names)
def _ordinary_binding(node: ast.AST, scope: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    ordinary = False
    while node is not scope:
        if isinstance(node, (ast.NamedExpr, ast.AugAssign)): return False
        if isinstance(node, (ast.Assign, ast.AnnAssign)): ordinary = True
        elif isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.Match, ast.BoolOp, ast.IfExp, ast.GeneratorExp, ast.ListComp, ast.SetComp, ast.DictComp, ast.comprehension)): return False
        node = parents[node]
    return ordinary
def _assigned_value(name: str, assignments: tuple[tuple[str, ast.AST], ...], call: ast.Call, parents: dict[ast.AST, ast.AST]) -> ast.AST | None:
    scope = _scope(call, parents)
    bindings = tuple(node for node in ast.walk(scope) if _binds(node, name) and _scope(parents[node], parents) is scope and (node.lineno, node.col_offset) < (call.lineno, call.col_offset))
    values = tuple(value for candidate, value in assignments if candidate == name and _scope(value, parents) is scope and (value.lineno, value.col_offset) < (call.lineno, call.col_offset))
    return values[0] if len(values) == len(bindings) == 1 and _ordinary_binding(bindings[0], scope, parents) else None
def _safe_count(node: ast.AST, tainted: frozenset[str], assignments: tuple[tuple[str, ast.AST], ...], call: ast.Call, parents: dict[ast.AST, ast.AST], seen: frozenset[str] = frozenset()) -> bool:
    if _expression_is_tainted(node, tainted): return False
    if isinstance(node, ast.Constant): return type(node.value) is int and node.value >= 0
    if isinstance(node, ast.Name):
        if node.id in seen: return False
        value = _assigned_value(node.id, assignments, call, parents)
        return value is not None and _safe_count(value, tainted, assignments, call, parents, seen | {node.id})
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd): return _safe_count(node.operand, tainted, assignments, call, parents, seen)
    return False
def _safe_elapsed(node: ast.AST, tainted: frozenset[str]) -> bool:
    if _expression_is_tainted(node, tainted): return False
    if isinstance(node, ast.Constant): return type(node.value) in {int, float} and not isinstance(node.value, bool) and node.value >= 0
    if isinstance(node, ast.Name): return node.id in {"elapsed_ms", "duration_ms"} or node.id.endswith("_ms")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd): return _safe_elapsed(node.operand, tainted)
    if isinstance(node, ast.BinOp): return _safe_elapsed(node.left, tainted) and _safe_elapsed(node.right, tainted)
    return False
# fmt: on


# fmt: off
def _safe_fields(keywords: list[ast.keyword], tainted: frozenset[str], helpers: frozenset[str], assignments: tuple[tuple[str, ast.AST], ...], call: ast.Call, parents: dict[ast.AST, ast.AST]) -> bool:
# fmt: on
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
        and _safe_count(values["count"], tainted, assignments, call, parents)
        and _safe_elapsed(values["elapsed_ms"], tainted)
    )


def scan_source(*, path: Path, source: str) -> tuple[LoggingViolation, ...]:
    try:
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, ValueError):
        return (LoggingViolation(path, 0, "invalid_python_source"),)
    functions, logger_names = _logging_bindings(tree)
    helpers = _redaction_helpers(tree)
    assignments = _assignments(tree)
    tainted = _tainted_names(tree)
    # fmt: off
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    # fmt: on
    findings: list[LoggingViolation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        level = _log_call(node, functions, logger_names)
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
        elif not _safe_fields(
            node.keywords, tainted, helpers, assignments, node, parents
        ):
            reason = "unknown_or_payload_field_shape"
        else:
            continue
        findings.append(LoggingViolation(path, node.lineno, reason))
    return tuple(sorted(findings, key=lambda item: (item.line, item.reason)))


def find_violations(repo: Path = REPO) -> tuple[LoggingViolation, ...]:
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
