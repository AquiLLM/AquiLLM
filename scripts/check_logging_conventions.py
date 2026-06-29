#!/usr/bin/env python3
"""Fail if any logger.<level>() call uses a first argument that does not match obs.<domain>.<event>."""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AQUILLM = REPO / "aquillm"

_OBS_RE = re.compile(r"^obs\.[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
_LOG_LEVELS = frozenset({"debug", "info", "warning", "error", "critical", "exception"})
_SKIP_PARTS = frozenset({"migrations", "__pycache__", "node_modules", "tests"})

# Paths (relative to repo root) that are exempt from the convention.
_ALLOWLIST: frozenset[str] = frozenset()


def _is_logger_call(node: ast.Call) -> str | None:
    """Return the log level name if *node* is ``logger.<level>(...)``, else None."""
    if (
        isinstance(node.func, ast.Attribute)
        and node.func.attr in _LOG_LEVELS
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "logger"
    ):
        return node.func.attr
    return None


def _first_arg_event(node: ast.Call) -> str | None:
    """Return the first positional string-literal argument, or None."""
    if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
        return node.args[0].value
    return None


def _check_file(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        lines = source.splitlines()
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return errors

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        level = _is_logger_call(node)
        if level is None:
            continue
        # Skip calls annotated with "# ignore" on any line they span.
        end = node.end_lineno or node.lineno
        if any("# ignore" in lines[i] for i in range(node.lineno - 1, min(end, len(lines)))):
            continue
        event = _first_arg_event(node)
        if event is None:
            errors.append(
                f"  line {node.lineno}: logger.{level}() first arg is not a string literal"
            )
        elif not _OBS_RE.match(event):
            errors.append(
                f"  line {node.lineno}: logger.{level}(\"{event}\", ...) does not match obs.<domain>.<event>"
            )
    return errors


def main() -> int:
    all_errors: list[tuple[str, list[str]]] = []

    for path in sorted(AQUILLM.rglob("*.py")):
        if not path.is_file():
            continue
        if any(p in _SKIP_PARTS for p in path.parts):
            continue
        rel = path.relative_to(REPO).as_posix()
        if rel in _ALLOWLIST:
            continue
        errors = _check_file(path)
        if errors:
            all_errors.append((rel, errors))

    if all_errors:
        print("Logging convention violations (expected obs.<domain>.<event>):", file=sys.stderr)
        for rel, errors in all_errors:
            print(f"{rel}:", file=sys.stderr)
            for err in errors:
                print(err, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
