#!/usr/bin/env python3
"""Ratchet logger calls toward the ``obs.<domain>.<event>`` convention."""

from __future__ import annotations

import ast
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

_OBS_RE = re.compile(r"^obs\.[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
_LOG_LEVELS = frozenset({"debug", "info", "warning", "error", "critical", "exception"})
_SKIP_PARTS = frozenset({"migrations", "__pycache__", "node_modules", "tests"})

# Exact legacy fingerprints and counts. Any new occurrence fails; resolving an
# occurrence also fails until this ratchet is reduced in the same change.
_BASELINE_VIOLATIONS: Counter[tuple[str, str, str]] = Counter(
    {
        (
            "aquillm/apps/chat/consumers/chat.py",
            "warning",
            "Failed to queue conversation indexing task for convo %s: %s",
        ): 1,
        (
            "aquillm/apps/chat/services/conversation_indexing.py",
            "info",
            "Indexed conversation %s into %d chunks",
        ): 1,
        (
            "aquillm/apps/chat/services/conversation_indexing.py",
            "warning",
            "index_conversation: no conversation %s",
        ): 1,
        (
            "aquillm/apps/chat/services/conversation_indexing.py",
            "warning",
            "Batch embedding failed for conversation %s: %s",
        ): 1,
        (
            "aquillm/apps/chat/services/conversation_indexing.py",
            "warning",
            "Per-window embedding failed for conversation %s window %s: %s",
        ): 1,
        (
            "aquillm/apps/chat/services/conversation_search.py",
            "error",
            "Error during conversation chunk search: %s",
        ): 1,
        (
            "aquillm/apps/chat/services/conversation_search.py",
            "warning",
            "Conversation vector embed failed; trigram-only retrieval. Error: %s",
        ): 1,
        (
            "aquillm/apps/chat/services/rag_metrics.py",
            "info",
            "rag_direct_turn",
        ): 1,
        (
            "aquillm/apps/chat/services/rag_pipeline.py",
            "info",
            "direct_rag_turn_handled retrieved=%d retained=%d status=%s",
        ): 1,
        (
            "aquillm/apps/chat/services/rag_pipeline.py",
            "exception",
            "direct_rag_turn_failed; falling back to tool loop",
        ): 1,
        (
            "aquillm/apps/documents/views/api.py",
            "warning",
            "citation_narrow_empty",
        ): 1,
    }
)


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
    if (
        node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ):
        return node.args[0].value
    return None


def _invalid_calls(path: Path) -> list[tuple[int, str, str]]:
    violations: list[tuple[int, str, str]] = []
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        lines = source.splitlines()
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return violations

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        level = _is_logger_call(node)
        if level is None:
            continue
        # Skip calls annotated with "# ignore" on any line they span.
        end = node.end_lineno or node.lineno
        if any(
            "# ignore" in lines[i] for i in range(node.lineno - 1, min(end, len(lines)))
        ):
            continue
        event = _first_arg_event(node)
        if event is None:
            violations.append((node.lineno, level, "<nonliteral>"))
        elif not _OBS_RE.match(event):
            violations.append((node.lineno, level, event))
    return violations


def _find_ratchet_violations(
    repo: Path,
    baseline: Counter[tuple[str, str, str]],
) -> list[tuple[tuple[str, str, str], int, str]]:
    current: Counter[tuple[str, str, str]] = Counter()
    aquillm = repo / "aquillm"

    for path in sorted(aquillm.rglob("*.py")):
        if not path.is_file() or any(part in _SKIP_PARTS for part in path.parts):
            continue
        rel = path.relative_to(repo).as_posix()
        current.update(
            (rel, level, event) for _line, level, event in _invalid_calls(path)
        )

    violations = [
        (key, count, "new logging violation")
        for key, count in (current - baseline).items()
    ]
    violations.extend(
        (key, count, "remove resolved logging baseline")
        for key, count in (baseline - current).items()
    )
    return sorted(violations)


def main() -> int:
    violations = _find_ratchet_violations(REPO, _BASELINE_VIOLATIONS)

    if violations:
        print("Logging convention ratchet violations:", file=sys.stderr)
        for (rel, level, event), count, reason in violations:
            print(
                f"{rel}: logger.{level}({event!r}) x{count}: {reason}",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
