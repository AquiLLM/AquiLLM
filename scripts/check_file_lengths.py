#!/usr/bin/env python3
"""Fail if source files exceed line-count budget (with allowlist for legacy hotspots)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MAX_LINES = 300

# Paths relative to repo root; trim trailing slashes for consistency
_ALLOWLIST: frozenset[str] = frozenset(
    {
        # Backend hotspots (remediation in progress; see architecture remediation plan)
        "aquillm/apps/chat/consumers/chat.py",
        "aquillm/apps/documents/services/chunk_rerank.py",
        "aquillm/apps/collections/views/api.py",
        "aquillm/aquillm/ingestion/figure_extraction/pdf.py",
        "aquillm/aquillm/ingestion/parsers.py",
        "aquillm/aquillm/settings.py",
        "aquillm/aquillm/tasks.py",
        "aquillm/aquillm/zotero_views.py",
        "aquillm/lib/llm/providers/openai.py",
        # Frontend (legacy large components; feature modules still over budget)
        "react/src/components/CollectionsPage.tsx",
        "react/src/features/collections/components/CollectionView.tsx",
        "react/src/features/documents/components/FileSystemViewer.tsx",
        "react/src/features/platform_admin/components/UserManagementModal.tsx",
    }
)


def _line_count(path: Path) -> int:
    return sum(1 for _ in path.open("r", encoding="utf-8", errors="replace"))


def main() -> int:
    roots = [
        (REPO / "aquillm", {".py"}),
        (REPO / "react" / "src", {".ts", ".tsx"}),
    ]
    skip_parts = frozenset({"migrations", "__pycache__", "node_modules", "dist", "build"})
    offenders: list[tuple[str, int]] = []

    for base, suffixes in roots:
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in suffixes:
                continue
            if any(p in skip_parts for p in path.parts):
                continue
            rel = path.relative_to(REPO).as_posix()
            n = _line_count(path)
            if n > MAX_LINES and rel not in _ALLOWLIST:
                offenders.append((rel, n))

    if offenders:
        offenders.sort()
        print(f"Files over {MAX_LINES} lines (not in allowlist):", file=sys.stderr)
        for rel, n in offenders:
            print(f"  {n:5d}  {rel}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
