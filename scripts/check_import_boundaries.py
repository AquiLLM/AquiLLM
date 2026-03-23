#!/usr/bin/env python3
"""Static import-boundary checks (complements pytest architecture tests)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AQUILLM = REPO / "aquillm"

_SKIP_DIR = frozenset({"migrations", "tests", "__pycache__", "node_modules"})
_APPS_IMPORT = re.compile(r"^\s*(from\s+apps\.|import\s+apps\.)", re.MULTILINE)
_AQUILLM_MODELS = re.compile(
    r"^\s*from\s+aquillm\.models\s+import\s", re.MULTILINE
)


def _iter_py(root: Path):
    for path in root.rglob("*.py"):
        if not path.is_file():
            continue
        if any(p in _SKIP_DIR for p in path.parts):
            continue
        yield path


def main() -> int:
    errors: list[str] = []

    lib_root = AQUILLM / "lib"
    if lib_root.is_dir():
        for path in _iter_py(lib_root):
            text = path.read_text(encoding="utf-8", errors="replace")
            if _APPS_IMPORT.search(text):
                rel = path.relative_to(REPO).as_posix()
                errors.append(f"{rel}: lib must not import apps.*")

    apps_root = AQUILLM / "apps"
    if apps_root.is_dir():
        for path in _iter_py(apps_root):
            text = path.read_text(encoding="utf-8", errors="replace")
            if _AQUILLM_MODELS.search(text):
                rel = path.relative_to(REPO).as_posix()
                errors.append(f"{rel}: apps runtime must not import aquillm.models (use domain modules)")

    if errors:
        print("Import boundary violations:", file=sys.stderr)
        for line in sorted(errors):
            print(f"  {line}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
