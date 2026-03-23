"""Guardrails: domain apps must not depend on the aquillm.models compatibility barrel."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_AQUILLM_ROOT = Path(__file__).resolve().parents[2]
_APPS_ROOT = _AQUILLM_ROOT / "apps"
_LIB_ROOT = _AQUILLM_ROOT / "lib"

_SKIP_DIR_NAMES = frozenset({"migrations", "tests", "__pycache__"})
_DIRECT_IMPORT = re.compile(
    r"^\s*from\s+aquillm\.models\s+import\s",
    re.MULTILINE,
)


def _iter_runtime_py_files(root: Path):
    if not root.is_dir():
        return
    for path in root.rglob("*.py"):
        if any(part in _SKIP_DIR_NAMES for part in path.parts):
            continue
        yield path


@pytest.mark.parametrize(
    "base",
    [_APPS_ROOT, _LIB_ROOT],
    ids=["apps", "lib"],
)
def test_no_direct_aquillm_models_imports_in_runtime_modules(base: Path):
    offenders: list[str] = []
    for path in _iter_runtime_py_files(base):
        text = path.read_text(encoding="utf-8")
        if _DIRECT_IMPORT.search(text):
            offenders.append(str(path.relative_to(_AQUILLM_ROOT)))
    assert not offenders, "from aquillm.models import found in:\n" + "\n".join(sorted(offenders))
