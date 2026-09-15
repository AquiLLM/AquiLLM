from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_file_lengths.py"


def _module():
    spec = importlib.util.spec_from_file_location("check_file_lengths", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_lines(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("line\n" * count, encoding="utf-8")


def test_reviewed_overage_must_match_its_exact_ratchet(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "aquillm" / "large.py"
    baseline = {"aquillm/large.py": 302}

    _write_lines(path, 302)
    assert module._find_violations(tmp_path, baseline) == []

    _write_lines(path, 303)
    assert module._find_violations(tmp_path, baseline) == [
        ("aquillm/large.py", 303, "grew beyond reviewed maximum 302")
    ]

    _write_lines(path, 301)
    assert module._find_violations(tmp_path, baseline) == [
        ("aquillm/large.py", 301, "ratchet reviewed maximum down from 302")
    ]


def test_new_oversize_and_stale_baseline_entries_fail(tmp_path: Path) -> None:
    module = _module()
    _write_lines(tmp_path / "aquillm" / "new.py", 301)
    _write_lines(tmp_path / "aquillm" / "small.py", 300)

    violations = module._find_violations(
        tmp_path,
        {
            "aquillm/missing.py": 401,
            "aquillm/small.py": 301,
        },
    )

    assert violations == [
        ("aquillm/missing.py", 0, "remove missing reviewed path"),
        ("aquillm/new.py", 301, "new file exceeds 300"),
        ("aquillm/small.py", 300, "remove reviewed path now within 300"),
    ]
