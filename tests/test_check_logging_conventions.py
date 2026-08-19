from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "check_logging_conventions.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("check_logging_conventions", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_logger_calls(path: Path, calls: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "import logging\nlogger = logging.getLogger(__name__)\n" + "\n".join(calls),
        encoding="utf-8",
    )


def test_logging_baseline_is_an_exact_multiset_ratchet(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "aquillm" / "service.py"
    legacy = ("aquillm/service.py", "warning", "legacy message")
    baseline = Counter({legacy: 1})

    _write_logger_calls(path, ('logger.warning("legacy message")',))
    assert module._find_ratchet_violations(tmp_path, baseline) == []

    _write_logger_calls(
        path,
        (
            'logger.warning("legacy message")',
            'logger.warning("legacy message")',
            'logger.info("obs.search.completed")',
        ),
    )
    assert module._find_ratchet_violations(tmp_path, baseline) == [
        (legacy, 1, "new logging violation")
    ]


def test_fixed_or_moved_logging_baseline_must_be_removed(tmp_path: Path) -> None:
    module = _module()
    legacy = ("aquillm/old.py", "error", "legacy failure")
    baseline = Counter({legacy: 1})
    _write_logger_calls(
        tmp_path / "aquillm" / "old.py",
        ('logger.error("obs.search.failed")',),
    )

    assert module._find_ratchet_violations(tmp_path, baseline) == [
        (legacy, 1, "remove resolved logging baseline")
    ]
