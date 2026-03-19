"""Guardrails against debug breakpoints and print-based diagnostics in runtime paths."""
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]


def test_no_breakpoint_calls_in_core_pages():
    pages = PROJECT / "apps" / "core" / "views" / "pages.py"
    text = pages.read_text(encoding="utf-8")
    assert "breakpoint()" not in text


def test_no_print_calls_in_auth_adapter():
    adapters = PROJECT / "aquillm" / "adapters.py"
    text = adapters.read_text(encoding="utf-8")
    assert "print(" not in text
