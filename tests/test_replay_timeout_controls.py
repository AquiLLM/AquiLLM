"""Comparison reports distinguish the actual bounded development profile."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from apps.chat.evals.evidence_effective_config import additional_controls


def test_retrieval_tools_report_configured_allowance_without_an_active_turn(
    monkeypatch,
):
    monkeypatch.setenv("TOOL_CALL_TIMEOUT_SECONDS", "30")
    tools = additional_controls()["tools"]
    assert tools["retrieval_timeout_seconds"] == 30
    assert tools["timeout_seconds"] == 10


def test_replay_reports_loaded_graph_limits_and_configured_tool_limit(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "timeout_replay_cli",
        Path(__file__).parents[1] / "scripts/replay_retrieval.py",
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setenv("TOOL_CALL_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("KG_GRAPH_OVERALL_TIMEOUT_MS", "300")
    settings = SimpleNamespace(
        RAG_CACHE_ENABLED=False,
        KG_GRAPH_OVERALL_TIMEOUT_MS=20000,
        KG_GRAPH_DIRECT_TIMEOUT_MS=15000,
        KG_GRAPH_EXTENDED_TIMEOUT_MS=18000,
    )
    report = cli.effective_config(
        settings,
        SimpleNamespace(vector_top_k=15, trigram_top_k=15),
    )
    assert report["retrieval_tool_timeout_seconds"] == 30
    assert report["graph_timeouts_ms"] == {
        "overall": 20000,
        "direct": 15000,
        "extended": 18000,
    }
