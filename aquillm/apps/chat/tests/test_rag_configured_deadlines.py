"""Opt-in development time allowances retain finite parent and work limits."""

from types import SimpleNamespace

import pytest

from apps.chat.services import rag_config, rag_turn
from lib.knowledge_graph.retrieval_config import (
    HybridRetrievalConfigError,
    load_hybrid_retrieval_settings,
)
from lib.llm.turn_context import TurnContext, bind_turn, current_turn, tool_timeout
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


@pytest.mark.parametrize(
    "configured,expected",
    [
        (None, 15000),
        ("30000", 30000),
        ("999999", 30000),
        ("bad", 15000),
        ("-1", 3000),
    ],
)
def test_configured_retrieval_time_is_bounded(monkeypatch, configured, expected):
    monkeypatch.delenv("RAG_RETRIEVAL_TIMEOUT_MS", raising=False)
    if configured is not None:
        monkeypatch.setenv("RAG_RETRIEVAL_TIMEOUT_MS", configured)
    assert rag_config.retrieval_timeout_ms() == expected


def test_longer_turn_still_expires_and_retains_work_caps(monkeypatch):
    monkeypatch.setenv("TOOL_CALL_TIMEOUT_SECONDS", "30")
    now = [0.0]
    budget = TurnBudget(TurnLimits(retrieval_ms=30000), clock=lambda: now[0])
    with bind_turn(TurnContext(budget)):
        assert tool_timeout("vector_search") == 30.0
        assert tool_timeout(retrieval=True) == 30.0
        assert tool_timeout("non_retrieval_tool") == 10.0
        now[0] = 23.0
        assert tool_timeout("vector_search") == 7.0
        assert budget.reserve_pairs(45, phase="final")
        assert not budget.reserve_pairs(1, phase="final")
        now[0] = 30.0
        assert tool_timeout("vector_search") == 0.0
        assert not budget.publish(lambda: pytest.fail("late publication"))
    assert TurnLimits().retrieval_ms == 15000
    assert TurnLimits(retrieval_ms=30000).clamped_to(TurnLimits()).retrieval_ms == 15000
    with pytest.raises(ValueError):
        TurnLimits(retrieval_ms=30001)


@pytest.mark.asyncio
async def test_request_uses_configured_budget_and_nested_request_keeps_parent(
    monkeypatch,
):
    from apps.documents.services.source_loading import SourceRuntime

    monkeypatch.setenv("RAG_RETRIEVAL_TIMEOUT_MS", "30000")
    monkeypatch.setattr(
        rag_turn, "rag_preservation_config", lambda: SimpleNamespace(active=True)
    )
    monkeypatch.setattr(
        rag_turn, "create_runtime", lambda consumer, budget: SourceRuntime(budget, None)
    )
    consumer = SimpleNamespace(
        user=SimpleNamespace(pk=1), col_ref=SimpleNamespace(collections=[])
    )
    async with rag_turn.preservation_turn(consumer, max_func_calls=3):
        budget = current_turn().budget
        assert budget.limits.retrieval_ms == 30000
        monkeypatch.setenv("RAG_RETRIEVAL_TIMEOUT_MS", "999999")
        async with rag_turn.preservation_turn(consumer, max_func_calls=3):
            assert current_turn().budget is budget
    assert not budget.can_publish()


def test_long_graph_allowance_is_explicit_v2_only():
    values = dict(
        KG_GRAPH_OVERALL_TIMEOUT_MS="20000",
        KG_GRAPH_DIRECT_TIMEOUT_MS="20000",
        KG_GRAPH_EXTENDED_TIMEOUT_MS="20000",
    )
    settings = load_hybrid_retrieval_settings(
        {**values, "KG_GRAPH_REMEDIATION_MODE": "v2"}
    )
    assert (
        settings.graph_overall_timeout_ms == settings.graph_direct_timeout_ms == 20000
    )
    assert settings.graph_extended_timeout_ms == 20000
    for mode in ("legacy", "shadow"):
        with pytest.raises(HybridRetrievalConfigError):
            load_hybrid_retrieval_settings(
                {**values, "KG_GRAPH_REMEDIATION_MODE": mode}
            )
    with pytest.raises(HybridRetrievalConfigError):
        load_hybrid_retrieval_settings(
            {
                **values,
                "KG_GRAPH_REMEDIATION_MODE": "v2",
                "KG_GRAPH_OVERALL_TIMEOUT_MS": "20001",
            }
        )
    defaults = load_hybrid_retrieval_settings({})
    assert (
        defaults.graph_overall_timeout_ms,
        defaults.graph_direct_timeout_ms,
        defaults.graph_extended_timeout_ms,
    ) == (300, 125, 125)
