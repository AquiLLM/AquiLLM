"""Unit tests for adaptive tool-call budget policy primitives."""
from __future__ import annotations

from lib.llm.providers.tool_budget import (
    ToolBudgetConfig,
    ToolBudgetPolicy,
    ToolCallObservation,
    parse_csv_positive_int_map,
)


def test_parse_csv_positive_int_map_accepts_valid_entries():
    parsed = parse_csv_positive_int_map(
        "vector_search:4, search_single_document:3, whole_document:2",
        setting_name="LLM_TOOL_CALL_LIMITS",
    )
    assert parsed == {
        "vector_search": 4,
        "search_single_document": 3,
        "whole_document": 2,
    }


def test_parse_csv_positive_int_map_ignores_invalid_entries():
    parsed = parse_csv_positive_int_map(
        "vector_search:4,invalid,search_single_document:abc,whole_document:-1,:2",
        setting_name="LLM_TOOL_CALL_LIMITS",
    )
    assert parsed == {"vector_search": 4}


def test_tool_budget_config_uses_per_tool_override_with_default_fallback(monkeypatch):
    monkeypatch.setenv("LLM_MAX_CALLS_PER_TOOL_NAME", "2")
    monkeypatch.setenv("LLM_TOOL_CALL_LIMITS", "vector_search:4")
    cfg = ToolBudgetConfig.from_env(max_func_calls=6)

    assert cfg.resolve_per_tool_limit("vector_search") == 4
    assert cfg.resolve_per_tool_limit("whole_document") == 2


def test_weighted_budget_breaks_when_units_exhausted():
    cfg = ToolBudgetConfig(
        per_tool_limit_default=10,
        per_tool_limits={},
        repeat_signature_break_threshold=99,
        no_progress_break_threshold=99,
        budget_units_per_turn=3,
        tool_cost_weights={"vector_search": 2},
    )
    policy = ToolBudgetPolicy(cfg)

    first = policy.observe_tool_call(
        ToolCallObservation(
            tool_name="vector_search",
            signature='vector_search|{"q":"alpha"}',
            latest_result_dict={"result": [{"id": 1}]},
        )
    )
    second = policy.observe_tool_call(
        ToolCallObservation(
            tool_name="vector_search",
            signature='vector_search|{"q":"beta"}',
            latest_result_dict={"result": [{"id": 2}]},
        )
    )

    assert first.should_continue is True
    assert second.should_continue is False
    assert second.stop_reason == "budget_units_exhausted"


def test_no_progress_breaks_after_threshold():
    cfg = ToolBudgetConfig(
        per_tool_limit_default=10,
        per_tool_limits={},
        repeat_signature_break_threshold=99,
        no_progress_break_threshold=2,
        budget_units_per_turn=None,
        tool_cost_weights={},
    )
    policy = ToolBudgetPolicy(cfg)

    first = policy.observe_tool_call(
        ToolCallObservation(
            tool_name="vector_search",
            signature='vector_search|{"q":"alpha"}',
            latest_result_dict={"exception": "Tool call timed out"},
        )
    )
    second = policy.observe_tool_call(
        ToolCallObservation(
            tool_name="vector_search",
            signature='vector_search|{"q":"beta"}',
            latest_result_dict={"exception": "Tool call timed out"},
        )
    )

    assert first.should_continue is True
    assert second.should_continue is False
    assert second.stop_reason == "no_progress_break"

