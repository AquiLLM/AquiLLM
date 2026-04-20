"""Deterministic context packing: pinning and section budgets."""
from __future__ import annotations

import logging

import structlog

from lib.llm.utils.context_packer import ContextPackerConfig, pack_messages_for_budget


def test_context_packer_pins_latest_user_and_tool_chain():
    assistant_tool = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "vector_search", "arguments": "{}"},
            }
        ],
    }
    tool_res = {
        "role": "user",
        "content": "Tool:vector_search\n{}\n" + ("bulkdata " * 300),
    }
    final = {"role": "user", "content": "final question unique"}
    msgs = [
        {"role": "user", "content": "noise " * 500},
        {"role": "assistant", "content": "old assistant"},
        {"role": "user", "content": "earlier user turn"},
        assistant_tool,
        tool_res,
        final,
    ]
    cfg = ContextPackerConfig(
        pin_last_turns=1,
        budget_history_tokens=400,
        budget_tool_evidence_tokens=400,
    )
    out = pack_messages_for_budget(
        "short system",
        msgs,
        context_limit=900,
        max_tokens=128,
        cfg=cfg,
        slack=32,
    )
    packed = out["messages"]
    flat = str(packed)
    assert "final question unique" in flat
    assert "bulkdata" in flat
    assert "noise " not in flat


def test_context_packer_respects_section_budgets():
    old_tool = {
        "role": "user",
        "content": "Tool:t1\n{}\n" + ("OLDTOOL " * 400),
    }
    new_tool = {
        "role": "user",
        "content": "Tool:t2\n{}\nkeep_marker " + ("x" * 80),
    }
    msgs = [
        {"role": "user", "content": "hi"},
        old_tool,
        {"role": "assistant", "content": "between"},
        new_tool,
        {"role": "user", "content": "latest user asks"},
    ]
    cfg = ContextPackerConfig(
        pin_last_turns=1,
        budget_history_tokens=6000,
        budget_tool_evidence_tokens=120,
    )
    out = pack_messages_for_budget(
        "s",
        msgs,
        context_limit=4096,
        max_tokens=256,
        cfg=cfg,
        slack=32,
    )
    packed = out["messages"]
    flat = str(packed)
    assert "keep_marker" in flat
    assert "latest user asks" in flat
    assert "OLDTOOL" not in flat


def test_pruning_stage_order_dedupe_then_compress_then_hard_trim():
    shared = "Tool:vector_search\n{}\nshared header line one\nshared header line two\n"
    filler = (
        "First sentence about nothing. Second sentence also filler. "
        "Third runs on with extra detail that should be trimmed away under pressure."
    )
    msgs = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": filler * 18},
        {"role": "user", "content": shared + ("alpha " * 80)},
        {"role": "user", "content": shared + ("beta " * 80)},
        {"role": "user", "content": "last"},
    ]
    cfg = ContextPackerConfig(
        pin_last_turns=1,
        budget_history_tokens=150,
        budget_tool_evidence_tokens=9000,
    )
    out = pack_messages_for_budget(
        "sys",
        msgs,
        context_limit=650,
        max_tokens=96,
        cfg=cfg,
        slack=24,
    )
    stages = out["stats"]["stages_applied"]
    assert "dedupe" in stages
    if "extractive" in stages and "hard_trim" in stages:
        assert stages.index("extractive") < stages.index("hard_trim")
    if "hard_trim" in stages:
        assert stages.index("dedupe") < stages.index("hard_trim")


def test_context_packer_logs_stats_without_prompt_body(caplog):
    secret = "SECRET_USER_TEXT_DO_NOT_LOG"
    msgs = [{"role": "user", "content": secret}]
    with caplog.at_level(logging.INFO, logger="lib.llm.utils.context_packer"):
        pack_messages_for_budget(
            "system also hidden",
            msgs,
            context_limit=2048,
            max_tokens=256,
            cfg=ContextPackerConfig(),
            slack=32,
        )
    joined = " ".join(r.message for r in caplog.records)
    assert "context_pack stats" in joined
    assert secret not in joined
    assert "system also hidden" not in joined


def test_context_packer_limits_token_estimator_calls_under_pressure(monkeypatch):
    import lib.llm.utils.context_packer as cp

    real_estimate = cp.estimate_prompt_tokens
    call_counter = {"count": 0}

    def wrapped_estimate(messages, encoder):
        call_counter["count"] += 1
        return real_estimate(messages, encoder)

    monkeypatch.setattr(cp, "estimate_prompt_tokens", wrapped_estimate)

    msgs = [{"role": "user", "content": f"turn {i} " + ("data " * 220)} for i in range(18)]
    out = cp.pack_messages_for_budget(
        "short system",
        msgs,
        context_limit=1200,
        max_tokens=128,
        cfg=ContextPackerConfig(
            pin_last_turns=1,
            budget_history_tokens=320,
            budget_tool_evidence_tokens=320,
        ),
        slack=24,
    )
    assert out["stats"].get("fail_open") is not True
    assert call_counter["count"] <= 12


