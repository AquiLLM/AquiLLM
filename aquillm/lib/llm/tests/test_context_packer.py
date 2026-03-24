"""Deterministic context packing: pinning and section budgets."""
from __future__ import annotations

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
        "content": "Tool vector_search result:\nargs {}\n" + ("bulkdata " * 300),
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
        "content": "Tool t1 result:\nargs {}\n" + ("OLDTOOL " * 400),
    }
    new_tool = {
        "role": "user",
        "content": "Tool t2 result:\nargs {}\nkeep_marker " + ("x" * 80),
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
