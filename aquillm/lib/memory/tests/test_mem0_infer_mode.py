"""Focused tests for intelligent Mem0 write mode."""

from __future__ import annotations

from typing import Any

from .test_mem0_graph_mode import _reload_mem0_operations


def test_add_mem0_messages_uses_infer_with_graph_enabled(monkeypatch):
    """Intelligent OSS writes should pass conversation turns with infer enabled."""
    ops_module = _reload_mem0_operations(
        monkeypatch,
        MEM0_GRAPH_ENABLED=1,
        MEM0_GRAPH_ADD_ENABLED=1,
        MEM0_GRAPH_FAIL_OPEN=1,
    )

    captured: list[dict[str, Any]] = []

    class FakeMem0:
        def add(self, messages, **kwargs):
            captured.append({"messages": messages, **kwargs})
            return {"results": [{"event": "ADD"}]}

    monkeypatch.setattr(ops_module, "get_mem0_oss", lambda: FakeMem0())

    wrote = ops_module.add_mem0_messages(
        user_id="1",
        messages=[
            {"role": "user", "content": "Remember that we use Memgraph."},
            {"role": "assistant", "content": "I will remember that."},
        ],
        conversation_id=9,
        assistant_message_uuid="abc-123",
    )

    assert wrote is True
    assert captured == [
        {
            "messages": [
                {"role": "user", "content": "Remember that we use Memgraph."},
                {"role": "assistant", "content": "I will remember that."},
            ],
            "user_id": "1",
            "metadata": {
                "conversation_id": 9,
                "assistant_message_uuid": "abc-123",
                "source": "aquillm",
                "memory_type": "episodic",
            },
            "infer": True,
            "enable_graph": True,
        }
    ]
