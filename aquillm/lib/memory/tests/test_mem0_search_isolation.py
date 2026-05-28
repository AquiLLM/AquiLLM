"""Regression tests for Mem0 memory ownership/session isolation."""

from __future__ import annotations

from lib.memory.mem0 import operations as ops_module
from lib.memory.mem0.search_parsing import parse_mem0_search_items


def test_parse_mem0_search_items_requires_matching_user_and_excludes_current_session():
    raw_items = [
        {
            "memory": "current user's older memory",
            "metadata": {"user_id": "42", "conversation_id": 7},
        },
        {
            "memory": "another user's memory",
            "metadata": {"user_id": "99", "conversation_id": 8},
        },
        {
            "memory": "legacy memory without owner proof",
            "metadata": {"conversation_id": 9},
        },
        {
            "memory": "top-level owner proof",
            "user_id": 42,
            "metadata": {"conversation_id": 10},
        },
        {
            "memory": "current session memory",
            "metadata": {"user_id": "42", "conversation_id": 11},
        },
    ]

    parsed = parse_mem0_search_items(
        raw_items,
        top_k=5,
        exclude_conversation_id=11,
        user_id="42",
    )

    assert [item.content for item in parsed] == [
        "current user's older memory",
        "top-level owner proof",
    ]
    assert [item.conversation_id for item in parsed] == [7, 10]


def test_mem0_writes_stamp_user_id_in_metadata(monkeypatch):
    captured_metadata: list[dict[str, object]] = []

    class FakeMem0:
        def add(self, _payload, **kwargs):
            captured_metadata.append(kwargs["metadata"])
            return {"results": [{"event": "ADD"}]}

    monkeypatch.setattr(ops_module, "get_mem0_oss", lambda: FakeMem0())

    assert ops_module.add_mem0_messages(
        user_id="42",
        messages=[
            {"role": "user", "content": "Remember my project stack."},
            {"role": "assistant", "content": "I will remember it."},
        ],
        conversation_id=7,
        assistant_message_uuid="assistant-1",
    )
    assert ops_module.add_mem0_raw_facts(
        user_id="42",
        facts=["AquiLLM uses Qdrant and Memgraph for memory."],
        conversation_id=8,
        assistant_message_uuid="assistant-2",
    )

    assert [item["user_id"] for item in captured_metadata] == ["42", "42"]
