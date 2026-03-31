"""Regression tests for promoting durable facts into profile memory."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

from aquillm import memory as memory_module


class _FakeMessages:
    def __init__(self, rows):
        self._rows = rows

    def order_by(self, *_args, **_kwargs):
        return self

    def values(self, *_args, **_kwargs):
        return list(self._rows)


def test_create_episodic_memories_promotes_durable_facts_to_profile_memory(monkeypatch):
    assistant_uuid = uuid4()
    db_convo = SimpleNamespace(
        owner_id=1,
        owner=SimpleNamespace(id=1),
        id=99,
        db_messages=_FakeMessages(
            [
                {
                    "sequence_number": 0,
                    "role": "user",
                    "content": "Remember that we use Qdrant and Memgraph for memory and I prefer concise updates.",
                    "message_uuid": uuid4(),
                },
                {
                    "sequence_number": 1,
                    "role": "assistant",
                    "content": "I will remember your stack and response style.",
                    "message_uuid": assistant_uuid,
                },
            ]
        ),
    )

    promoted = []

    monkeypatch.setattr(memory_module, "use_mem0", lambda: True)
    monkeypatch.setattr(
        memory_module,
        "extract_stable_facts",
        lambda *_args, **_kwargs: [
            "We use Qdrant and Memgraph for memory",
            "I prefer concise updates",
        ],
    )
    monkeypatch.setattr(memory_module, "has_remember_intent", lambda _text: False)
    monkeypatch.setattr(memory_module, "heuristic_facts_from_turn", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(memory_module, "add_mem0_raw_facts", lambda **_kwargs: True)
    monkeypatch.setattr(
        memory_module.EpisodicMemory.objects,
        "filter",
        lambda **_kwargs: SimpleNamespace(exists=lambda: False),
    )
    monkeypatch.setattr(
        memory_module.UserMemoryFact.objects,
        "get_or_create",
        lambda **kwargs: promoted.append(kwargs) or (SimpleNamespace(), True),
    )

    memory_module.create_episodic_memories_for_conversation(db_convo)

    assert promoted == [
        {
            "user": db_convo.owner,
            "fact": "We use Qdrant and Memgraph for memory",
            "defaults": {"category": "project"},
        },
        {
            "user": db_convo.owner,
            "fact": "I prefer concise updates",
            "defaults": {"category": "preference"},
        },
    ]
