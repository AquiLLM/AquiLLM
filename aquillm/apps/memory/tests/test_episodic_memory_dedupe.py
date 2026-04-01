"""Regression tests for episodic memory dedupe races."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest
from django.db import IntegrityError

from aquillm import memory as memory_module


class _FakeMessages:
    def __init__(self, rows):
        self._rows = rows

    def order_by(self, *_args, **_kwargs):
        return self

    def values(self, *_args, **_kwargs):
        return list(self._rows)


def test_create_episodic_memories_ignores_duplicate_insert_race(monkeypatch):
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
                    "content": "Remember our Memgraph setup.",
                    "message_uuid": uuid4(),
                },
                {
                    "sequence_number": 1,
                    "role": "assistant",
                    "content": "I will keep the Memgraph setup in mind.",
                    "message_uuid": assistant_uuid,
                },
            ]
        ),
    )

    monkeypatch.setattr(memory_module, "use_mem0", lambda: False)

    def fake_filter(**kwargs):
        assert kwargs == {"user": 1, "assistant_message_uuid": assistant_uuid}
        return SimpleNamespace(exists=lambda: False)

    def fake_create(**kwargs):
        raise IntegrityError(
            'duplicate key value violates unique constraint "unique_episodic_per_assistant_msg"'
        )

    monkeypatch.setattr(memory_module.EpisodicMemory.objects, "filter", fake_filter)
    monkeypatch.setattr(memory_module.EpisodicMemory.objects, "create", fake_create)

    memory_module.create_episodic_memories_for_conversation(db_convo)


def test_create_episodic_memories_still_raises_unrelated_integrity_errors(monkeypatch):
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
                    "content": "Remember our Qdrant setup.",
                    "message_uuid": uuid4(),
                },
                {
                    "sequence_number": 1,
                    "role": "assistant",
                    "content": "I will keep the Qdrant setup in mind.",
                    "message_uuid": assistant_uuid,
                },
            ]
        ),
    )

    monkeypatch.setattr(memory_module, "use_mem0", lambda: False)
    monkeypatch.setattr(
        memory_module.EpisodicMemory.objects,
        "filter",
        lambda **_kwargs: SimpleNamespace(exists=lambda: False),
    )
    monkeypatch.setattr(
        memory_module.EpisodicMemory.objects,
        "create",
        lambda **_kwargs: (_ for _ in ()).throw(IntegrityError("some other integrity problem")),
    )

    with pytest.raises(IntegrityError, match="some other integrity problem"):
        memory_module.create_episodic_memories_for_conversation(db_convo)
