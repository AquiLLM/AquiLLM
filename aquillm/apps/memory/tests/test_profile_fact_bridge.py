"""Regression tests for promoting durable facts into profile memory."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4
import pytest

from aquillm import memory as memory_module
from aquillm import tasks as tasks_module


@pytest.fixture(autouse=True)
def mem0_only_runtime(monkeypatch):
    # These unit tests supply fake users and exercise the remote/profile path.
    # Real local dual-write persistence is covered by provider-completion tests.
    monkeypatch.setattr(memory_module, 'MEM0_DUAL_WRITE_LOCAL', False)


@pytest.fixture
def queued_promotions(monkeypatch):
    queued = []
    monkeypatch.setattr(tasks_module.promote_profile_facts_task, 'delay', lambda **kwargs: queued.append(kwargs))
    return queued


def run_queued_promotion(monkeypatch, user, queued):
    assert len(queued) == 1
    assert queued[0]['user_id'] == user.id
    monkeypatch.setattr(memory_module.User.objects, 'filter', lambda **kwargs: SimpleNamespace(first=lambda: user))
    tasks_module.promote_profile_facts_task.run(**queued[0])


class _FakeMessages:
    def __init__(self, rows):
        self._rows = rows

    def order_by(self, *_args, **_kwargs):
        return self

    def values(self, *_args, **_kwargs):
        return list(self._rows)


def test_create_episodic_memories_promotes_durable_facts_via_background_task(monkeypatch, queued_promotions):
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
    monkeypatch.setattr(memory_module, "add_mem0_memory_with_client", lambda **_kwargs: True)
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
    assert promoted == []
    run_queued_promotion(monkeypatch, db_convo.owner, queued_promotions)

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


def test_create_episodic_memories_keeps_fact_promotion_with_intelligent_mem0_write(monkeypatch, queued_promotions):
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
    captured_write: list[dict[str, object]] = []

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
    monkeypatch.setattr(
        memory_module,
        "add_mem0_memory_with_client",
        lambda **kwargs: captured_write.append(kwargs) or True,
    )
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
    assert promoted == []
    run_queued_promotion(monkeypatch, db_convo.owner, queued_promotions)

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
    assert captured_write == [
        {
            "user_id": "1",
            "user_content": "Remember that we use Qdrant and Memgraph for memory and I prefer concise updates.",
            "assistant_content": "I will remember your stack and response style.",
            "conversation_id": 99,
            "assistant_message_uuid": str(assistant_uuid),
            "strict": True,
        }
    ]


def test_create_episodic_memories_still_writes_intelligent_mem0_when_no_facts_promote(monkeypatch, queued_promotions):
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
                    "content": "Can you summarize the last thing we discussed?",
                    "message_uuid": uuid4(),
                },
                {
                    "sequence_number": 1,
                    "role": "assistant",
                    "content": "Here is a summary of the last thing we discussed.",
                    "message_uuid": assistant_uuid,
                },
            ]
        ),
    )

    captured_write: list[dict[str, object]] = []

    monkeypatch.setattr(memory_module, "use_mem0", lambda: True)
    monkeypatch.setattr(memory_module, "extract_stable_facts", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(memory_module, "has_remember_intent", lambda _text: False)
    monkeypatch.setattr(memory_module, "heuristic_facts_from_turn", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        memory_module,
        "add_mem0_memory_with_client",
        lambda **kwargs: captured_write.append(kwargs) or True,
    )
    monkeypatch.setattr(
        memory_module.EpisodicMemory.objects,
        "filter",
        lambda **_kwargs: SimpleNamespace(exists=lambda: False),
    )
    monkeypatch.setattr(
        memory_module.UserMemoryFact.objects,
        "get_or_create",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("no facts should be promoted")),
    )

    memory_module.create_episodic_memories_for_conversation(db_convo)
    run_queued_promotion(monkeypatch, db_convo.owner, queued_promotions)

    assert captured_write == [
        {
            "user_id": "1",
            "user_content": "Can you summarize the last thing we discussed?",
            "assistant_content": "Here is a summary of the last thing we discussed.",
            "conversation_id": 99,
            "assistant_message_uuid": str(assistant_uuid),
            "strict": True,
        }
    ]


def test_create_episodic_memories_defers_profile_fact_extraction_to_background_task(monkeypatch):
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
                    "content": "Remember that we use Qdrant and Memgraph for memory.",
                    "message_uuid": uuid4(),
                },
                {
                    "sequence_number": 1,
                    "role": "assistant",
                    "content": "I will remember your memory stack.",
                    "message_uuid": assistant_uuid,
                },
            ]
        ),
    )

    queued_promotions: list[dict[str, object]] = []
    captured_write: list[dict[str, object]] = []

    monkeypatch.setattr(memory_module, "use_mem0", lambda: True)
    monkeypatch.setattr(
        memory_module,
        "extract_stable_facts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("profile fact extraction should be deferred")
        ),
    )
    monkeypatch.setattr(
        memory_module,
        "_enqueue_profile_fact_promotion",
        lambda **kwargs: queued_promotions.append(kwargs),
        raising=False,
    )
    monkeypatch.setattr(
        memory_module,
        "add_mem0_memory_with_client",
        lambda **kwargs: captured_write.append(kwargs) or True,
    )
    monkeypatch.setattr(
        memory_module.EpisodicMemory.objects,
        "filter",
        lambda **_kwargs: SimpleNamespace(exists=lambda: False),
    )

    memory_module.create_episodic_memories_for_conversation(db_convo)

    assert queued_promotions == [
        {
            "user_id": 1,
            "user_content": "Remember that we use Qdrant and Memgraph for memory.",
            "assistant_content": "I will remember your memory stack.",
        }
    ]
    assert captured_write == [
        {
            "user_id": "1",
            "user_content": "Remember that we use Qdrant and Memgraph for memory.",
            "assistant_content": "I will remember your memory stack.",
            "conversation_id": 99,
            "assistant_message_uuid": str(assistant_uuid),
            "strict": True,
        }
    ]


def test_promote_profile_facts_task_extracts_and_persists_durable_facts(monkeypatch):
    captured_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        memory_module,
        "promote_profile_facts_for_turn",
        lambda **kwargs: captured_calls.append(kwargs),
        raising=False,
    )

    tasks_module.promote_profile_facts_task(
        user_id=7,
        user_content="Remember that I prefer concise updates.",
        assistant_content="I will keep responses concise.",
    )

    assert captured_calls == [
        {
            "user_id": 7,
            "user_content": "Remember that I prefer concise updates.",
            "assistant_content": "I will keep responses concise.",
        }
    ]
