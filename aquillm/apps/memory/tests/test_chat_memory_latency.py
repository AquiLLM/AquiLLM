"""Chat-time memory augmentation latency tests."""

from __future__ import annotations

import asyncio

import pytest

from aquillm import memory as memory_module


class _FakeConversation:
    messages: list[object] = []
    system: str | None = None


class _FakeUser:
    id = 42


@pytest.mark.asyncio
async def test_local_episodic_lookup_uses_interruptible_executor(monkeypatch):
    captured: dict[str, object] = {}

    def fake_database_sync_to_async(func, *, thread_sensitive=True):
        captured.setdefault("thread_sensitive", thread_sensitive)

        async def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(memory_module, "database_sync_to_async", fake_database_sync_to_async)
    from apps.memory.services import profile_facts
    monkeypatch.setattr(profile_facts, 'filter_reconciled_memories', lambda user, items: items)
    monkeypatch.setattr(memory_module, "use_mem0", lambda: False)
    monkeypatch.setattr(
        memory_module,
        "_get_episodic_memories_pgvector",
        lambda *_args, **_kwargs: ["memory"],
    )

    result = await memory_module.get_episodic_memories_async(
        _FakeUser(),
        "compare these papers",
    )

    assert result == ["memory"]
    assert captured["thread_sensitive"] is False


@pytest.mark.asyncio
async def test_async_memory_augmentation_times_out_slow_episodic_lookup(monkeypatch):
    """A slow episodic backend should not block chat generation startup."""
    profile_fact = object()
    captured: dict[str, object] = {}

    def fake_database_sync_to_async(func, **_kwargs):
        async def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    async def slow_episodic_lookup(*args, **kwargs):
        await asyncio.sleep(10)
        return ["too late"]

    def fake_format(profile_facts, episodic, *, max_chars):
        captured["profile_facts"] = profile_facts
        captured["episodic"] = episodic
        return "\n\n<memory>"

    monkeypatch.setenv("MEMORY_RETRIEVAL_TIMEOUT_SECONDS", "0.01")
    monkeypatch.setattr(memory_module, "database_sync_to_async", fake_database_sync_to_async)
    monkeypatch.setattr(memory_module, "get_last_user_message_text", lambda convo: "tell me about this")
    monkeypatch.setattr(memory_module, "get_user_profile_facts", lambda user: [profile_fact])
    monkeypatch.setattr(memory_module, "get_episodic_memories_async", slow_episodic_lookup)
    monkeypatch.setattr(memory_module, "format_memories_for_system", fake_format)

    convo = _FakeConversation()

    await asyncio.wait_for(
        memory_module.augment_conversation_with_memory_async(convo, _FakeUser(), "base"),
        timeout=0.5,
    )

    assert captured == {"profile_facts": [profile_fact], "episodic": []}
    assert convo.system == "base\n\n<memory>"


@pytest.mark.asyncio
async def test_async_memory_augmentation_can_skip_cross_chat_episodic_lookup(monkeypatch):
    """Selected-collection RAG should not wait for unrelated cross-chat retrieval."""
    profile_fact = object()
    captured: dict[str, object] = {}

    def fake_database_sync_to_async(func, **_kwargs):
        async def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    async def unexpected_episodic_lookup(*_args, **_kwargs):
        raise AssertionError("episodic retrieval should be skipped")

    def fake_format(profile_facts, episodic, *, max_chars):
        captured["profile_facts"] = profile_facts
        captured["episodic"] = episodic
        return "\n\n<memory>"

    monkeypatch.setattr(memory_module, "database_sync_to_async", fake_database_sync_to_async)
    monkeypatch.setattr(memory_module, "get_last_user_message_text", lambda convo: "compare these papers")
    monkeypatch.setattr(memory_module, "get_user_profile_facts", lambda user: [profile_fact])
    monkeypatch.setattr(memory_module, "get_episodic_memories_async", unexpected_episodic_lookup)
    monkeypatch.setattr(memory_module, "format_memories_for_system", fake_format)

    convo = _FakeConversation()

    await memory_module.augment_conversation_with_memory_async(
        convo,
        _FakeUser(),
        "base",
        include_episodic=False,
    )

    assert captured == {"profile_facts": [profile_fact], "episodic": []}
    assert convo.system == "base\n\n<memory>"


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
async def test_revocation_during_episodic_lookup_cannot_inject_a_stale_profile(monkeypatch):
    from channels.db import database_sync_to_async
    from django.contrib.auth import get_user_model
    from apps.memory.models import UserMemoryFact
    from apps.memory.services.profile_facts import revoke_profile_fact
    from lib.llm import Conversation, UserMessage
    user = await get_user_model().objects.acreate(username='augmentation-revocation')
    fact = await UserMemoryFact.objects.acreate(user=user, origin='manual', fact='I prefer telescope A.')
    async def revoke_while_fetching(*args, **kwargs):
        await asyncio.sleep(0.05)
        await database_sync_to_async(revoke_profile_fact)(user=user, fact_id=fact.pk, expected_revision=1)
        return []
    monkeypatch.setattr(memory_module, '_get_episodic_memories_with_latency_budget', revoke_while_fetching)
    monkeypatch.setattr(memory_module, '_remaining_memory_chars', lambda *args: 8000)
    convo = Conversation(system='base', messages=[UserMessage(content='hello')])
    await memory_module.augment_conversation_with_memory_async(convo, user, 'base')
    assert 'telescope A' not in convo.system


@pytest.mark.asyncio
async def test_augmentation_clamps_memory_without_trimming_existing_context(monkeypatch):
    from lib.llm import Conversation, UserMessage
    from types import SimpleNamespace
    from apps.chat.consumers import utils
    monkeypatch.setattr(utils, 'CHAT_MAX_TOKENS', 256)
    monkeypatch.setenv('OPENAI_CONTEXT_LIMIT', '1024')
    monkeypatch.setattr(memory_module, 'get_user_profile_facts', lambda user: [SimpleNamespace(fact='I prefer short replies.')])
    async def empty(*args, **kwargs):
        return []
    monkeypatch.setattr(memory_module, '_get_episodic_memories_with_latency_budget', empty)
    convo = Conversation(system='base', messages=[UserMessage(content='hello')])
    base = 'z' * 950
    await memory_module.augment_conversation_with_memory_async(convo, _FakeUser(), base)
    assert convo.system == base
    assert convo.messages[0].content == 'hello'


@pytest.mark.django_db(transaction=True)
def test_sync_revocation_during_episodic_lookup_cannot_inject_a_stale_profile(monkeypatch):
    from django.contrib.auth import get_user_model
    from apps.memory.models import UserMemoryFact
    from apps.memory.services.profile_facts import revoke_profile_fact
    from lib.llm import Conversation, UserMessage
    user = get_user_model().objects.create(username='sync-augmentation-revocation')
    fact = UserMemoryFact.objects.create(user=user, origin='manual', fact='I prefer telescope A.')
    def revoke_while_fetching(*args, **kwargs):
        revoke_profile_fact(user=user, fact_id=fact.pk, expected_revision=1)
        return []
    monkeypatch.setattr(memory_module, 'get_episodic_memories', revoke_while_fetching)
    monkeypatch.setattr(memory_module, '_remaining_memory_chars', lambda *args: 8000)
    convo = Conversation(system='base', messages=[UserMessage(content='hello')])
    memory_module.augment_conversation_with_memory(convo, user, 'base')
    assert 'telescope A' not in convo.system
