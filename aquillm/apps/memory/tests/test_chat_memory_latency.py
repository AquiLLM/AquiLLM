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
        captured["thread_sensitive"] = thread_sensitive

        async def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(memory_module, "database_sync_to_async", fake_database_sync_to_async)
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

    def fake_format(profile_facts, episodic):
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

    def fake_format(profile_facts, episodic):
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
