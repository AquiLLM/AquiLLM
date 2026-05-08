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
async def test_async_memory_augmentation_times_out_slow_episodic_lookup(monkeypatch):
    """A slow episodic backend should not block chat generation startup."""
    profile_fact = object()
    captured: dict[str, object] = {}

    def fake_database_sync_to_async(func):
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
