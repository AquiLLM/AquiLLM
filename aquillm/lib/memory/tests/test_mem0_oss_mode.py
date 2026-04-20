"""Tests for Mem0 OSS mode configuration and operations."""

import importlib
import sys
import types
from unittest.mock import patch

import pytest


def _reload_memory_client_module(monkeypatch, **env):
    """Helper to reload the mem0 client module with fresh environment."""
    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, str(value))
    
    from lib.memory.mem0 import client as client_module
    
    importlib.reload(client_module)
    return client_module


@pytest.mark.asyncio
async def test_search_mem0_episodic_memories_async_uses_oss_when_available(monkeypatch):
    """Async search delegates to OSS async helper first."""
    monkeypatch.setenv("MEMORY_BACKEND", "mem0")

    from lib.memory import search_mem0_episodic_memories_async, RetrievedEpisodicMemory
    from lib.memory.mem0 import operations as ops_module

    expected = [RetrievedEpisodicMemory(content="remembered", conversation_id=7)]

    async def fake_via_oss_async(*_args, **_kwargs):
        return expected

    monkeypatch.setattr(ops_module, "search_mem0_via_oss_async", fake_via_oss_async)

    actual = await search_mem0_episodic_memories_async(
        user_id="42",
        query="status update",
        top_k=3,
        exclude_conversation_id=None,
    )

    assert actual == expected


def test_search_mem0_uses_oss_when_available(monkeypatch):
    """Test that search uses OSS SDK when available."""
    monkeypatch.setenv("MEMORY_BACKEND", "mem0")
    
    from lib.memory import search_mem0_episodic_memories, RetrievedEpisodicMemory
    from lib.memory.mem0 import operations as ops_module
    
    expected = [RetrievedEpisodicMemory(content="remembered", conversation_id=7)]
    
    monkeypatch.setattr(
        ops_module,
        "search_mem0_via_oss",
        lambda *_args, **_kwargs: expected,
    )
    
    actual = search_mem0_episodic_memories(
        user_id="42",
        query="status update",
        top_k=3,
        exclude_conversation_id=None,
    )
    
    assert actual == expected


def test_add_mem0_raw_facts_uses_oss_sdk(monkeypatch):
    """Test that add_mem0_raw_facts uses OSS SDK when available."""
    monkeypatch.setenv("MEMORY_BACKEND", "mem0")
    
    from lib.memory.mem0 import client as client_module
    
    class FakeMem0:
        def add(self, fact, user_id=None, metadata=None, infer=False):
            return {"results": [{"event": "ADD"}]}
    
    monkeypatch.setattr(client_module, "_MEM0_OSS", FakeMem0())
    monkeypatch.setattr(client_module, "_MEM0_OSS_INIT_ATTEMPTED", True)
    
    from lib.memory import add_mem0_raw_facts
    
    result = add_mem0_raw_facts(
        user_id="42",
        facts=["Please remember that AquiLLM uses Qdrant and Memgraph for memory."],
        conversation_id=9,
        assistant_message_uuid="abc-123",
    )
    
    assert result is True


def test_get_mem0_oss_keeps_vector_store_dims_for_openai_embedder_by_default(monkeypatch):
    """Test that vector store dims are preserved for OpenAI embedder by default."""
    captured = {}

    class FakeMemory:
        @staticmethod
        def from_config(config):
            captured["config"] = config
            return object()

    monkeypatch.setitem(sys.modules, "mem0", types.SimpleNamespace(Memory=FakeMemory))
    
    client_module = _reload_memory_client_module(
        monkeypatch,
        MEMORY_BACKEND="mem0",
        MEM0_EMBED_PROVIDER="openai",
        MEM0_EMBED_DIMS="2048",
        MEM0_EMBED_MODEL="Qwen/Qwen3-VL-Embedding-2B",
        MEM0_EMBED_BASE_URL="http://vllm_embed:8000/v1",
    )

    client_module.get_mem0_oss()

    embed_config = captured["config"]["embedder"]["config"]
    vector_config = captured["config"]["vector_store"]["config"]
    assert embed_config["embedding_dims"] is None
    assert vector_config["embedding_model_dims"] == 2048


def test_get_mem0_oss_allows_openai_embed_dims_override_when_opted_in(monkeypatch):
    """Test that embedder dims can be overridden when explicitly opted in."""
    captured = {}

    class FakeMemory:
        @staticmethod
        def from_config(config):
            captured["config"] = config
            return object()

    monkeypatch.setitem(sys.modules, "mem0", types.SimpleNamespace(Memory=FakeMemory))
    
    client_module = _reload_memory_client_module(
        monkeypatch,
        MEMORY_BACKEND="mem0",
        MEM0_EMBED_PROVIDER="openai",
        MEM0_EMBED_DIMS="2048",
        MEM0_EMBED_ALLOW_DIMENSIONS_OVERRIDE="1",
    )

    client_module.get_mem0_oss()

    embed_config = captured["config"]["embedder"]["config"]
    vector_config = captured["config"]["vector_store"]["config"]
    assert embed_config["embedding_dims"] == 2048
    assert vector_config["embedding_model_dims"] == 2048


def test_clear_mem0_embedding_dims_override_walks_nested_objects():
    """Test that _clear_mem0_embedding_dims_override walks nested objects."""
    from lib.memory.mem0.client import _clear_mem0_embedding_dims_override

    class Config:
        def __init__(self):
            self.embedding_dims = 1536

    class Node:
        def __init__(self):
            self.config = Config()
            self.children = [{"embedding_dims": 1536}]

    root = Node()

    _clear_mem0_embedding_dims_override(root)

    assert root.config.embedding_dims is None
    assert root.children[0]["embedding_dims"] is None
