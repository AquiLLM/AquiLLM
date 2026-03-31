"""Tests for optional Mem0 graph memory configuration and fail-open behavior."""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any

import pytest


GRAPH_ENV_KEYS = (
    "MEM0_GRAPH_ENABLED",
    "MEM0_GRAPH_PROVIDER",
    "MEM0_GRAPH_URL",
    "MEM0_GRAPH_USERNAME",
    "MEM0_GRAPH_PASSWORD",
    "MEM0_GRAPH_DATABASE",
    "MEM0_GRAPH_CUSTOM_PROMPT",
    "MEM0_GRAPH_THRESHOLD",
    "MEM0_GRAPH_FAIL_OPEN",
    "MEM0_GRAPH_ADD_ENABLED",
    "MEM0_GRAPH_SEARCH_ENABLED",
)


def _apply_env(monkeypatch, **env: Any) -> None:
    for key in GRAPH_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, str(value))


def _reload_memory_config(monkeypatch, **env: Any):
    _apply_env(monkeypatch, **env)
    from lib.memory import config as config_module

    importlib.reload(config_module)
    return config_module


def _reload_mem0_client(monkeypatch, **env: Any):
    _apply_env(monkeypatch, **env)
    from lib.memory.mem0 import client as client_module

    importlib.reload(client_module)
    return client_module


def _reload_mem0_operations(monkeypatch, **env: Any):
    _apply_env(monkeypatch, **env)
    from lib.memory.mem0 import operations as ops_module

    importlib.reload(ops_module)
    return ops_module


def test_graph_env_defaults(monkeypatch):
    """Graph mode should be disabled by default with fail-open enabled."""
    cfg = _reload_memory_config(monkeypatch)

    assert cfg.MEM0_GRAPH_ENABLED is False
    assert cfg.MEM0_GRAPH_PROVIDER == ""
    assert cfg.MEM0_GRAPH_URL == "bolt://memgraph:7687"
    assert cfg.MEM0_GRAPH_USERNAME == "memgraph"
    assert cfg.MEM0_GRAPH_PASSWORD == ""
    assert cfg.MEM0_GRAPH_DATABASE is None
    assert cfg.MEM0_GRAPH_CUSTOM_PROMPT is None
    assert cfg.MEM0_GRAPH_THRESHOLD is None
    assert cfg.MEM0_GRAPH_FAIL_OPEN is True
    assert cfg.MEM0_GRAPH_ADD_ENABLED is True
    assert cfg.MEM0_GRAPH_SEARCH_ENABLED is True


def test_graph_env_flags_parse(monkeypatch):
    """Graph env booleans should parse from 1/0 values."""
    cfg = _reload_memory_config(
        monkeypatch,
        MEM0_GRAPH_ENABLED=1,
        MEM0_GRAPH_FAIL_OPEN=0,
        MEM0_GRAPH_ADD_ENABLED=0,
        MEM0_GRAPH_SEARCH_ENABLED=1,
    )

    assert cfg.MEM0_GRAPH_ENABLED is True
    assert cfg.MEM0_GRAPH_FAIL_OPEN is False
    assert cfg.MEM0_GRAPH_ADD_ENABLED is False
    assert cfg.MEM0_GRAPH_SEARCH_ENABLED is True


def test_build_config_omits_graph_store_when_disabled(monkeypatch):
    """Mem0 OSS config should not include graph_store by default."""
    client_module = _reload_mem0_client(monkeypatch, MEM0_GRAPH_ENABLED=0)

    config, _ = client_module._build_mem0_oss_config_dict()
    assert "graph_store" not in config


def test_build_config_includes_graph_store_when_enabled(monkeypatch):
    """Mem0 OSS config should include graph_store when graph mode is valid."""
    client_module = _reload_mem0_client(
        monkeypatch,
        MEM0_GRAPH_ENABLED=1,
        MEM0_GRAPH_PROVIDER="memgraph",
        MEM0_GRAPH_URL="bolt://memgraph:7687",
        MEM0_GRAPH_USERNAME="memgraph",
        MEM0_GRAPH_PASSWORD="secret",
        MEM0_GRAPH_DATABASE="mem0",
        MEM0_GRAPH_CUSTOM_PROMPT="extract graph relations",
        MEM0_GRAPH_THRESHOLD="0.75",
    )

    config, _ = client_module._build_mem0_oss_config_dict()
    graph_store = config["graph_store"]
    assert graph_store["provider"] == "memgraph"
    assert graph_store["config"]["url"] == "bolt://memgraph:7687"
    assert graph_store["config"]["username"] == "memgraph"
    assert graph_store["config"]["password"] == "secret"
    assert graph_store["config"]["refresh_schema"] is False
    assert graph_store["config"]["database"] == "mem0"
    assert graph_store["custom_prompt"] == "extract graph relations"
    assert graph_store["config"]["threshold"] == 0.75


def test_build_config_invalid_graph_fail_open(monkeypatch):
    """Invalid graph config should fail open and omit graph_store."""
    client_module = _reload_mem0_client(
        monkeypatch,
        MEM0_GRAPH_ENABLED=1,
        MEM0_GRAPH_PROVIDER="memgraph",
        MEM0_GRAPH_URL="",
        MEM0_GRAPH_USERNAME="",
        MEM0_GRAPH_PASSWORD="",
        MEM0_GRAPH_FAIL_OPEN=1,
    )

    config, _ = client_module._build_mem0_oss_config_dict()
    assert "graph_store" not in config


def test_build_config_invalid_graph_strict_raises(monkeypatch):
    """Invalid graph config should raise when fail-open is disabled."""
    client_module = _reload_mem0_client(
        monkeypatch,
        MEM0_GRAPH_ENABLED=1,
        MEM0_GRAPH_PROVIDER="memgraph",
        MEM0_GRAPH_FAIL_OPEN=0,
        MEM0_GRAPH_URL="",
        MEM0_GRAPH_USERNAME="",
        MEM0_GRAPH_PASSWORD="",
    )

    with pytest.raises(ValueError):
        client_module._build_mem0_oss_config_dict()


def test_register_memgraph_compat_provider(monkeypatch):
    """Mem0 memgraph provider should be redirected to the local compatibility shim."""
    client_module = _reload_mem0_client(monkeypatch)
    fake_factory_module = types.ModuleType("mem0.utils.factory")

    class FakeGraphStoreFactory:
        provider_to_class = {"memgraph": "mem0.memory.memgraph_memory.MemoryGraph"}

    fake_factory_module.GraphStoreFactory = FakeGraphStoreFactory
    monkeypatch.setitem(sys.modules, "mem0.utils.factory", fake_factory_module)

    client_module._register_memgraph_compat_provider()

    assert (
        FakeGraphStoreFactory.provider_to_class["memgraph"]
        == "lib.memory.mem0.memgraph_compat.CompatibleMemgraphMemoryGraph"
    )


def test_search_uses_enable_graph_when_enabled(monkeypatch):
    """OSS search should pass enable_graph=True when graph search is enabled."""
    ops_module = _reload_mem0_operations(
        monkeypatch,
        MEM0_GRAPH_ENABLED=1,
        MEM0_GRAPH_SEARCH_ENABLED=1,
        MEM0_GRAPH_FAIL_OPEN=1,
    )

    captured: list[dict[str, Any]] = []

    class FakeMem0:
        def search(self, *_args, **kwargs):
            captured.append(kwargs)
            return {"results": [{"memory": "graph answer"}]}

    monkeypatch.setattr(ops_module, "get_mem0_oss", lambda: FakeMem0())

    results = ops_module.search_mem0_via_oss(
        user_id="1",
        query="who do i collaborate with?",
        top_k=3,
        exclude_conversation_id=None,
    )

    assert results[0].content == "graph answer"
    assert captured[0]["enable_graph"] is True


def test_search_graph_failure_retries_vector_only(monkeypatch):
    """When graph search fails, fail-open should retry with enable_graph=False."""
    ops_module = _reload_mem0_operations(
        monkeypatch,
        MEM0_GRAPH_ENABLED=1,
        MEM0_GRAPH_SEARCH_ENABLED=1,
        MEM0_GRAPH_FAIL_OPEN=1,
    )

    seen_enable_graph: list[Any] = []

    class FakeMem0:
        def search(self, *_args, **kwargs):
            seen_enable_graph.append(kwargs.get("enable_graph"))
            if kwargs.get("enable_graph") is True:
                raise RuntimeError("graph backend unavailable")
            return {"results": [{"memory": "vector fallback"}]}

    monkeypatch.setattr(ops_module, "get_mem0_oss", lambda: FakeMem0())

    results = ops_module.search_mem0_via_oss(
        user_id="1",
        query="status",
        top_k=3,
        exclude_conversation_id=None,
    )

    assert [flag for flag in seen_enable_graph if isinstance(flag, bool)] == [True, False]
    assert results[0].content == "vector fallback"


def test_add_uses_enable_graph_when_enabled(monkeypatch):
    """OSS add should pass enable_graph=True when graph add is enabled."""
    ops_module = _reload_mem0_operations(
        monkeypatch,
        MEM0_GRAPH_ENABLED=1,
        MEM0_GRAPH_ADD_ENABLED=1,
        MEM0_GRAPH_FAIL_OPEN=1,
    )

    captured: list[dict[str, Any]] = []

    class FakeMem0:
        def add(self, _fact, **kwargs):
            captured.append(kwargs)
            return {"results": [{"event": "ADD"}]}

    monkeypatch.setattr(ops_module, "get_mem0_oss", lambda: FakeMem0())

    wrote = ops_module.add_mem0_raw_facts(
        user_id="1",
        facts=["remember this"],
        conversation_id=9,
        assistant_message_uuid="abc-123",
    )

    assert wrote is True
    assert captured[0]["enable_graph"] is True


def test_add_graph_failure_retries_vector_only(monkeypatch):
    """When graph add fails, fail-open should retry with enable_graph=False."""
    ops_module = _reload_mem0_operations(
        monkeypatch,
        MEM0_GRAPH_ENABLED=1,
        MEM0_GRAPH_ADD_ENABLED=1,
        MEM0_GRAPH_FAIL_OPEN=1,
    )

    seen_enable_graph: list[Any] = []

    class FakeMem0:
        def add(self, _fact, **kwargs):
            seen_enable_graph.append(kwargs.get("enable_graph"))
            if kwargs.get("enable_graph") is True:
                raise RuntimeError("graph add failed")
            return {"results": [{"event": "ADD"}]}

    monkeypatch.setattr(ops_module, "get_mem0_oss", lambda: FakeMem0())

    wrote = ops_module.add_mem0_raw_facts(
        user_id="1",
        facts=["remember this"],
        conversation_id=9,
        assistant_message_uuid="abc-123",
    )

    assert wrote is True
    assert [flag for flag in seen_enable_graph if isinstance(flag, bool)] == [True, False]


def test_search_mem0_episodic_memories_remains_oss_only(monkeypatch):
    """Episodic search should not use Mem0 cloud fallback in OSS-only mode."""
    ops_module = _reload_mem0_operations(monkeypatch, MEM0_API_KEY="test-key")
    monkeypatch.setattr(ops_module, "search_mem0_via_oss", lambda **_kwargs: [])

    results = ops_module.search_mem0_episodic_memories(
        user_id="1",
        query="status",
        top_k=3,
        exclude_conversation_id=None,
    )

    assert results == []
