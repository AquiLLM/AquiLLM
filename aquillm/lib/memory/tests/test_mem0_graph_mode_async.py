"""Async tests for optional Mem0 graph memory add operations."""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from .test_mem0_graph_mode import _reload_mem0_operations


@pytest.mark.asyncio
async def test_search_async_uses_enable_graph_when_enabled(monkeypatch):
    """Async OSS search should pass enable_graph=True when graph search is enabled."""
    ops_module = _reload_mem0_operations(
        monkeypatch,
        MEM0_GRAPH_ENABLED=1,
        MEM0_GRAPH_SEARCH_ENABLED=1,
        MEM0_GRAPH_FAIL_OPEN=1,
    )

    captured: list[dict[str, Any]] = []

    class FakeMem0:
        async def search(self, *_args, **kwargs):
            captured.append(kwargs)
            return {"results": [{"memory": "graph answer"}]}

    async def fake_get_mem0_oss_async():
        return FakeMem0()

    monkeypatch.setattr(ops_module, "get_mem0_oss_async", fake_get_mem0_oss_async)

    results = await ops_module.search_mem0_via_oss_async(
        user_id="1",
        query="who do i collaborate with?",
        top_k=3,
        exclude_conversation_id=None,
    )

    assert results[0].content == "graph answer"
    assert captured[0]["enable_graph"] is True


@pytest.mark.asyncio
async def test_search_async_graph_failure_retries_vector_only(monkeypatch):
    """Async graph search should retry once with enable_graph=False when fail-open is on."""
    ops_module = _reload_mem0_operations(
        monkeypatch,
        MEM0_GRAPH_ENABLED=1,
        MEM0_GRAPH_SEARCH_ENABLED=1,
        MEM0_GRAPH_FAIL_OPEN=1,
    )

    seen_enable_graph: list[Any] = []

    class FakeMem0:
        async def search(self, *_args, **kwargs):
            seen_enable_graph.append(kwargs.get("enable_graph"))
            if kwargs.get("enable_graph") is True:
                raise RuntimeError("graph backend unavailable")
            return {"results": [{"memory": "vector fallback"}]}

    async def fake_get_mem0_oss_async():
        return FakeMem0()

    monkeypatch.setattr(ops_module, "get_mem0_oss_async", fake_get_mem0_oss_async)

    results = await ops_module.search_mem0_via_oss_async(
        user_id="1",
        query="status",
        top_k=3,
        exclude_conversation_id=None,
    )

    assert [flag for flag in seen_enable_graph if isinstance(flag, bool)] == [True, False]
    assert results[0].content == "vector fallback"


@pytest.mark.asyncio
async def test_search_async_graph_attempt_uses_shorter_timeout_budget(monkeypatch):
    """Async graph-enabled search should time out sooner than the full Mem0 timeout."""
    ops_module = _reload_mem0_operations(
        monkeypatch,
        MEM0_GRAPH_ENABLED=1,
        MEM0_GRAPH_SEARCH_ENABLED=1,
        MEM0_GRAPH_FAIL_OPEN=1,
        MEM0_TIMEOUT_SECONDS=15,
    )

    seen_timeouts: list[float] = []
    seen_enable_graph: list[Any] = []

    class FakeMem0:
        async def search(self, *_args, **kwargs):
            seen_enable_graph.append(kwargs.get("enable_graph"))
            return {"results": [{"memory": "vector fallback"}]}

    async def fake_get_mem0_oss_async():
        return FakeMem0()

    async def fake_wait_for(awaitable, timeout):
        seen_timeouts.append(timeout)
        if timeout < 15:
            raise asyncio.TimeoutError()
        return await awaitable

    monkeypatch.setattr(ops_module, "get_mem0_oss_async", fake_get_mem0_oss_async)
    monkeypatch.setattr(ops_module.asyncio, "wait_for", fake_wait_for)

    results = await ops_module.search_mem0_via_oss_async(
        user_id="1",
        query="status",
        top_k=3,
        exclude_conversation_id=None,
    )

    assert results[0].content == "vector fallback"
    assert seen_enable_graph == [False]
    assert seen_timeouts[0] < 15
    assert seen_timeouts[1] == 15


@pytest.mark.asyncio
async def test_search_async_timeout_log_reports_actual_graph_budget(monkeypatch):
    """Timeout warning should report the real graph-attempt timeout, not just the overall timeout."""
    ops_module = _reload_mem0_operations(
        monkeypatch,
        MEM0_GRAPH_ENABLED=1,
        MEM0_GRAPH_SEARCH_ENABLED=1,
        MEM0_GRAPH_FAIL_OPEN=1,
        MEM0_TIMEOUT_SECONDS=15,
    )

    warnings: list[tuple[str, tuple[Any, ...]]] = []

    class FakeMem0:
        async def search(self, *_args, **kwargs):
            return {"results": [{"memory": "vector fallback"}]}

    async def fake_get_mem0_oss_async():
        return FakeMem0()

    async def fake_wait_for(awaitable, timeout):
        if timeout < 15:
            awaitable.close()
            raise asyncio.TimeoutError()
        return await awaitable

    monkeypatch.setattr(ops_module, "get_mem0_oss_async", fake_get_mem0_oss_async)
    monkeypatch.setattr(ops_module.asyncio, "wait_for", fake_wait_for)
    monkeypatch.setattr(
        ops_module.logger,
        "warning",
        lambda message, *args: warnings.append((message, args)),
    )

    results = await ops_module.search_mem0_via_oss_async(
        user_id="1",
        query="status",
        top_k=3,
        exclude_conversation_id=None,
    )

    assert results[0].content == "vector fallback"
    assert warnings[0][0] == (
        "Mem0 OSS async graph search timed out after %.1fs; retrying vector-only "
        "(overall timeout %.1fs)."
    )
    assert warnings[0][1] == (5.0, 15.0)


@pytest.mark.asyncio
async def test_search_async_offloads_mem0_call_to_thread(monkeypatch):
    """Async search should offload Mem0 work so graph search cannot block the event loop."""
    ops_module = _reload_mem0_operations(
        monkeypatch,
        MEM0_GRAPH_ENABLED=1,
        MEM0_GRAPH_SEARCH_ENABLED=1,
        MEM0_GRAPH_FAIL_OPEN=1,
    )

    seen: dict[str, Any] = {}

    class FakeMem0:
        def search(self, *_args, **kwargs):
            seen["kwargs"] = kwargs
            return {"results": [{"memory": "graph answer"}]}

    async def fake_get_mem0_oss_async():
        return FakeMem0()

    async def fake_to_thread(func, *args, **kwargs):
        seen["used_to_thread"] = True
        result = func(*args, **kwargs)
        if inspect.isawaitable(result):
            return result
        return result

    monkeypatch.setattr(ops_module, "get_mem0_oss_async", fake_get_mem0_oss_async)
    monkeypatch.setattr(ops_module.asyncio, "to_thread", fake_to_thread)

    results = await ops_module.search_mem0_via_oss_async(
        user_id="1",
        query="who do i collaborate with?",
        top_k=3,
        exclude_conversation_id=None,
    )

    assert results[0].content == "graph answer"
    assert seen["used_to_thread"] is True
    assert seen["kwargs"]["enable_graph"] is True


@pytest.mark.asyncio
async def test_add_async_uses_enable_graph_when_enabled(monkeypatch):
    """Async OSS add should pass enable_graph=True when graph add is enabled."""
    ops_module = _reload_mem0_operations(
        monkeypatch,
        MEM0_GRAPH_ENABLED=1,
        MEM0_GRAPH_ADD_ENABLED=1,
        MEM0_GRAPH_FAIL_OPEN=1,
    )

    captured: list[dict[str, Any]] = []

    class FakeMem0:
        async def add(self, _fact, **kwargs):
            captured.append(kwargs)
            return {"results": [{"event": "ADD"}]}

    async def fake_get_mem0_oss_async():
        return FakeMem0()

    monkeypatch.setattr(ops_module, "get_mem0_oss_async", fake_get_mem0_oss_async)

    wrote = await ops_module.add_mem0_raw_facts_async(
        user_id="1",
        facts=["Please remember that AquiLLM uses Qdrant and Memgraph for memory."],
        conversation_id=9,
        assistant_message_uuid="abc-123",
    )

    assert wrote is True
    assert captured[0]["enable_graph"] is True


@pytest.mark.asyncio
async def test_add_async_graph_failure_retries_vector_only(monkeypatch):
    """Async graph add should retry once with enable_graph=False when fail-open is on."""
    ops_module = _reload_mem0_operations(
        monkeypatch,
        MEM0_GRAPH_ENABLED=1,
        MEM0_GRAPH_ADD_ENABLED=1,
        MEM0_GRAPH_FAIL_OPEN=1,
    )

    seen_enable_graph: list[Any] = []

    class FakeMem0:
        async def add(self, _fact, **kwargs):
            seen_enable_graph.append(kwargs.get("enable_graph"))
            if kwargs.get("enable_graph") is True:
                raise RuntimeError("graph add failed")
            return {"results": [{"event": "ADD"}]}

    async def fake_get_mem0_oss_async():
        return FakeMem0()

    monkeypatch.setattr(ops_module, "get_mem0_oss_async", fake_get_mem0_oss_async)

    wrote = await ops_module.add_mem0_raw_facts_async(
        user_id="1",
        facts=["Please remember that AquiLLM uses Qdrant and Memgraph for memory."],
        conversation_id=9,
        assistant_message_uuid="abc-123",
    )

    assert wrote is True
    assert [flag for flag in seen_enable_graph if isinstance(flag, bool)] == [True, False]
