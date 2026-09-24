import asyncio
from dataclasses import replace
from threading import Event
from time import monotonic

import pytest

from apps.knowledge_graph.retrieval.topology import gateway_service as service
from apps.knowledge_graph.tests.test_topology_gateway_service import (
    Adapter, Driver, _call, _headers, _request, _settings,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["topology", "readiness"])
async def test_blocked_read_leaves_health_and_independent_reads_responsive(monkeypatch, route):
    entered, release = Event(), Event()

    class BlockingAdapter(Adapter):
        def execute_read(self, **kwargs):
            if route == "topology" and not entered.is_set():
                entered.set()
                assert release.wait(2), "event loop could not service independent requests"
            return super().execute_read(**kwargs)

    class BlockingDriver(Driver):
        def execute_read(self, *args, **kwargs):
            entered.set()
            assert release.wait(2)
            return super().execute_read(*args, **kwargs)

    settings = replace(_settings(), timeout_ms=1000)
    runtime = service.TopologyGatewayRuntime(settings, BlockingDriver(), BlockingAdapter())
    monkeypatch.setattr(service, "load_topology_gateway_settings", lambda _: runtime.settings)
    monkeypatch.setattr(service, "_get_runtime", lambda *_: runtime)
    monkeypatch.setattr(service, "monotonic", lambda: 10.0)
    body = _request()
    blocked = asyncio.create_task(
        _call(body=body, headers=_headers(body)) if route == "topology"
        else _call(path="/readyz", method="GET")
    )
    try:
        for _ in range(200):
            if entered.is_set():
                break
            await asyncio.sleep(0.001)
        health = await _call(path="/healthz", method="GET")
        independent = await _call(body=body, headers=_headers(body))
        assert not blocked.done()
        assert health[0]["status"] == independent[0]["status"] == 200
    finally:
        release.set()
        await blocked


@pytest.mark.asyncio
@pytest.mark.parametrize("abandon", ["timeout", "cancel"])
async def test_abandoned_workers_retain_all_capacity_until_actual_completion(abandon):
    from apps.knowledge_graph.retrieval.topology.gateway_workers import GatewayWorkerPool

    pool = GatewayWorkerPool()
    entered = [Event() for _ in range(4)]
    release = Event()

    def block(index):
        entered[index].set()
        assert release.wait(2)
        return index

    tasks = [asyncio.create_task(pool.run(
        block, index, expires=monotonic() + (0.05 if abandon == "timeout" else 1.0),
    )) for index in range(4)]
    try:
        for _ in range(200):
            if all(event.is_set() for event in entered):
                break
            await asyncio.sleep(0.001)
        assert all(event.is_set() for event in entered)
        with pytest.raises(RuntimeError, match="capacity"):
            await pool.run(lambda: None, expires=monotonic() + 1)
        if abandon == "cancel":
            for task in tasks:
                task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        expected = TimeoutError if abandon == "timeout" else asyncio.CancelledError
        assert all(isinstance(result, expected) for result in results)
        with pytest.raises(RuntimeError, match="capacity"):
            await pool.run(lambda: None, expires=monotonic() + 1)
        release.set()
        for _ in range(200):
            try:
                recovered = await pool.run(lambda: "recovered", expires=monotonic() + 1)
                break
            except RuntimeError:
                await asyncio.sleep(0.001)
        else:
            pytest.fail("finished workers did not release capacity")
        assert recovered == "recovered"
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        pool._executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_expired_worker_never_executes_and_late_success_is_ignored():
    from apps.knowledge_graph.retrieval.topology.gateway_workers import GatewayWorkerPool

    pool = GatewayWorkerPool()
    called = []
    try:
        with pytest.raises(TimeoutError):
            await pool.run(lambda: called.append(True), expires=1.0, clock=lambda: 2.0)
        times = iter((1.0, 3.0))
        with pytest.raises(TimeoutError):
            await pool.run(lambda: called.append(True), expires=2.0, clock=lambda: next(times))
        assert called == []

        def late():
            called.append(True)
            return "late"

        times = iter((1.0, 1.0, 3.0))
        with pytest.raises(TimeoutError):
            await pool.run(late, expires=2.0, clock=lambda: next(times))
        assert called == [True]
        assert await pool.run(lambda: "current", expires=monotonic() + 1) == "current"
    finally:
        pool._executor.shutdown(wait=True)
