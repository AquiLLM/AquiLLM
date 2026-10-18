"""Relative ingress deadlines and shared gateway worker admission."""

import asyncio
from dataclasses import replace
from threading import Event

import pytest

from apps.knowledge_graph.retrieval.topology import gateway_service as service
from apps.knowledge_graph.retrieval.topology.frontier_gateway_contracts_v2 import (
    BINARY_CHECKSUM,
    BINARY_PATH,
    BINARY_VERSION,
    CHECKSUM,
    PATH,
    VERSION,
    FrontierGatewayBatchRequestV2,
    FrontierGatewayRequestV2,
    encode_request,
)
from apps.knowledge_graph.tests.frontier_fixture_support import FrontierFixture
from apps.knowledge_graph.tests.frontier_index_fixture_driver import (
    FrontierIndexFixtureDriver,
)
from apps.knowledge_graph.tests.test_topology_gateway_service import _call, _settings


@pytest.fixture(params=[False, True], ids=["scalar", "binary"])
def binary(request):
    return request.param


def request(fixture, timeout=b"100", *, binary=False):
    body = encode_request(
        FrontierGatewayRequestV2(
            "manifest",
            fixture.bundles[0].generation.generation_key,
            (),
            fixture.group.scope.ready,
            1048576,
        )
    )
    if binary:
        generation = fixture.bundles[0].generation.generation_key
        body = encode_request(
            FrontierGatewayBatchRequestV2(
                "manifest_batch",
                ((generation, generation),),
                fixture.group.scope.ready,
                1048576,
            )
        )
    return body, [
        (b"authorization", b"Bearer private-gateway-token"),
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode()),
        (
            b"x-topology-schema-version",
            (BINARY_VERSION if binary else VERSION).encode(),
        ),
        (
            b"x-topology-schema-checksum",
            (BINARY_CHECKSUM if binary else CHECKSUM).encode(),
        ),
        (b"x-frontier-timeout-ms", timeout),
    ]


@pytest.mark.asyncio
async def test_relative_timeout_consumes_ingress_body_time(monkeypatch, binary):
    fixture = FrontierFixture()
    driver = FrontierIndexFixtureDriver(fixture.bundles)
    runtime = service.TopologyGatewayRuntime(_settings(), driver, object())
    monkeypatch.setattr(
        service, "load_topology_gateway_settings", lambda _: runtime.settings
    )
    monkeypatch.setattr(service, "_get_runtime", lambda *_: runtime)
    now = [10.0]
    monkeypatch.setattr(service, "monotonic", lambda: now[0])
    body, headers = request(fixture, binary=binary)

    async def receive():
        now[0] += 0.2
        return {"type": "http.request", "body": body, "more_body": False}

    sent = []

    async def send(message):
        sent.append(message)

    await service.app(
        {
            "type": "http",
            "method": "POST",
            "path": BINARY_PATH if binary else PATH,
            "headers": headers,
        },
        receive,
        send,
    )
    assert sent[0]["status"] == 504
    assert driver.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [b"0", b"010", b"60001", b"-1", b"NaN"])
async def test_noncanonical_relative_timeout_rejected_before_runtime(
    monkeypatch, timeout, binary
):
    monkeypatch.setattr(
        service, "load_topology_gateway_settings", lambda _: _settings()
    )
    monkeypatch.setattr(
        service,
        "_get_runtime",
        lambda *_: pytest.fail("invalid ingress touched runtime"),
    )
    body, headers = request(FrontierFixture(), timeout, binary=binary)
    result = await _call(
        path=BINARY_PATH if binary else PATH, body=body, headers=headers
    )
    assert result[0]["status"] == 400


@pytest.mark.asyncio
async def test_relative_frontier_uses_existing_worker_capacity(monkeypatch, binary):
    fixture = FrontierFixture()
    driver = FrontierIndexFixtureDriver(fixture.bundles)
    runtime = service.TopologyGatewayRuntime(
        replace(_settings(), timeout_ms=500), driver, object()
    )
    monkeypatch.setattr(
        service, "load_topology_gateway_settings", lambda _: runtime.settings
    )
    monkeypatch.setattr(service, "_get_runtime", lambda *_: runtime)

    class FullPool:
        async def run(self, *_args, **_kwargs):
            raise RuntimeError("capacity exhausted")

    monkeypatch.setattr(service, "GATEWAY_WORKERS", FullPool())
    body, headers = request(fixture, binary=binary)
    result = await _call(
        path=BINARY_PATH if binary else PATH, body=body, headers=headers
    )
    assert result[0]["status"] == 503
    assert driver.calls == []


@pytest.mark.asyncio
async def test_timed_out_frontier_workers_keep_slots_until_driver_finishes(
    monkeypatch, binary
):
    from apps.knowledge_graph.retrieval.topology.gateway_workers import (
        GatewayWorkerPool,
    )

    fixture, release = FrontierFixture(), Event()
    entered = []

    class BlockingDriver(FrontierIndexFixtureDriver):
        def execute_read(self, *args, **kwargs):
            entered.append(True)
            assert release.wait(3)
            return super().execute_read(*args, **kwargs)

    pool = GatewayWorkerPool()
    driver = BlockingDriver(fixture.bundles)
    runtime = service.TopologyGatewayRuntime(_settings(), driver, object())
    monkeypatch.setattr(
        service, "load_topology_gateway_settings", lambda _: runtime.settings
    )
    monkeypatch.setattr(service, "_get_runtime", lambda *_: runtime)
    monkeypatch.setattr(service, "GATEWAY_WORKERS", pool)
    body, headers = request(fixture, b"100", binary=binary)
    path = BINARY_PATH if binary else PATH
    tasks = [
        asyncio.create_task(_call(path=path, body=body, headers=headers))
        for _ in range(4)
    ]
    try:
        results = await asyncio.gather(*tasks)
        assert len(entered) == 4
        assert all(r[0]["status"] == 504 for r in results)
        rejected = await _call(path=path, body=body, headers=headers)
        assert rejected[0]["status"] == 503
        assert len(entered) == 4
        release.set()
        for _ in range(200):
            recovered = await _call(path=path, body=body, headers=headers)
            if recovered[0]["status"] == 200:
                break
            await asyncio.sleep(0.001)
        assert recovered[0]["status"] == 200
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        pool._executor.shutdown(wait=True)
