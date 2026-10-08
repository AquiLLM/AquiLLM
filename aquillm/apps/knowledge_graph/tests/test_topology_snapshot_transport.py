import asyncio
from dataclasses import replace
from email.message import Message
from unittest.mock import patch

import pytest

from apps.knowledge_graph.projection.topology_adapter import (
    Neo4jProjectedTopologyQueryAdapter,
)
from apps.knowledge_graph.retrieval.projected_types import (
    canonical_projected_snapshot_bytes,
)
from apps.knowledge_graph.retrieval.topology import gateway_client, memgraph
from apps.knowledge_graph.retrieval.topology import gateway_service as service
from apps.knowledge_graph.retrieval.topology import gateway_snapshot_contracts as v2
from apps.knowledge_graph.retrieval.topology.failures import TopologyLoadError
from apps.knowledge_graph.retrieval.topology.gateway_contracts import (
    GatewayFailureReason,
    TopologyGatewayFailureV1,
)
from apps.knowledge_graph.tests.test_topology_gateway_client import Response
from apps.knowledge_graph.tests.test_topology_gateway_service import (
    _body,
    _call,
    _settings,
)
from apps.knowledge_graph.tests.test_topology_snapshot_adapter import fixture


def headers(body):
    return [
        (b"authorization", b"Bearer private-gateway-token"),
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode()),
        (b"x-topology-schema-version", v2.SCHEMA_VERSION.encode()),
        (b"x-topology-schema-checksum", v2.SCHEMA_CHECKSUM.encode()),
    ]


def setup(monkeypatch, *, cap=None):
    driver, args, parameters = fixture()
    adapter = Neo4jProjectedTopologyQueryAdapter(driver, clock=lambda: 40.0)
    settings = _settings()
    if cap is not None:
        settings = replace(settings, max_response_bytes=cap)
    runtime = service.TopologyGatewayRuntime(settings, driver, adapter)
    monkeypatch.setattr(service, "load_topology_gateway_settings", lambda _: settings)
    monkeypatch.setattr(service, "_get_runtime", lambda *_: runtime)
    monkeypatch.setattr(service, "monotonic", lambda: 40.0)
    return runtime, args, parameters


@pytest.mark.asyncio
async def test_v2_asgi_attests_and_returns_snapshot_under_its_own_schema(monkeypatch):
    runtime, _, parameters = setup(monkeypatch)
    body = v2.encode_request(v2.TopologyGatewayRequestV2(parameters, 42.5))
    with patch.object(
        runtime.adapter, "_decode", wraps=runtime.adapter._decode
    ) as decode:
        result = await _call(v2.SNAPSHOT_PATH, body=body, headers=headers(body))
    assert result[0]["status"] == 200
    assert (
        dict(result[0]["headers"])[b"x-topology-schema-checksum"]
        == v2.SCHEMA_CHECKSUM.encode()
    )
    assert type(v2.decode_response(_body(result))) is v2.TopologyGatewaySnapshotV2
    assert decode.call_count == 1
    assert "AS collection_key" in runtime.driver.calls[0][0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["auth", "schema", "duplicate", "transfer", "length", "body"]
)
async def test_validation_precedes_runtime_access(monkeypatch, kind):
    _, _, parameters = setup(monkeypatch)
    body = v2.encode_request(v2.TopologyGatewayRequestV2(parameters, 42.5))
    wire = headers(body)
    if kind == "auth":
        wire[0] = (b"authorization", b"Bearer wrong")
    elif kind == "schema":
        wire[-1] = (wire[-1][0], b"wrong")
    elif kind == "duplicate":
        wire.append(wire[-1])
    elif kind == "transfer":
        wire.append((b"transfer-encoding", b"chunked"))
    elif kind == "length":
        wire[2] = (b"content-length", b"00")
    else:
        body = b"{}"
        wire = headers(body)

    def forbidden(*_):
        pytest.fail("malformed requests cannot access the runtime")

    monkeypatch.setattr(service, "_get_runtime", forbidden)
    result = await _call(v2.SNAPSHOT_PATH, body=body, headers=wire)
    assert result[0]["status"] == (401 if kind == "auth" else 400)


@pytest.mark.asyncio
async def test_tighter_wire_ceiling_selects_families_only_after_full_build(monkeypatch):
    runtime, _, parameters = setup(monkeypatch, cap=800)
    body = v2.encode_request(v2.TopologyGatewayRequestV2(parameters, 42.5))
    result = await _call(v2.SNAPSHOT_PATH, body=body, headers=headers(body))
    assert result[0]["status"] == 200
    assert (
        type(v2.decode_response(_body(result))) is v2.TopologyGatewayFamilyTransportV2
    )
    assert len(runtime.adapter._cache) == 1


@pytest.mark.asyncio
async def test_even_directive_over_wire_ceiling_returns_result_cap(monkeypatch):
    _, _, parameters = setup(monkeypatch, cap=100)
    body = v2.encode_request(v2.TopologyGatewayRequestV2(parameters, 42.5))
    result = await _call(v2.SNAPSHOT_PATH, body=body, headers=headers(body))
    assert result[0]["status"] == 422
    assert v2.decode_response(_body(result)).reason is GatewayFailureReason.RESULT_CAP


def test_enabled_client_loader_uses_one_http_exchange_and_matches_v1(monkeypatch):
    runtime, args, _ = setup(monkeypatch)
    golden = memgraph.MemgraphProjectedTopologyLoader(runtime.adapter).load(**args)
    monkeypatch.setattr(gateway_client.time, "monotonic", lambda: 40.0)
    monkeypatch.setattr(memgraph, "monotonic", lambda: 40.0)
    calls = []

    class Bridge:
        def open(self, request, *, timeout):
            calls.append((request, timeout))
            messages = asyncio.run(
                _call(
                    request.selector,
                    body=request.data,
                    headers=[
                        (k.lower().encode(), v.encode())
                        for k, v in request.header_items()
                    ]
                    + [(b"content-length", str(len(request.data)).encode())],
                )
            )
            return Response(
                _body(messages),
                status=messages[0]["status"],
                headers={
                    k.decode().title(): v.decode() for k, v in messages[0]["headers"]
                },
            )

    monkeypatch.setattr(gateway_client, "build_opener", lambda *_: Bridge())
    client = gateway_client.TopologyGatewayClient(
        "https://gateway.internal",
        "private-gateway-token",
        5.0,
        snapshot_enabled=True,
    )
    actual = memgraph.MemgraphProjectedTopologyLoader(client).load(**args)
    assert canonical_projected_snapshot_bytes(
        actual
    ) == canonical_projected_snapshot_bytes(golden)
    assert len(calls) == 1
    assert calls[0][0].selector == v2.SNAPSHOT_PATH


@pytest.mark.parametrize("enabled", [0, 1, "true", None])
def test_snapshot_opt_in_requires_an_exact_bool(enabled):
    with pytest.raises(ValueError):
        gateway_client.TopologyGatewayClient(
            "https://gateway.internal", "secret", 5.0, snapshot_enabled=enabled
        )


def test_disabled_snapshot_client_fails_before_network(monkeypatch):
    monkeypatch.setattr(
        gateway_client, "build_opener", lambda *_: pytest.fail("disabled")
    )
    client = gateway_client.TopologyGatewayClient(
        "https://gateway.internal", "secret", 5.0
    )
    assert client.snapshot_enabled is False
    with pytest.raises(ValueError):
        client.execute_snapshot(parameters={}, deadline=42.5)


@pytest.mark.parametrize("bad", ["schema", "duplicate", "failure", "late"])
def test_v2_client_failure_never_attempts_another_request(monkeypatch, bad):
    runtime, args, parameters = setup(monkeypatch)
    result = runtime.adapter.execute_snapshot(parameters=parameters, deadline=42.5)
    raw = v2.encode_response(result)
    clock = [40.0]
    h = Message()
    for key, value in {
        "Content-Length": str(len(raw)),
        "Content-Type": "application/json",
        "X-Topology-Schema-Version": v2.SCHEMA_VERSION,
        "X-Topology-Schema-Checksum": v2.SCHEMA_CHECKSUM,
    }.items():
        h[key] = value
    if bad == "schema":
        h.replace_header("X-Topology-Schema-Version", "wrong")
    if bad == "duplicate":
        h["X-Topology-Schema-Checksum"] = v2.SCHEMA_CHECKSUM
    if bad == "failure":
        raw = v2.encode_response(
            TopologyGatewayFailureV1(GatewayFailureReason.DEADLINE)
        )
        h.replace_header("Content-Length", str(len(raw)))
    response = Response(raw, status=504 if bad == "failure" else 200)
    response.headers = h
    calls = []

    class Opener:
        def open(self, *_args, **_kwargs):
            calls.append(1)
            if bad == "late":
                clock[0] = 43.0
            return response

    monkeypatch.setattr(gateway_client.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(memgraph, "monotonic", lambda: clock[0])
    monkeypatch.setattr(gateway_client, "build_opener", lambda *_: Opener())
    client = gateway_client.TopologyGatewayClient(
        "https://gateway.internal", "secret", 5.0, snapshot_enabled=True
    )
    with pytest.raises(TopologyLoadError):
        memgraph.MemgraphProjectedTopologyLoader(client).load(**args)
    assert calls == [1]
