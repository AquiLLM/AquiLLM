from __future__ import annotations

from dataclasses import replace

import pytest

from apps.knowledge_graph.projection.memgraph_driver import MemgraphDriverError
from apps.knowledge_graph.retrieval.topology import gateway_service as service
from apps.knowledge_graph.retrieval.topology.contracts import (
    TopologyFailureReason,
    TopologyQueryName,
)
from apps.knowledge_graph.retrieval.topology.failures import TopologyLoadError
from apps.knowledge_graph.retrieval.topology.gateway_config import (
    GatewayConfigError,
    load_topology_gateway_settings,
)
from apps.knowledge_graph.retrieval.topology.gateway_contracts import (
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    SCHEMA_CHECKSUM,
    SCHEMA_VERSION,
    GatewayFailureReason,
    TopologyGatewayRequestV1,
    TopologyGatewaySuccessV1,
    decode_response,
    encode_request,
)

ENV = {
    "KG_TOPOLOGY_GATEWAY_BEARER_TOKEN": "private-gateway-token",
    "KG_MEMGRAPH_URI": "bolt://memgraph_knowledge_graph:7687",
    "KG_MEMGRAPH_DATABASE": "memgraph",
    "KG_MEMGRAPH_QUERY_USERNAME": "gateway",
    "KG_MEMGRAPH_QUERY_PASSWORD": "memgraph-secret",
    "KG_TOPOLOGY_GATEWAY_MAX_REQUEST_BYTES": str(MAX_REQUEST_BYTES),
    "KG_TOPOLOGY_GATEWAY_MAX_RESPONSE_BYTES": str(MAX_RESPONSE_BYTES),
    "KG_TOPOLOGY_GATEWAY_TIMEOUT_MS": "100",
}


class Driver:
    def __init__(self, error: str | None = None) -> None:
        self.error = error
        self.calls = []

    def execute_read(self, query, parameters, *, timeout_seconds, max_records):
        self.calls.append((query, parameters, timeout_seconds, max_records))
        if self.error is not None:
            raise MemgraphDriverError(self.error)
        return ({"ready": 1},)


class Adapter:
    def __init__(self) -> None:
        self.calls = []
        self.failure: Exception | None = None
        self.rows = ({"value": 1},)

    def execute_read(self, *, query, parameters, deadline, max_records):
        self.calls.append((query, parameters, deadline, max_records))
        if self.failure is not None:
            raise self.failure
        return self.rows


def _settings():
    return load_topology_gateway_settings(ENV)


def _request(query=TopologyQueryName.GENERATION_MANIFESTS, max_records=1):
    return encode_request(
        TopologyGatewayRequestV1(query, {"scope": "opaque"}, 99.0, max_records)
    )


def _headers(body: bytes, *, token: bytes = b"private-gateway-token"):
    return [
        (b"authorization", b"Bearer " + token),
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode()),
        (b"x-topology-schema-version", SCHEMA_VERSION.encode()),
        (b"x-topology-schema-checksum", SCHEMA_CHECKSUM.encode()),
    ]


async def _call(path="/v1/topology/read", method="POST", body=b"", headers=None):
    scope = {"type": "http", "path": path, "method": method, "headers": headers or []}
    messages = [{"type": "http.request", "body": body, "more_body": False}]
    sent = []

    async def receive():
        return messages.pop(0)

    async def send(message):
        sent.append(message)

    await service.app(scope, receive, send)
    return sent


def _body(messages):
    return b"".join(message.get("body", b"") for message in messages[1:])


@pytest.fixture(autouse=True)
def runtime(monkeypatch):
    settings = _settings()
    value = service.TopologyGatewayRuntime(settings, Driver(), Adapter())
    monkeypatch.setattr(
        service, "load_topology_gateway_settings", lambda _env: settings
    )
    monkeypatch.setattr(service, "_get_runtime", lambda *_args: value)
    monkeypatch.setattr(service, "monotonic", lambda: 10.0)
    return value


def test_config_is_exact_internal_bounded_and_redacted() -> None:
    settings = _settings()
    rendered = repr(settings)
    assert settings.memgraph_uri == "bolt://memgraph_knowledge_graph:7687"
    assert settings.bearer_token.get_secret_value() == "private-gateway-token"
    assert "private-gateway-token" not in rendered
    assert "memgraph-secret" not in rendered
    assert "memgraph_knowledge_graph" not in rendered

    for update in (
        {"KG_TOPOLOGY_GATEWAY_BEARER_TOKEN": ""},
        {"KG_MEMGRAPH_URI": "bolt://public.example:7687"},
        {"KG_TOPOLOGY_GATEWAY_TIMEOUT_MS": "0100"},
        {"KG_TOPOLOGY_GATEWAY_MAX_RESPONSE_BYTES": str(MAX_RESPONSE_BYTES + 1)},
    ):
        with pytest.raises(GatewayConfigError):
            load_topology_gateway_settings({**ENV, **update})
    with pytest.raises(GatewayConfigError):
        load_topology_gateway_settings(
            {
                **ENV,
                "KG_MEMGRAPH_URI": "",
                "KG_MEMGRAPH_PROJECTION_URI": ENV["KG_MEMGRAPH_URI"],
            }
        )


@pytest.mark.asyncio
async def test_health_has_no_backend_io_and_ready_probe_is_bounded(
    runtime, monkeypatch
) -> None:
    health_calls = 0

    def forbidden(*_args):
        nonlocal health_calls
        health_calls += 1
        raise AssertionError

    monkeypatch.setattr(service, "_get_runtime", forbidden)
    health = await _call(path="/healthz", method="GET")
    assert health[0]["status"] == 200 and health_calls == 0
    monkeypatch.setattr(service, "_get_runtime", lambda *_args: runtime)
    ready = await _call(path="/readyz", method="GET")
    assert ready[0]["status"] == 200
    assert runtime.driver.calls == [("RETURN 1 AS ready", {}, 0.1, 1)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "maximum"),
    (
        (TopologyQueryName.GENERATION_MANIFESTS, 1),
        (TopologyQueryName.AUTOMATIC_MEMBERSHIPS, 2),
        (TopologyQueryName.RELATION_TOPOLOGY, 3),
        (TopologyQueryName.EVIDENCE_MENTIONS, 4),
    ),
)
async def test_read_dispatches_only_named_queries_with_clamped_deadline(
    runtime, query, maximum
) -> None:
    body = _request(query, maximum)
    messages = await _call(body=body, headers=_headers(body))
    assert messages[0]["status"] == 200
    assert decode_response(_body(messages)) == TopologyGatewaySuccessV1(({"value": 1},))
    assert runtime.adapter.calls == [(query, {"scope": "opaque"}, 10.1, maximum)]
    headers = dict(messages[0]["headers"])
    assert headers[b"content-length"] == str(len(_body(messages))).encode()
    assert headers[b"x-topology-schema-checksum"] == SCHEMA_CHECKSUM.encode()


@pytest.mark.asyncio
async def test_auth_and_wire_caps_precede_runtime_and_never_echo(
    runtime, monkeypatch
) -> None:
    calls = 0

    def counted(*_args):
        nonlocal calls
        calls += 1
        return runtime

    monkeypatch.setattr(service, "_get_runtime", counted)
    body = _request()
    unauthorized = await _call(body=body, headers=_headers(body, token=b"wrong"))
    assert unauthorized[0]["status"] == 401 and calls == 0
    noncanonical = b'{"secret-canary":1}'
    malformed = await _call(body=noncanonical, headers=_headers(noncanonical))
    assert malformed[0]["status"] == 400 and calls == 0
    headers = [pair for pair in _headers(b"") if pair[0] != b"content-length"]
    headers.append((b"content-length", str(MAX_REQUEST_BYTES + 1).encode()))
    oversized = await _call(body=b"", headers=headers)
    assert oversized[0]["status"] == 413 and calls == 0
    headers[-1] = (b"content-length", b"9" * 5_000)
    hostile = await _call(body=b"", headers=headers)
    assert hostile[0]["status"] == 413 and calls == 0
    assert b"secret-canary" not in repr((malformed, oversized)).encode()


@pytest.mark.asyncio
async def test_family_and_response_caps_are_local_failures(
    runtime, monkeypatch
) -> None:
    body = _request(TopologyQueryName.GENERATION_MANIFESTS, 65)
    capped = await _call(body=body, headers=_headers(body))
    assert capped[0]["status"] == 422 and runtime.adapter.calls == []
    runtime.adapter.rows = ({"value": "x" * 64},)
    settings = replace(_settings(), max_response_bytes=32)
    monkeypatch.setattr(
        service, "load_topology_gateway_settings", lambda _env: settings
    )
    monkeypatch.setattr(
        service, "_get_runtime", lambda *_args: replace(runtime, settings=settings)
    )
    body = _request()
    response = await _call(body=body, headers=_headers(body))
    assert response[0]["status"] == 422
    runtime.adapter.rows = ({"value": 1}, {"value": 2})
    settings = _settings()
    monkeypatch.setattr(
        service, "load_topology_gateway_settings", lambda _env: settings
    )
    monkeypatch.setattr(service, "_get_runtime", lambda *_args: runtime)
    response = await _call(body=body, headers=_headers(body))
    assert response[0]["status"] == 422


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "reason", "status"),
    (
        (TimeoutError(), GatewayFailureReason.DEADLINE, 504),
        (
            TopologyLoadError(TopologyFailureReason.BACKEND_AUTHENTICATION),
            GatewayFailureReason.AUTHENTICATION,
            401,
        ),
        (
            TopologyLoadError(TopologyFailureReason.BACKEND_PROVENANCE_MISMATCH),
            GatewayFailureReason.PROVENANCE,
            409,
        ),
        (
            TopologyLoadError(TopologyFailureReason.BACKEND_SCHEMA_MISMATCH),
            GatewayFailureReason.SCHEMA,
            502,
        ),
        (
            TopologyLoadError(TopologyFailureReason.BACKEND_UNAVAILABLE),
            GatewayFailureReason.UNAVAILABLE,
            503,
        ),
    ),
)
async def test_backend_failures_use_the_closed_envelope(
    runtime, failure, reason, status
) -> None:
    runtime.adapter.failure = failure
    body = _request()
    messages = await _call(body=body, headers=_headers(body))
    decoded = decode_response(_body(messages))
    assert messages[0]["status"] == status and decoded.reason is reason


@pytest.mark.asyncio
async def test_raw_cypher_and_unknown_routes_do_not_exist() -> None:
    assert (await _call(path="/v1/cypher"))[0]["status"] == 404
    assert (await _call(path="/v1/topology/read", method="GET"))[0]["status"] == 404
