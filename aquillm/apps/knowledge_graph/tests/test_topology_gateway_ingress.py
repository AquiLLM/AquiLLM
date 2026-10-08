"""Both protocols preserve ingress error precedence and exact response framing."""

import pytest

from apps.knowledge_graph.retrieval.topology import gateway_contracts as v1
from apps.knowledge_graph.retrieval.topology import gateway_service as service
from apps.knowledge_graph.retrieval.topology import gateway_snapshot_contracts as v2
from apps.knowledge_graph.retrieval.topology.contracts import TopologyQueryName
from apps.knowledge_graph.tests.test_topology_gateway_service import (
    _body,
    _call,
    _settings,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", [v1, v2])
@pytest.mark.parametrize(
    "case,status,payload",
    [
        ("settings", 503, b'{"ok":false,"reason":"unavailable","status":503}'),
        ("auth", 401, b'{"ok":false,"reason":"authentication","status":401}'),
        ("oversized", 413, b'{"reason":"request_too_large"}'),
        ("framing", 400, b'{"reason":"malformed_request"}'),
        ("body", 400, b'{"reason":"malformed_request"}'),
        ("deadline", 504, b'{"ok":false,"reason":"deadline","status":504}'),
    ],
)
async def test_ingress_rejection_order_and_bytes(
    monkeypatch, protocol, case, status, payload
):
    settings = _settings()
    body = protocol.encode_request(
        v1.TopologyGatewayRequestV1(TopologyQueryName.GENERATION_MANIFESTS, {}, 9.0, 1)
        if protocol is v1
        else v2.TopologyGatewayRequestV2({}, 9.0)
    )
    if case == "body":
        body = b"{}"
    headers = [
        (b"authorization", b"Bearer private-gateway-token"),
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode()),
        (b"x-topology-schema-version", protocol.SCHEMA_VERSION.encode()),
        (b"x-topology-schema-checksum", protocol.SCHEMA_CHECKSUM.encode()),
    ]
    if case in {"settings", "auth"}:
        headers[0] = (b"authorization", b"Bearer wrong")
    if case in {"settings", "auth", "oversized"}:
        headers[2] = (b"content-length", str(settings.max_request_bytes + 1).encode())
    if case in {"settings", "auth", "oversized", "framing"}:
        headers[-1] = (b"x-topology-schema-checksum", b"wrong")

    def load(_):
        if case == "settings":
            raise ValueError("invalid configuration")
        return settings

    monkeypatch.setattr(service, "load_topology_gateway_settings", load)
    monkeypatch.setattr(service, "monotonic", lambda: 10.0)
    monkeypatch.setattr(
        service,
        "_get_runtime",
        lambda *_: pytest.fail("rejected ingress accessed runtime"),
    )
    messages = await _call(
        "/v1/topology/read" if protocol is v1 else v2.SNAPSHOT_PATH,
        body=body,
        headers=headers,
    )
    assert messages[0]["status"] == status
    assert _body(messages) == payload
    assert messages[0]["headers"] == [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(payload)).encode()),
        (b"x-topology-schema-version", protocol.SCHEMA_VERSION.encode()),
        (b"x-topology-schema-checksum", protocol.SCHEMA_CHECKSUM.encode()),
    ]


@pytest.mark.asyncio
async def test_v1_family_cap_precedes_expired_deadline(monkeypatch):
    from apps.knowledge_graph.tests.test_topology_gateway_service import _headers

    monkeypatch.setattr(
        service, "load_topology_gateway_settings", lambda _: _settings()
    )
    monkeypatch.setattr(service, "monotonic", lambda: 10.0)
    monkeypatch.setattr(
        service, "_get_runtime", lambda *_: pytest.fail("cap accessed runtime")
    )
    body = v1.encode_request(
        v1.TopologyGatewayRequestV1(
            TopologyQueryName.GENERATION_MANIFESTS,
            {},
            9.0,
            129,
        )
    )
    messages = await _call(body=body, headers=_headers(body))
    assert messages[0]["status"] == 422
    assert _body(messages) == b'{"ok":false,"reason":"result_cap","status":422}'
