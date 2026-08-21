"""The topology gateway driver is bounded, private, and failure-typed."""

from __future__ import annotations

from io import BytesIO
from unittest.mock import Mock
from urllib.error import HTTPError

import pytest
from scripts.check_retrieval_logging import LANE_PATHS, scan_source

from apps.knowledge_graph.retrieval.topology import contracts
from apps.knowledge_graph.retrieval.topology.failures import TopologyLoadError
from apps.knowledge_graph.retrieval.topology.gateway_client import (
    TopologyGatewayClient,
    TopologyGatewayRequestError,
)
from apps.knowledge_graph.retrieval.topology.gateway_contracts import (
    SCHEMA_CHECKSUM,
    SCHEMA_VERSION,
    GatewayFailureReason,
    TopologyGatewayFailureV1,
    TopologyGatewaySuccessV1,
    encode_response,
)
from apps.knowledge_graph.retrieval.topology.memgraph import (
    MemgraphProjectedTopologyLoader,
)
from apps.knowledge_graph.tests.test_memgraph_topology import K, ready


class Response:
    def __init__(self, body: bytes, *, status: int = 200, headers: dict | None = None):
        self._body = BytesIO(body)
        self.status = status
        self.headers = {
            "Content-Length": str(len(body)),
            "Content-Type": "application/json",
            "X-Topology-Schema-Checksum": SCHEMA_CHECKSUM,
            "X-Topology-Schema-Version": SCHEMA_VERSION,
            **(headers or {}),
        }

    def read(self, amount: int = -1) -> bytes:
        return self._body.read(amount)

    def close(self) -> None:
        self._body.close()


class ReadFailureResponse(Response):
    def __init__(self, error: Exception):
        super().__init__(b"{}")
        self.error = error

    def read(self, _amount: int = -1) -> bytes:
        raise self.error


class Opener:
    def __init__(self, response):
        self.response = response
        self.open = Mock(side_effect=self._open)

    def _open(self, *_args, **_kwargs):
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _http_error(failure: TopologyGatewayFailureV1) -> HTTPError:
    body = encode_response(failure)
    return HTTPError(
        "https://gateway.internal/v1/topology/read",
        failure.status,
        "ignored",
        {
            "Content-Length": str(len(body)),
            "Content-Type": "application/json",
            "X-Topology-Schema-Checksum": SCHEMA_CHECKSUM,
            "X-Topology-Schema-Version": SCHEMA_VERSION,
        },
        BytesIO(body),
    )


def _client(monkeypatch, response, *, ceiling: float = 5.0):
    from apps.knowledge_graph.retrieval.topology import gateway_client

    opener = Opener(response)
    monkeypatch.setattr(gateway_client, "build_opener", lambda *_handlers: opener)
    monkeypatch.setattr(gateway_client.time, "monotonic", lambda: 10.0)
    return TopologyGatewayClient(
        "https://gateway.internal", "top-secret", ceiling
    ), opener


def _read(client):
    return client.execute_read(
        query=contracts.TopologyQueryName.GENERATION_MANIFESTS,
        parameters={"opaque_key": "opaque-value"},
        deadline=12.0,
        max_records=1,
    )


def test_posts_canonical_contract_with_pinned_transport_headers(monkeypatch) -> None:
    client, opener = _client(
        monkeypatch, Response(encode_response(TopologyGatewaySuccessV1(({"id": 1},))))
    )

    assert _read(client) == ({"id": 1},)
    (request,) = opener.open.call_args.args
    assert request.full_url == "https://gateway.internal/v1/topology/read"
    assert request.get_method() == "POST"
    assert request.data == (
        b'{"deadline":12.0,"max_records":1,"parameters":{"opaque_key":"opaque-value"},'
        b'"query":"generation_manifests"}'
    )
    assert request.get_header("Authorization") == "Bearer top-secret"
    assert request.get_header("Content-type") == "application/json"
    assert request.get_header("X-topology-schema-version") == SCHEMA_VERSION
    assert request.get_header("X-topology-schema-checksum") == SCHEMA_CHECKSUM
    assert opener.open.call_args.kwargs["timeout"] == 2.0
    assert isinstance(client, contracts.ProjectedTopologyQueryDriver)


def test_expired_deadline_fails_before_network_io(monkeypatch) -> None:
    client, opener = _client(monkeypatch, Response(b""))

    with pytest.raises(TopologyGatewayRequestError) as captured:
        client.execute_read(
            query=contracts.TopologyQueryName.GENERATION_MANIFESTS,
            parameters={},
            deadline=10.0,
            max_records=1,
        )

    assert captured.value.reason is GatewayFailureReason.DEADLINE
    opener.open.assert_not_called()


@pytest.mark.parametrize(
    ("error", "expected"),
    (
        (OSError("network canary"), TopologyLoadError),
        (TimeoutError("timeout canary"), TopologyGatewayRequestError),
    ),
)
def test_response_read_transport_failures_are_not_schema_failures(
    monkeypatch, error, expected
) -> None:
    client, _ = _client(monkeypatch, ReadFailureResponse(error))

    with pytest.raises(expected) as captured:
        _read(client)

    if expected is TopologyLoadError:
        assert (
            captured.value.reason is contracts.TopologyFailureReason.BACKEND_UNAVAILABLE
        )
    else:
        assert captured.value.reason is GatewayFailureReason.DEADLINE


@pytest.mark.parametrize(
    "headers,body",
    (
        ({"Content-Length": ""}, b""),
        ({"Transfer-Encoding": "chunked"}, b"{}"),
        ({"Content-Type": "application/json; charset=utf-8"}, b"{}"),
        ({"X-Topology-Schema-Version": "wrong"}, b"{}"),
        ({"Content-Length": "02"}, b"{}"),
        ({"Content-Length": "2"}, b"{}x"),
        ({}, b'{"ok":true,"rows":[] }'),
    ),
)
def test_invalid_success_transport_is_a_schema_failure(
    monkeypatch, headers, body
) -> None:
    client, _ = _client(monkeypatch, Response(body, headers=headers))

    with pytest.raises(TopologyLoadError) as captured:
        _read(client)

    assert captured.value.reason is (
        contracts.TopologyFailureReason.BACKEND_SCHEMA_MISMATCH
    )


def test_noncanonical_content_length_is_a_schema_failure(monkeypatch) -> None:
    body = encode_response(TopologyGatewaySuccessV1(()))
    client, _ = _client(
        monkeypatch, Response(body, headers={"Content-Length": f"0{len(body)}"})
    )

    with pytest.raises(TopologyLoadError) as captured:
        _read(client)

    assert captured.value.reason is (
        contracts.TopologyFailureReason.BACKEND_SCHEMA_MISMATCH
    )


@pytest.mark.parametrize(
    ("gateway_reason", "topology_reason"),
    (
        (
            GatewayFailureReason.AUTHENTICATION,
            contracts.TopologyFailureReason.BACKEND_AUTHENTICATION,
        ),
        (
            GatewayFailureReason.UNAVAILABLE,
            contracts.TopologyFailureReason.BACKEND_UNAVAILABLE,
        ),
        (
            GatewayFailureReason.SCHEMA,
            contracts.TopologyFailureReason.BACKEND_SCHEMA_MISMATCH,
        ),
        (
            GatewayFailureReason.PROVENANCE,
            contracts.TopologyFailureReason.BACKEND_PROVENANCE_MISMATCH,
        ),
    ),
)
def test_gateway_wide_failures_keep_their_typed_scope(
    monkeypatch, gateway_reason, topology_reason
) -> None:
    failure = TopologyGatewayFailureV1(gateway_reason)
    client, _ = _client(monkeypatch, _http_error(failure))

    with pytest.raises(TopologyLoadError) as captured:
        _read(client)

    assert captured.value.reason is topology_reason


@pytest.mark.parametrize(
    "reason", (GatewayFailureReason.DEADLINE, GatewayFailureReason.RESULT_CAP)
)
def test_gateway_request_failures_are_typed_and_redacted(monkeypatch, reason) -> None:
    failure = TopologyGatewayFailureV1(reason)
    client, _ = _client(monkeypatch, _http_error(failure))

    with pytest.raises(TopologyGatewayRequestError) as captured:
        _read(client)

    assert captured.value.reason is reason
    rendered = repr(captured.value) + str(captured.value)
    assert "top-secret" not in rendered
    assert "opaque" not in rendered


@pytest.mark.parametrize(
    "reason", (GatewayFailureReason.DEADLINE, GatewayFailureReason.RESULT_CAP)
)
def test_gateway_local_failures_remain_local_at_the_existing_loader_boundary(
    reason,
) -> None:
    class LocalFailureDriver:
        def execute_read(self, **_kwargs):
            raise TopologyGatewayRequestError(reason)

    with pytest.raises(TopologyLoadError) as captured:
        MemgraphProjectedTopologyLoader(LocalFailureDriver()).load(
            ready=ready(),
            seeds=(contracts.ProjectedSeedV1(K[4], 1.0),),
            caps=contracts.TopologyCapsV1(
                contracts.HybridBranchKind.DIRECT, 32, 2, 200, 1_000, 20
            ),
            deadline=42.5,
        )

    assert captured.value.reason is (
        contracts.TopologyFailureReason.DIRECT_TOPOLOGY_TIMEOUT
    )


def test_rejects_external_or_secret_bearing_origins_and_secret_repr() -> None:
    for origin in (
        "https://public.example",
        "https://user:pass@gateway.internal",
        "https://gateway.internal/?x=1",
    ):
        with pytest.raises(ValueError):
            TopologyGatewayClient(origin, "top-secret", 1.0)
    with pytest.raises(ValueError):
        TopologyGatewayClient("https://gateway.internal", "token\u0080", 1.0)
    client = TopologyGatewayClient("http://localhost:8080", "top-secret", 1.0)
    assert "top-secret" not in repr(client)
    assert "localhost" not in repr(client)


def test_logging_checker_covers_gateway_and_rejects_gateway_canaries() -> None:
    assert (
        "aquillm/apps/knowledge_graph/retrieval/topology/gateway_client.py"
        in LANE_PATHS
    )
    source = "\n".join(
        "logger.info('obs.gateway.read', **retrieval_log_fields("
        f"reason='completed', count={name}, elapsed_ms=0))"
        for name in ("payload", "token", "key", "exception", "userinfo")
    )
    findings = scan_source(path=__file__, source=source)
    assert len(findings) == 5
