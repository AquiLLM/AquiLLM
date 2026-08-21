"""Dependency-free ASGI service for four fixed projected-topology reads."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from hmac import compare_digest
from os import environ
from time import monotonic
from typing import Any, Final

from apps.knowledge_graph.projection.memgraph_driver import Neo4jMemgraphDriver
from apps.knowledge_graph.projection.topology_adapter import (
    Neo4jProjectedTopologyQueryAdapter,
)

from .contracts import TopologyFailureReason, TopologyQueryName
from .failures import TopologyLoadError
from .gateway_config import (
    TopologyGatewaySettings,
    load_topology_gateway_settings,
)
from .gateway_contracts import (
    SCHEMA_CHECKSUM,
    SCHEMA_VERSION,
    GatewayFailureReason,
    TopologyGatewayFailureV1,
    TopologyGatewaySuccessV1,
    decode_request,
    encode_response,
)

_FAMILY_CAPS: Final = {
    TopologyQueryName.GENERATION_MANIFESTS: 64,
    TopologyQueryName.AUTOMATIC_MEMBERSHIPS: 200,
    TopologyQueryName.RELATION_TOPOLOGY: 1_000,
    TopologyQueryName.EVIDENCE_MENTIONS: 3_400,
}
_MALFORMED = b'{"reason":"malformed_request"}'
_OVERSIZED = b'{"reason":"request_too_large"}'
_OK = b'{"status":"ok"}'


@dataclass(frozen=True, slots=True)
class TopologyGatewayRuntime:
    settings: TopologyGatewaySettings
    driver: object
    adapter: object


_runtime: TopologyGatewayRuntime | None = None


def _get_runtime(
    settings: TopologyGatewaySettings | None = None,
) -> TopologyGatewayRuntime:
    global _runtime
    if settings is None:
        settings = load_topology_gateway_settings(environ)
    if _runtime is None:
        driver = Neo4jMemgraphDriver(
            settings.memgraph_uri,
            settings.query_username,
            settings.query_password.get_secret_value(),
            database=settings.memgraph_database,
        )
        _runtime = TopologyGatewayRuntime(
            settings, driver, Neo4jProjectedTopologyQueryAdapter(driver)
        )
    if _runtime.settings != settings:
        raise RuntimeError("topology gateway configuration drift")
    return _runtime


def _header_values(scope: dict[str, Any], name: bytes) -> tuple[bytes, ...]:
    return tuple(
        value
        for key, value in scope.get("headers", ())
        if type(key) is bytes and type(value) is bytes and key.lower() == name
    )


def _single_header(scope: dict[str, Any], name: bytes) -> bytes | None:
    values = _header_values(scope, name)
    return values[0] if len(values) == 1 else None


def _response_headers(body: bytes) -> list[tuple[bytes, bytes]]:
    return [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode()),
        (b"x-topology-schema-version", SCHEMA_VERSION.encode()),
        (b"x-topology-schema-checksum", SCHEMA_CHECKSUM.encode()),
    ]


async def _respond(
    send: Callable[..., Awaitable[None]], status: int, body: bytes
) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": _response_headers(body),
        }
    )
    await send({"type": "http.response.body", "body": body})


async def _failure(
    send: Callable[..., Awaitable[None]], reason: GatewayFailureReason
) -> None:
    value = TopologyGatewayFailureV1(reason)
    await _respond(send, value.status, encode_response(value))


async def _read_body(
    receive: Callable[..., Awaitable[dict[str, Any]]], expected: int
) -> bytes | None:
    chunks: list[bytes] = []
    size = 0
    while True:
        message = await receive()
        if message.get("type") != "http.request":
            return None
        chunk = message.get("body", b"")
        if type(chunk) is not bytes:
            return None
        size += len(chunk)
        if size > expected:
            return None
        chunks.append(chunk)
        if not message.get("more_body", False):
            body = b"".join(chunks)
            return body if len(body) == expected else None


def _wire_valid(scope: dict[str, Any]) -> bool:
    return (
        _single_header(scope, b"content-type") == b"application/json"
        and _single_header(scope, b"x-topology-schema-version")
        == SCHEMA_VERSION.encode()
        and _single_header(scope, b"x-topology-schema-checksum")
        == SCHEMA_CHECKSUM.encode()
        and not _header_values(scope, b"transfer-encoding")
    )


def _length(scope: dict[str, Any], maximum: int) -> int | None:
    raw = _single_header(scope, b"content-length")
    if raw is None or not raw.isascii():
        return None
    text = raw.decode("ascii")
    if not text.isdecimal() or (len(text) > 1 and text[0] == "0"):
        return None
    return maximum + 1 if len(text) > len(str(maximum)) else int(text)


def _mapped_reason(error: TopologyLoadError) -> GatewayFailureReason:
    return {
        TopologyFailureReason.BACKEND_AUTHENTICATION: (
            GatewayFailureReason.AUTHENTICATION
        ),
        TopologyFailureReason.BACKEND_UNAVAILABLE: GatewayFailureReason.UNAVAILABLE,
        TopologyFailureReason.BACKEND_PROVENANCE_MISMATCH: (
            GatewayFailureReason.PROVENANCE
        ),
        TopologyFailureReason.READINESS_MISMATCH: GatewayFailureReason.PROVENANCE,
        TopologyFailureReason.BACKEND_SCHEMA_MISMATCH: GatewayFailureReason.SCHEMA,
        TopologyFailureReason.AUTHORIZATION_CONTEXT_INVALID: (
            GatewayFailureReason.SCHEMA
        ),
        TopologyFailureReason.DIRECT_TOPOLOGY_INVALID: GatewayFailureReason.SCHEMA,
        TopologyFailureReason.EXTENDED_TOPOLOGY_INVALID: GatewayFailureReason.SCHEMA,
        TopologyFailureReason.OVERALL_DEADLINE: GatewayFailureReason.DEADLINE,
        TopologyFailureReason.DIRECT_TOPOLOGY_TIMEOUT: GatewayFailureReason.DEADLINE,
        TopologyFailureReason.EXTENDED_TOPOLOGY_TIMEOUT: GatewayFailureReason.DEADLINE,
    }[error.reason]


async def healthz(
    _scope: dict[str, Any], _receive: Callable[..., Awaitable[dict[str, Any]]], send
) -> None:
    await _respond(send, 200, _OK)


async def readyz(scope, receive, send) -> None:
    del scope, receive
    try:
        settings = load_topology_gateway_settings(environ)
        runtime = _get_runtime(settings)
        rows = runtime.driver.execute_read(
            "RETURN 1 AS ready",
            {},
            timeout_seconds=settings.timeout_ms / 1000.0,
            max_records=1,
        )
        if rows != ({"ready": 1},):
            raise RuntimeError("invalid readiness response")
    except Exception:
        await _failure(send, GatewayFailureReason.UNAVAILABLE)
        return
    await _respond(send, 200, _OK)


async def topology_read(scope, receive, send) -> None:
    try:
        settings = load_topology_gateway_settings(environ)
    except Exception:
        await _failure(send, GatewayFailureReason.UNAVAILABLE)
        return
    expected = b"Bearer " + settings.bearer_token.get_secret_value().encode("ascii")
    authorization = _single_header(scope, b"authorization") or b""
    if not compare_digest(authorization, expected):
        await _failure(send, GatewayFailureReason.AUTHENTICATION)
        return
    length = _length(scope, settings.max_request_bytes)
    if length is not None and length > settings.max_request_bytes:
        await _respond(send, 413, _OVERSIZED)
        return
    if length is None or not _wire_valid(scope):
        await _respond(send, 400, _MALFORMED)
        return
    body = await _read_body(receive, length)
    try:
        request = decode_request(body) if body is not None else None
    except ValueError:
        request = None
    if request is None:
        await _respond(send, 400, _MALFORMED)
        return
    if request.max_records > _FAMILY_CAPS[request.query]:
        await _failure(send, GatewayFailureReason.RESULT_CAP)
        return
    deadline = min(request.deadline, monotonic() + settings.timeout_ms / 1000.0)
    if deadline <= monotonic():
        await _failure(send, GatewayFailureReason.DEADLINE)
        return
    try:
        runtime = _get_runtime(settings)
        rows = runtime.adapter.execute_read(
            query=request.query,
            parameters=request.parameters,
            deadline=deadline,
            max_records=request.max_records,
        )
        if type(rows) is not tuple:
            raise TopologyLoadError(TopologyFailureReason.BACKEND_SCHEMA_MISMATCH)
        if len(rows) > request.max_records:
            raise OverflowError
        try:
            response = TopologyGatewaySuccessV1(rows)
        except (TypeError, ValueError):
            raise TopologyLoadError(
                TopologyFailureReason.BACKEND_SCHEMA_MISMATCH
            ) from None
        payload = encode_response(response)
        if len(payload) > settings.max_response_bytes:
            raise OverflowError
    except OverflowError:
        await _failure(send, GatewayFailureReason.RESULT_CAP)
        return
    except TimeoutError:
        await _failure(send, GatewayFailureReason.DEADLINE)
        return
    except TopologyLoadError as error:
        await _failure(send, _mapped_reason(error))
        return
    except Exception:
        await _failure(send, GatewayFailureReason.UNAVAILABLE)
        return
    await _respond(send, 200, payload)


async def app(scope, receive, send) -> None:
    if scope.get("type") != "http":
        return
    route = (scope.get("method"), scope.get("path"))
    if route == ("GET", "/healthz"):
        await healthz(scope, receive, send)
    elif route == ("GET", "/readyz"):
        await readyz(scope, receive, send)
    elif route == ("POST", "/v1/topology/read"):
        await topology_read(scope, receive, send)
    else:
        await _respond(send, 404, b'{"reason":"not_found"}')


def run() -> None:
    import uvicorn

    uvicorn.run(
        "apps.knowledge_graph.retrieval.topology.gateway_service:app",
        access_log=False,
    )


if __name__ == "__main__":
    run()
