"""Dependency-free ASGI service for fixed projected-topology reads."""

from dataclasses import dataclass
from functools import partial
from hmac import compare_digest
from os import environ
from time import monotonic
from typing import Final

from apps.knowledge_graph.projection.memgraph_driver import Neo4jMemgraphDriver
from apps.knowledge_graph.projection.topology_adapter import (
    Neo4jProjectedTopologyQueryAdapter,
)

from .contracts import TopologyFailureReason, TopologyQueryName
from .failures import TopologyLoadError, TopologyResultCapError
from .gateway_config import (
    TopologyGatewaySettings,
    load_topology_gateway_settings,
)
from .gateway_contracts import (
    SCHEMA_CHECKSUM,
    SCHEMA_VERSION,
    GatewayFailureReason,
    TopologyGatewaySuccessV1,
    decode_request,
    encode_response,
)
from .gateway_wire import (
    _failure as _failure,
)
from .gateway_wire import (
    _header_values as _header_values,
)
from .gateway_wire import (
    _length as _length,
)
from .gateway_wire import (
    _read_body as _read_body,
)
from .gateway_wire import (
    _respond as _respond,
)
from .gateway_wire import (
    _response_headers as _response_headers,
)
from .gateway_wire import (
    _single_header as _single_header,
)
from .gateway_wire import (
    _wire_valid as _wire_valid,
)
from .gateway_workers import GATEWAY_WORKERS

_FAMILY_CAPS: Final = {
    TopologyQueryName.GENERATION_MANIFESTS: 128,
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
            # Retrying a managed read would reuse a stale per-transaction
            # timeout beyond the caller's absolute branch deadline.
            max_transaction_retry_time=0.0,
        )
        _runtime = TopologyGatewayRuntime(
            settings, driver, Neo4jProjectedTopologyQueryAdapter(driver)
        )
    if _runtime.settings != settings:
        raise RuntimeError("topology gateway configuration drift")
    return _runtime


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


async def healthz(_scope, _receive, send) -> None:
    await _respond(send, 200, _OK)


async def readyz(scope, receive, send) -> None:
    del scope, receive
    try:
        settings = load_topology_gateway_settings(environ)
        runtime = _get_runtime(settings)
        rows = await GATEWAY_WORKERS.run(
            runtime.driver.execute_read,
            "RETURN 1 AS ready",
            {},
            expires=monotonic() + settings.timeout_ms / 1000.0,
            clock=monotonic,
            timeout_seconds=settings.timeout_ms / 1000.0,
            max_records=1,
        )
        if rows != ({"ready": 1},):
            raise RuntimeError("invalid readiness response")
    except Exception:
        await _failure(send, GatewayFailureReason.UNAVAILABLE)
        return
    await _respond(send, 200, _OK)


async def _validate_ingress(
    scope,
    receive,
    send,
    *,
    decoder,
    version,
    checksum,
    exceeds_cap=None,
):
    """Validate in fixed precedence before either route may access its runtime."""
    respond = partial(_respond, send, version=version, checksum=checksum)
    failure = partial(_failure, send, version=version, checksum=checksum)
    try:
        settings = load_topology_gateway_settings(environ)
    except Exception:
        await failure(GatewayFailureReason.UNAVAILABLE)
        return
    expected = b"Bearer " + settings.bearer_token.get_secret_value().encode("ascii")
    authorization = _single_header(scope, b"authorization") or b""
    if not compare_digest(authorization, expected):
        await failure(GatewayFailureReason.AUTHENTICATION)
        return
    length = _length(scope, settings.max_request_bytes)
    if length is not None and length > settings.max_request_bytes:
        await respond(413, _OVERSIZED)
        return
    if length is None or not _wire_valid(scope, version=version, checksum=checksum):
        await respond(400, _MALFORMED)
        return
    body = await _read_body(receive, length)
    try:
        request = decoder(body) if body is not None else None
    except ValueError:
        request = None
    if request is None:
        await respond(400, _MALFORMED)
        return
    if exceeds_cap is not None and exceeds_cap(request):
        await failure(GatewayFailureReason.RESULT_CAP)
        return
    deadline = min(request.deadline, monotonic() + settings.timeout_ms / 1000.0)
    if deadline <= monotonic():
        await failure(GatewayFailureReason.DEADLINE)
        return
    return settings, request, deadline


def _family_cap_exceeded(request):
    return request.max_records > _FAMILY_CAPS[request.query]


async def topology_read(scope, receive, send) -> None:
    validated = await _validate_ingress(
        scope,
        receive,
        send,
        decoder=decode_request,
        version=SCHEMA_VERSION,
        checksum=SCHEMA_CHECKSUM,
        exceeds_cap=_family_cap_exceeded,
    )
    if validated is None:
        return
    settings, request, deadline = validated
    try:
        runtime = _get_runtime(settings)
        rows = await GATEWAY_WORKERS.run(
            runtime.adapter.execute_read,
            expires=deadline,
            clock=monotonic,
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
    except (OverflowError, TopologyResultCapError):
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
    elif route == ("POST", "/v2/topology/snapshot"):
        from .gateway_snapshot_service import topology_snapshot

        await topology_snapshot(scope, receive, send)
    elif route == ("POST", "/v2/topology/frontier"):
        from .frontier_gateway_service_v2 import topology_frontier

        await topology_frontier(scope, receive, send)
    elif route == ("POST", "/v2/topology/frontier/batch"):
        from .frontier_gateway_service_v2 import topology_frontier

        await topology_frontier(scope, receive, send, batch=True)
    elif route == ("POST", "/v2/topology/frontier/batch-binary"):
        from .frontier_gateway_service_v2 import topology_frontier

        await topology_frontier(scope, receive, send, binary=True)
    elif route == ("POST", "/v2/topology/frontier/window"):
        from .frontier_gateway_service_v2 import topology_frontier

        await topology_frontier(scope, receive, send, window=True)
    elif route == ("POST", "/v3/topology/frontier/scope-window"):
        from .frontier_gateway_service_v2 import topology_frontier

        await topology_frontier(scope, receive, send, complete_scope=True)
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
