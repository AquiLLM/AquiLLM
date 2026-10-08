"""One admitted worker owns attestation, hydration and final V2 wire encoding."""

from hmac import compare_digest

from . import gateway_service as v1
from . import gateway_snapshot_contracts as v2
from .contracts import TopologyFailureReason
from .failures import TopologyLoadError, TopologyResultCapError
from .gateway_contracts import GatewayFailureReason, TopologyGatewayFailureV1


async def _respond(send, status, body):
    await v1._respond(
        send, status, body, version=v2.SCHEMA_VERSION, checksum=v2.SCHEMA_CHECKSUM
    )


async def _failure(send, reason):
    failure = TopologyGatewayFailureV1(reason)
    await _respond(send, failure.status, v2.encode_response(failure))


def _snapshot_payload(adapter, *, parameters, deadline, maximum):
    result = adapter.execute_snapshot(parameters=parameters, deadline=deadline)
    if type(result) is not v2.TopologyGatewaySnapshotV2:
        raise TopologyLoadError(TopologyFailureReason.BACKEND_SCHEMA_MISMATCH)
    # Only envelope size can select split delivery. An adapter/source/final DTO
    # error above must propagate without selecting an alternative transport.
    try:
        payload = v2.encode_response(result)
        if len(payload) > maximum:
            raise OverflowError
    except OverflowError:
        payload = v2.encode_response(
            v2.TopologyGatewayFamilyTransportV2(result.manifests)
        )
        if len(payload) > maximum:
            raise OverflowError
    if v1.monotonic() >= deadline:
        raise TimeoutError("snapshot deadline expired")
    return payload


async def topology_snapshot(scope, receive, send):
    try:
        settings = v1.load_topology_gateway_settings(v1.environ)
    except Exception:
        await _failure(send, GatewayFailureReason.UNAVAILABLE)
        return
    expected = b"Bearer " + settings.bearer_token.get_secret_value().encode("ascii")
    if not compare_digest(v1._single_header(scope, b"authorization") or b"", expected):
        await _failure(send, GatewayFailureReason.AUTHENTICATION)
        return
    length = v1._length(scope, settings.max_request_bytes)
    if length is not None and length > settings.max_request_bytes:
        await _respond(send, 413, v1._OVERSIZED)
        return
    if length is None or not v1._wire_valid(
        scope, version=v2.SCHEMA_VERSION, checksum=v2.SCHEMA_CHECKSUM
    ):
        await _respond(send, 400, v1._MALFORMED)
        return
    body = await v1._read_body(receive, length)
    try:
        request = v2.decode_request(body) if body is not None else None
    except ValueError:
        request = None
    if request is None:
        await _respond(send, 400, v1._MALFORMED)
        return
    deadline = min(request.deadline, v1.monotonic() + settings.timeout_ms / 1000.0)
    if deadline <= v1.monotonic():
        await _failure(send, GatewayFailureReason.DEADLINE)
        return
    try:
        runtime = v1._get_runtime(settings)
        payload = await v1.GATEWAY_WORKERS.run(
            _snapshot_payload,
            runtime.adapter,
            expires=deadline,
            clock=v1.monotonic,
            parameters=request.parameters,
            deadline=deadline,
            maximum=settings.max_response_bytes,
        )
        if v1.monotonic() >= deadline:
            raise TimeoutError
    except (OverflowError, TopologyResultCapError):
        await _failure(send, GatewayFailureReason.RESULT_CAP)
        return
    except TimeoutError:
        await _failure(send, GatewayFailureReason.DEADLINE)
        return
    except TopologyLoadError as error:
        await _failure(send, v1._mapped_reason(error))
        return
    except Exception:
        await _failure(send, GatewayFailureReason.UNAVAILABLE)
        return
    await _respond(send, 200, payload)
