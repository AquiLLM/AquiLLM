"""One admitted worker owns attestation, hydration and final V2 wire encoding."""

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
    validated = await v1._validate_ingress(
        scope,
        receive,
        send,
        decoder=v2.decode_request,
        version=v2.SCHEMA_VERSION,
        checksum=v2.SCHEMA_CHECKSUM,
    )
    if validated is None:
        return
    settings, request, deadline = validated
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
