"""Explicit V2 selection, sharing V1's bounded zero-retry HTTP boundary."""

from . import gateway_snapshot_contracts as v2
from .gateway_client import TopologyGatewayRequestError, _failure
from .gateway_contracts import (
    GatewayFailureReason,
    GatewayRequestSizeError,
    TopologyGatewayFailureV1,
)


def execute_snapshot(client, *, parameters, deadline):
    try:
        body = v2.encode_request(v2.TopologyGatewayRequestV2(parameters, deadline))
    except GatewayRequestSizeError:
        raise TopologyGatewayRequestError(GatewayFailureReason.RESULT_CAP) from None
    result = client._exchange(body, deadline=deadline, snapshot=True)
    if type(result) is TopologyGatewayFailureV1:
        _failure(result.reason)
    return result
