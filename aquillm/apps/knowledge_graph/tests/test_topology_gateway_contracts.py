from __future__ import annotations

import dataclasses
import json

import pytest

from apps.knowledge_graph.retrieval.topology.contracts import TopologyQueryName
from apps.knowledge_graph.retrieval.topology.gateway_contracts import (
    FAILURE_HTTP_STATUS,
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    MAX_RESULT_ROWS,
    GatewayFailureReason,
    TopologyGatewayFailureV1,
    TopologyGatewayRequestV1,
    TopologyGatewaySuccessV1,
    decode_request,
    decode_response,
    encode_request,
    encode_response,
)


def request() -> TopologyGatewayRequestV1:
    return TopologyGatewayRequestV1(
        query=TopologyQueryName.RELATION_TOPOLOGY,
        parameters={"collection": "c-1", "limit": 2},
        deadline=123.5,
        max_records=2,
    )


def test_request_is_frozen_slotted_and_has_exact_four_fields():
    assert dataclasses.fields(TopologyGatewayRequestV1)
    assert getattr(TopologyGatewayRequestV1, "__slots__")
    assert tuple(field.name for field in dataclasses.fields(request())) == (
        "query",
        "parameters",
        "deadline",
        "max_records",
    )
    assert encode_request(request()) == (
        b'{"deadline":123.5,"max_records":2,"parameters":{"collection":"c-1","limit":2},"query":"relation_topology"}'
    )


def test_request_round_trip_and_schema_rejects_unknown_or_raw_query_fields():
    assert decode_request(encode_request(request())) == request()
    for payload in (
        {
            "query": "relation_topology",
            "parameters": {},
            "deadline": 1.0,
            "max_records": 1,
            "cypher": "MATCH (n) RETURN n",
        },
        {"query": "not_a_query", "parameters": {}, "deadline": 1.0, "max_records": 1},
    ):
        with pytest.raises(ValueError):
            decode_request(json.dumps(payload, separators=(",", ":")).encode())


@pytest.mark.parametrize(
    "payload",
    [
        b'{"max_records":1,"deadline":1.0,"parameters":{},"query":"relation_topology"}',
        b'{ "deadline":1.0,"max_records":1,'
        b'"parameters":{},"query":"relation_topology"}',
        b'{"deadline":1.0,"max_records":1,"parameters":{},"query":"relation_topology","query":"relation_topology"}',
        b'{"deadline":1.0,"max_records":1,"parameters":{"x":"a\\u0000b"},"query":"relation_topology"}',
        b'{"deadline":1.0,"max_records":1,"parameters":{"x":NaN},"query":"relation_topology"}',
    ],
)
def test_request_decoder_accepts_only_canonical_safe_json(payload: bytes):
    with pytest.raises(ValueError):
        decode_request(payload)


def test_response_union_has_only_success_rows_or_failure_reason_and_status():
    success = TopologyGatewaySuccessV1(rows=({"id": "n1", "weight": 1.0},))
    failure = TopologyGatewayFailureV1(GatewayFailureReason.PROVENANCE)
    assert decode_response(encode_response(success)) == success
    assert decode_response(encode_response(failure)) == failure
    assert json.loads(encode_response(success)) == {
        "ok": True,
        "rows": [{"id": "n1", "weight": 1.0}],
    }
    assert json.loads(encode_response(failure)) == {
        "ok": False,
        "reason": "provenance",
        "status": 409,
    }


def test_failure_status_mapping_is_closed_and_fixed():
    assert FAILURE_HTTP_STATUS == {
        GatewayFailureReason.AUTHENTICATION: 401,
        GatewayFailureReason.UNAVAILABLE: 503,
        GatewayFailureReason.SCHEMA: 502,
        GatewayFailureReason.PROVENANCE: 409,
        GatewayFailureReason.DEADLINE: 504,
        GatewayFailureReason.RESULT_CAP: 422,
    }


def test_limits_and_malformed_bytes_fail_closed_without_echoing_input():
    assert MAX_REQUEST_BYTES < MAX_RESPONSE_BYTES
    with pytest.raises(ValueError) as exc:
        decode_request(b"x" * (MAX_REQUEST_BYTES + 1))
    assert "x" * 50 not in str(exc.value)
    with pytest.raises(ValueError):
        decode_response(b"x" * (MAX_RESPONSE_BYTES + 1))
    with pytest.raises(ValueError):
        TopologyGatewaySuccessV1(
            rows=tuple({"n": i} for i in range(MAX_RESULT_ROWS + 1))
        )
