from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator, Mapping
from hashlib import sha256

import pytest

from apps.knowledge_graph.retrieval.topology.contracts import TopologyQueryName
from apps.knowledge_graph.retrieval.topology.gateway_contracts import (
    FAILURE_HTTP_STATUS,
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    MAX_RESULT_ROWS,
    SCHEMA_CHECKSUM,
    SCHEMA_DESCRIPTOR_V1,
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


def test_schema_checksum_is_an_exact_digest_of_an_immutable_complete_descriptor():
    descriptor_bytes = json.dumps(
        SCHEMA_DESCRIPTOR_V1,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert type(SCHEMA_DESCRIPTOR_V1) is tuple
    assert (
        SCHEMA_CHECKSUM
        == "9cde1bee50b59842338fea042311a9db7a0fd53bbb1baf095b7155f28fa40ab1"
    )
    assert sha256(descriptor_bytes).hexdigest() == SCHEMA_CHECKSUM
    assert (
        sha256(
            json.dumps(
                ("mutated", *SCHEMA_DESCRIPTOR_V1), separators=(",", ":")
            ).encode()
        ).hexdigest()
        != SCHEMA_CHECKSUM
    )


def test_payload_bearing_repr_is_fixed_and_never_contains_payload_text():
    request_value = TopologyGatewayRequestV1(
        TopologyQueryName.RELATION_TOPOLOGY,
        {"canary": "secret-request"},
        123.5,
        2,
    )
    success = TopologyGatewaySuccessV1(({"canary": "secret-row"},))
    assert "secret-request" not in repr(request_value)
    assert "canary" not in repr(request_value)
    assert "secret-row" not in repr(success)
    assert "canary" not in repr(success)


class _HostileMapping(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        raise RuntimeError(f"canary-item-{key}")

    def __iter__(self) -> Iterator[str]:
        raise RuntimeError("canary-iteration")

    def __len__(self) -> int:
        raise RuntimeError("canary-length")

    def items(self):
        raise RuntimeError("canary-items")


@pytest.mark.parametrize("mapping", [_HostileMapping()])
def test_hostile_mapping_failures_are_fixed_non_echoing_contract_errors(mapping):
    with pytest.raises((TypeError, ValueError)) as request_error:
        TopologyGatewayRequestV1(TopologyQueryName.RELATION_TOPOLOGY, mapping, 1.0, 1)
    with pytest.raises((TypeError, ValueError)) as row_error:
        TopologyGatewaySuccessV1((mapping,))
    assert "canary" not in str(request_error.value)
    assert "canary" not in str(row_error.value)
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
