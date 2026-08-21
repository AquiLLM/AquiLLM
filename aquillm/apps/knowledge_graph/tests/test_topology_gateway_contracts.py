from __future__ import annotations

import dataclasses
import json
import sys
from collections.abc import Iterator, Mapping
from hashlib import sha256

import pytest

from apps.knowledge_graph.retrieval.topology.contracts import TopologyQueryName
from apps.knowledge_graph.retrieval.topology.gateway_contracts import (
    _REQUEST_FIELDS,
    FAILURE_HTTP_STATUS,
    MAX_MAPPING_ITEMS,
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    MAX_RESULT_ROWS,
    SCHEMA_CHECKSUM,
    SCHEMA_DESCRIPTOR_V1,
    GatewayFailureReason,
    TopologyGatewayFailureV1,
    TopologyGatewayRequestV1,
    TopologyGatewaySuccessV1,
    _canonical,
    decode_request,
    decode_response,
    encode_request,
    encode_response,
)


def test_request_is_frozen_slotted_and_has_exact_four_fields():
    assert getattr(TopologyGatewayRequestV1, "__slots__")
    assert {
        field.name for field in dataclasses.fields(TopologyGatewayRequestV1)
    } == _REQUEST_FIELDS
    assert type(_REQUEST_FIELDS) is frozenset


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
        == "b00e76c547740cb37d97fd787bc396691e1c9184b3b9d421913db33b0abe81f9"
    )
    assert sha256(descriptor_bytes).hexdigest() == SCHEMA_CHECKSUM
    assert (
        dict(SCHEMA_DESCRIPTOR_V1)["success_rows_type"]
        == "exact list[Mapping[str, TopologyScalar]]"
    )
    for name in (
        "request_query_type",
        "request_parameters_type",
        "success_rows_type",
        "deadline_rule",
        "max_records_rule",
        "success_discriminator",
        "failure_discriminator",
        "failure_reason_type",
        "failure_status_type",
        "input_bytes_rule",
    ):
        mutated = tuple(
            (entry_name, () if entry_name == name else value)
            for entry_name, value in SCHEMA_DESCRIPTOR_V1
        )
        assert sha256(_canonical(mutated)).hexdigest() != SCHEMA_CHECKSUM


def test_payload_bearing_repr_is_fixed_and_never_contains_payload_text():
    request_value = TopologyGatewayRequestV1(
        TopologyQueryName.RELATION_TOPOLOGY,
        {"canary": "secret-request"},
        123.5,
        2,
    )
    success = TopologyGatewaySuccessV1(({"canary": "secret-row"},))
    for value, secret in ((request_value, "secret-request"), (success, "secret-row")):
        assert secret not in repr(value)
        assert "canary" not in repr(value)


class _TypedHostileMapping(Mapping[str, object]):
    def __init__(self, mode: str, error: type[Exception]):
        self.mode = mode
        self.error = error

    def __getitem__(self, key: str) -> object:
        if self.mode == "lookup":
            raise self.error("canary-lookup")
        return "value"

    def __iter__(self) -> Iterator[str]:
        if self.mode in ("iteration", "pair-error"):
            raise self.error("canary-iteration")
        return iter(("key",))

    def __len__(self) -> int:
        return 1

    def items(self):
        if self.mode == "items":
            raise self.error("canary-items")
        if self.mode == "generator":

            def rows():
                yield ("key", "value")
                raise self.error("canary-generator")

            return rows()
        if self.mode.startswith("arity"):
            return [
                {"arity0": (), "arity1": ("key",), "arity3": ("a", "b", "c")}[self.mode]
            ]
        if self.mode == "pair-error":
            return [self]
        return super().items()


@pytest.mark.parametrize(
    "mode",
    [
        "items",
        "iteration",
        "lookup",
        "generator",
        "arity0",
        "arity1",
        "arity3",
        "pair-error",
    ],
)
@pytest.mark.parametrize("error", [TypeError, ValueError, RuntimeError])
def test_typed_mapping_failures_are_normalized_for_requests_and_rows(mode, error):
    mapping = _TypedHostileMapping(mode, error)
    with pytest.raises(ValueError, match="^invalid topology mapping$"):
        TopologyGatewayRequestV1(TopologyQueryName.RELATION_TOPOLOGY, mapping, 1.0, 1)
    with pytest.raises(ValueError, match="^invalid topology mapping$"):
        TopologyGatewaySuccessV1((mapping,))


def test_deep_json_under_byte_caps_is_a_fixed_decoder_error():
    nested = b"0"
    for _ in range(1_100):
        nested = b"[" + nested + b"]"
    payloads = (
        b'{"deadline":1.0,"max_records":1,"parameters":{"x":'
        + nested
        + b'},"query":"relation_topology"}',
        b'{"ok":true,"rows":[' + nested + b"]}",
    )
    for decoder, payload in zip((decode_request, decode_response), payloads):
        with pytest.raises(ValueError) as error:
            decoder(payload)
        assert "canary" not in str(error.value)


def test_signed_int64_boundaries_and_parse_limits_are_frozen():
    values = (-(2**63), 2**63 - 1)
    for value in values:
        assert (
            decode_request(
                encode_request(
                    TopologyGatewayRequestV1(
                        TopologyQueryName.RELATION_TOPOLOGY, {"x": value}, 1.0, 1
                    )
                )
            ).parameters["x"]
            == value
        )
    for value in (-(2**63) - 1, 2**63):
        with pytest.raises(ValueError):
            TopologyGatewayRequestV1(
                TopologyQueryName.RELATION_TOPOLOGY, {"x": value}, 1.0, 1
            )
    payload = encode_request(
        TopologyGatewayRequestV1(
            TopologyQueryName.RELATION_TOPOLOGY, {"x": values[0]}, 1.0, 1
        )
    )
    original = sys.get_int_max_str_digits()
    try:
        for limit in (640, 4_300):
            sys.set_int_max_str_digits(limit)
            assert decode_request(payload).parameters["x"] == values[0]
    finally:
        sys.set_int_max_str_digits(original)


def test_lone_surrogates_are_rejected_in_request_and_success_scalars():
    with pytest.raises(ValueError):
        TopologyGatewayRequestV1(
            TopologyQueryName.RELATION_TOPOLOGY, {"x": "\ud800"}, 1.0, 1
        )
    with pytest.raises(ValueError):
        TopologyGatewaySuccessV1(({"x": "\ud800"},))


def test_request_and_success_equality_is_canonical_wire_exact_and_unhashable():
    values = (1, 1.0, True, -0.0, 0.0)
    requests = tuple(
        TopologyGatewayRequestV1(
            TopologyQueryName.RELATION_TOPOLOGY, {"x": value}, 1.0, 1
        )
        for value in values
    )
    successes = tuple(TopologyGatewaySuccessV1(({"x": value},)) for value in values)
    assert len({encode_request(value) for value in requests}) == len(values)
    assert len({encode_response(value) for value in successes}) == len(values)
    for value, encoded in zip(requests, map(encode_request, requests)):
        assert value == decode_request(encoded)
        with pytest.raises(TypeError):
            hash(value)
    for valueset in (requests, successes):
        assert all(
            left != right
            for index, left in enumerate(valueset)
            for right in valueset[index + 1 :]
        )


class _ManyMapping(Mapping[str, object]):
    def __init__(self):
        self.consumed = 0

    def __getitem__(self, key: str) -> object:
        return "value"

    def __iter__(self):
        return iter(f"k{index}" for index in range(100_000))

    def __len__(self):
        return 100_000

    def items(self):
        for index in range(100_000):
            self.consumed += 1
            yield f"k{index}", "value"


def test_mapping_and_aggregate_response_caps_fail_before_large_materialization():
    mapping = _ManyMapping()
    with pytest.raises(ValueError):
        TopologyGatewayRequestV1(TopologyQueryName.RELATION_TOPOLOGY, mapping, 1.0, 1)
    assert mapping.consumed <= MAX_MAPPING_ITEMS + 1
    with pytest.raises(ValueError):
        TopologyGatewaySuccessV1(
            tuple({"x": "v" * 300} for _ in range(MAX_RESULT_ROWS))
        )


@pytest.mark.parametrize(
    "payload",
    [
        b'{"max_records":1,"deadline":1.0,"parameters":{},"query":"relation_topology"}',
        b'{ "deadline":1.0,"max_records":1,'
        b'"parameters":{},"query":"relation_topology"}',
        b'{"deadline":1.0,"max_records":1,"parameters":{},"query":"relation_topology","query":"relation_topology"}',
        b'{"deadline":1.0,"max_records":1,"parameters":{"x":"a\\u0000b"},"query":"relation_topology"}',
        b'{"deadline":1.0,"max_records":1,"parameters":{"x":NaN},"query":"relation_topology"}',
        b'{"cypher":"MATCH (n) RETURN n","deadline":1.0,"max_records":1,'
        b'"parameters":{},"query":"relation_topology"}',
        b'{"deadline":1.0,"max_records":1,"parameters":{},"query":"not_a_query"}',
    ],
)
def test_request_decoder_accepts_only_canonical_safe_json(payload: bytes):
    with pytest.raises(ValueError):
        decode_request(payload)


def test_response_union_has_only_success_rows_or_failure_reason_and_status():
    success = TopologyGatewaySuccessV1(rows=({"id": "n1", "weight": 1.0},))
    failure = TopologyGatewayFailureV1(GatewayFailureReason.PROVENANCE)
    for response in (success, failure):
        assert decode_response(encode_response(response)) == response


def test_failure_status_mapping_is_closed_and_fixed():
    assert tuple(FAILURE_HTTP_STATUS.values()) == (401, 503, 502, 409, 504, 422)


def test_limits_and_malformed_bytes_fail_closed_without_echoing_input():
    with pytest.raises(ValueError) as exc:
        decode_request(b"x" * (MAX_REQUEST_BYTES + 1))
    assert "x" * 50 not in str(exc.value)
    with pytest.raises(ValueError):
        decode_response(b"x" * (MAX_RESPONSE_BYTES + 1))
    with pytest.raises(ValueError):
        TopologyGatewaySuccessV1(
            rows=tuple({"n": i} for i in range(MAX_RESULT_ROWS + 1))
        )
