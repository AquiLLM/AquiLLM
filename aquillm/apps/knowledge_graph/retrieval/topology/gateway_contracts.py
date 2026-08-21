"""Closed, provider-neutral wire contract for the Community topology gateway."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from math import isfinite
from types import MappingProxyType
from typing import Final

from .contracts import TopologyQueryName, TopologyScalar

MAX_RESULT_ROWS: Final = 5_000
MAX_REQUEST_BYTES: Final = 16_384
MAX_RESPONSE_BYTES: Final = 1_048_576
SCHEMA_VERSION: Final = "topology-gateway-v1"
MALFORMED_REQUEST_STATUS: Final = 400
OVERSIZED_REQUEST_STATUS: Final = 413


class GatewayFailureReason(StrEnum):
    AUTHENTICATION = "authentication"
    UNAVAILABLE = "unavailable"
    SCHEMA = "schema"
    PROVENANCE = "provenance"
    DEADLINE = "deadline"
    RESULT_CAP = "result_cap"


FAILURE_HTTP_STATUS: Final = MappingProxyType(
    {
        GatewayFailureReason.AUTHENTICATION: 401,
        GatewayFailureReason.UNAVAILABLE: 503,
        GatewayFailureReason.SCHEMA: 502,
        GatewayFailureReason.PROVENANCE: 409,
        GatewayFailureReason.DEADLINE: 504,
        GatewayFailureReason.RESULT_CAP: 422,
    }
)


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        raise ValueError("gateway value is not canonical JSON") from None


SCHEMA_DESCRIPTOR_V1: Final = (
    ("version", SCHEMA_VERSION),
    ("query_names", tuple(query.value for query in TopologyQueryName)),
    (
        "scalar_rules",
        (
            "exact builtins only: str, int, float, bool, null; "
            "float finite; strings forbid C0/DEL",
        ),
    ),
    (
        "request_shape",
        (
            "exact fields: query, parameters, deadline, max_records",
            "query is TopologyQueryName; parameters is Mapping[str, TopologyScalar]",
            "deadline is exact finite positive absolute monotonic float",
            "max_records is exact int in [1, 5000]",
        ),
    ),
    (
        "response_shapes",
        (
            "success_shape: ok=true; rows=list[Mapping[str, TopologyScalar]]",
            "failure_shape: ok=false; reason=enum; status=fixed HTTP status",
        ),
    ),
    (
        "failure_http_status",
        tuple((r.value, FAILURE_HTTP_STATUS[r]) for r in GatewayFailureReason),
    ),
    (
        "canonical_json_rules",
        (
            "UTF-8; ensure_ascii=false; allow_nan=false; sort_keys=true",
            "separators=(',', ':'); duplicate keys rejected; bytes canonical",
        ),
    ),
    ("malformed_request_status", MALFORMED_REQUEST_STATUS),
    ("oversized_request_status", OVERSIZED_REQUEST_STATUS),
    ("max_result_rows", MAX_RESULT_ROWS),
    ("max_request_bytes", MAX_REQUEST_BYTES),
    ("max_response_bytes", MAX_RESPONSE_BYTES),
)
SCHEMA_CHECKSUM: Final = sha256(_canonical(SCHEMA_DESCRIPTOR_V1)).hexdigest()


def _safe_text(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise ValueError(f"{name} contains forbidden control text")
    return value


def _scalar(value: object, name: str) -> None:
    if value is None or type(value) is bool or type(value) is int or type(value) is str:
        if type(value) is str:
            _safe_text(value, name)
        return
    if type(value) is float and isfinite(value):
        return
    raise TypeError(f"{name} must be an exact topology scalar")


def _mapping(value: Mapping[str, TopologyScalar], name: str) -> MappingProxyType:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    try:
        snapshot = [tuple(pair) for pair in value.items()]
    except Exception:
        raise ValueError("invalid topology mapping") from None
    copied: dict[str, TopologyScalar] = {}
    for key, item in snapshot:
        _safe_text(key, f"{name} key")
        _scalar(item, f"{name} value")
        if key in copied:
            raise ValueError(f"{name} contains duplicate keys")
        copied[key] = item
    return MappingProxyType(copied)


@dataclass(frozen=True, slots=True)
class TopologyGatewayRequestV1:
    query: TopologyQueryName
    parameters: Mapping[str, TopologyScalar] = field(repr=False)
    deadline: float
    max_records: int

    def __post_init__(self) -> None:
        if type(self.query) is not TopologyQueryName:
            raise TypeError("query must be an exact TopologyQueryName")
        object.__setattr__(self, "parameters", _mapping(self.parameters, "parameters"))
        if (
            type(self.deadline) is not float
            or not isfinite(self.deadline)
            or self.deadline <= 0
        ):
            raise ValueError("deadline must be a positive finite monotonic float")
        if (
            type(self.max_records) is not int
            or not 1 <= self.max_records <= MAX_RESULT_ROWS
        ):
            raise ValueError("max_records exceeds the result cap")


@dataclass(frozen=True, slots=True)
class TopologyGatewaySuccessV1:
    rows: tuple[Mapping[str, TopologyScalar], ...] = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.rows) is not tuple or len(self.rows) > MAX_RESULT_ROWS:
            raise ValueError("rows exceed the result cap")
        object.__setattr__(
            self,
            "rows",
            tuple(_mapping(row, "row") for row in self.rows),
        )


@dataclass(frozen=True, slots=True)
class TopologyGatewayFailureV1:
    reason: GatewayFailureReason
    status: int = field(init=False)

    def __post_init__(self) -> None:
        if type(self.reason) is not GatewayFailureReason:
            raise TypeError("reason must be an exact GatewayFailureReason")
        object.__setattr__(self, "status", FAILURE_HTTP_STATUS[self.reason])


type TopologyGatewayResponseV1 = TopologyGatewaySuccessV1 | TopologyGatewayFailureV1


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _load(payload: bytes, maximum: int) -> object:
    if type(payload) is not bytes:
        raise ValueError("gateway payload must be bytes")
    if len(payload) > maximum:
        raise ValueError("gateway payload exceeds its byte cap")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ValueError("non-finite JSON number")
            ),
        )
        if _canonical(value) != payload:
            raise ValueError("gateway JSON is not canonical")
        return value
    except (ValueError, UnicodeError, json.JSONDecodeError):
        raise ValueError("malformed gateway payload") from None


def encode_request(value: TopologyGatewayRequestV1) -> bytes:
    if type(value) is not TopologyGatewayRequestV1:
        raise ValueError("invalid gateway request")
    payload = {
        "deadline": value.deadline,
        "max_records": value.max_records,
        "parameters": dict(value.parameters),
        "query": value.query.value,
    }
    encoded = _canonical(payload)
    if len(encoded) > MAX_REQUEST_BYTES:
        raise ValueError("gateway request exceeds its byte cap")
    return encoded


def decode_request(payload: bytes) -> TopologyGatewayRequestV1:
    value = _load(payload, MAX_REQUEST_BYTES)
    if type(value) is not dict or set(value) != {
        "query",
        "parameters",
        "deadline",
        "max_records",
    }:
        raise ValueError("gateway request schema mismatch")
    try:
        request = TopologyGatewayRequestV1(
            query=TopologyQueryName(value["query"]),
            parameters=value["parameters"],
            deadline=value["deadline"],
            max_records=value["max_records"],
        )
        if len(encode_request(request)) != len(payload):
            raise ValueError("gateway request is not canonical")
        return request
    except (TypeError, ValueError, KeyError):
        raise ValueError("invalid gateway request") from None


def encode_response(value: TopologyGatewayResponseV1) -> bytes:
    if type(value) is TopologyGatewaySuccessV1:
        payload: object = {"ok": True, "rows": [dict(row) for row in value.rows]}
    elif type(value) is TopologyGatewayFailureV1:
        payload = {"ok": False, "reason": value.reason.value, "status": value.status}
    else:
        raise ValueError("invalid gateway response")
    encoded = _canonical(payload)
    if len(encoded) > MAX_RESPONSE_BYTES:
        raise ValueError("gateway response exceeds its byte cap")
    return encoded


def decode_response(payload: bytes) -> TopologyGatewayResponseV1:
    value = _load(payload, MAX_RESPONSE_BYTES)
    if type(value) is not dict or type(value.get("ok")) is not bool:
        raise ValueError("invalid gateway response")
    try:
        if value["ok"] is True:
            if set(value) != {"ok", "rows"} or type(value["rows"]) is not list:
                raise ValueError
            response: TopologyGatewayResponseV1 = TopologyGatewaySuccessV1(
                tuple(value["rows"])
            )
        else:
            if set(value) != {"ok", "reason", "status"}:
                raise ValueError
            response = TopologyGatewayFailureV1(GatewayFailureReason(value["reason"]))
            if value["status"] != response.status or type(value["status"]) is not int:
                raise ValueError
        if encode_response(response) != payload:
            raise ValueError
        return response
    except (TypeError, ValueError, KeyError):
        raise ValueError("invalid gateway response") from None


def http_status_for_failure(reason: GatewayFailureReason) -> int:
    if type(reason) is not GatewayFailureReason:
        raise ValueError("invalid gateway failure reason")
    return FAILURE_HTTP_STATUS[reason]
