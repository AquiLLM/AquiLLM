from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from itertools import islice
from math import isfinite
from types import MappingProxyType
from typing import Final

from .contracts import TopologyQueryName, TopologyScalar

MAX_RESULT_ROWS: Final = 5_000
MAX_REQUEST_BYTES: Final = 16_384
MAX_RESPONSE_BYTES: Final = 1_048_576
MAX_MAPPING_ITEMS: Final = 512
INT64_MIN: Final = -(2**63)
INT64_MAX: Final = 2**63 - 1
_REQUEST_FIELDS = {"query", "parameters", "deadline", "max_records"}
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
    except (RecursionError, TypeError, ValueError, UnicodeError):
        raise ValueError("gateway value is not canonical JSON") from None


SCHEMA_DESCRIPTOR_V1: Final = (
    ("version", "topology-gateway-v1"),
    ("query_names", tuple(query.value for query in TopologyQueryName)),
    (
        "rules",
        "scalar types/int64/finite/C0-DEL-surrogate; request fields/monotonic; "
        "response canonical closed; JSON UTF8/sorted/compact/duplicate recursion fixed",
    ),
    (
        "failure_http_status",
        tuple((r.value, FAILURE_HTTP_STATUS[r]) for r in GatewayFailureReason),
    ),
    (
        "status_caps",
        (
            MALFORMED_REQUEST_STATUS,
            OVERSIZED_REQUEST_STATUS,
            MAX_RESULT_ROWS,
            MAX_REQUEST_BYTES,
            MAX_RESPONSE_BYTES,
            MAX_MAPPING_ITEMS,
        ),
    ),
    ("int64_domain", (INT64_MIN, INT64_MAX)),
)
SCHEMA_CHECKSUM: Final = sha256(_canonical(SCHEMA_DESCRIPTOR_V1)).hexdigest()


def _safe_text(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    if any(
        ord(char) < 0x20 or ord(char) == 0x7F or 0xD800 <= ord(char) <= 0xDFFF
        for char in value
    ):
        raise ValueError(f"{name} contains forbidden control text")


def _scalar(value: object, name: str) -> None:
    if value is None or type(value) is bool:
        return
    if type(value) is int:
        if INT64_MIN <= value <= INT64_MAX:
            return
        raise ValueError(f"{name} exceeds signed 64-bit range")
    if type(value) is str:
        _safe_text(value, name)
        return
    if type(value) is float and isfinite(value):
        return
    raise TypeError(f"{name} must be an exact topology scalar")


def _request_domains(deadline: float, max_records: int) -> None:
    if type(deadline) is not float or not isfinite(deadline) or deadline <= 0:
        raise ValueError("deadline must be a positive finite monotonic float")
    if type(max_records) is not int or not 1 <= max_records <= MAX_RESULT_ROWS:
        raise ValueError("max_records exceeds the result cap")


def _mapping(value: Mapping[str, TopologyScalar], name: str, limit: int) -> Mapping:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    try:
        snapshot = [
            (key, item) for key, item in islice(value.items(), MAX_MAPPING_ITEMS + 1)
        ]
        if len(snapshot) > MAX_MAPPING_ITEMS:
            raise ValueError("mapping item cap exceeded")
    except Exception:
        raise ValueError("invalid topology mapping") from None
    copied: dict[str, TopologyScalar] = {}
    size = 2
    for key, item in snapshot:
        _safe_text(key, f"{name} key")
        _scalar(item, f"{name} value")
        if key in copied:
            raise ValueError(f"{name} contains duplicate keys")
        copied[key] = item
        size += len(_canonical(key)) + len(_canonical(item)) + 1
        if size > limit:
            raise ValueError("topology mapping exceeds its byte cap")
    return MappingProxyType(copied)


def _wire_equal(value: object, other: object, encoder: object) -> bool:
    return type(other) is type(value) and encoder(value) == encoder(other)


@dataclass(frozen=True, slots=True, eq=False)
class TopologyGatewayRequestV1:
    query: TopologyQueryName
    parameters: Mapping[str, TopologyScalar] = field(repr=False)
    deadline: float
    max_records: int

    def __post_init__(self) -> None:
        if type(self.query) is not TopologyQueryName:
            raise TypeError("query must be an exact TopologyQueryName")
        parameters = _mapping(self.parameters, "parameters", MAX_REQUEST_BYTES)
        object.__setattr__(self, "parameters", parameters)
        _request_domains(self.deadline, self.max_records)
        if len(encode_request(self)) > MAX_REQUEST_BYTES:
            raise ValueError("gateway request exceeds its byte cap")

    def __eq__(self, other: object) -> bool:
        return _wire_equal(self, other, encode_request)

    __hash__ = None


@dataclass(frozen=True, slots=True, eq=False)
class TopologyGatewaySuccessV1:
    rows: tuple[Mapping[str, TopologyScalar], ...] = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.rows) is not tuple or len(self.rows) > MAX_RESULT_ROWS:
            raise ValueError("rows exceed the result cap")
        normalized = []
        total = len(b'{"ok":true,"rows":[]') + 1
        for row in self.rows:
            mapped = _mapping(row, "row", MAX_RESPONSE_BYTES)
            total += len(_canonical(dict(mapped))) + (1 if normalized else 0)
            if total > MAX_RESPONSE_BYTES:
                raise ValueError("gateway response exceeds its byte cap")
            normalized.append(mapped)
        object.__setattr__(self, "rows", tuple(normalized))

    def __eq__(self, other: object) -> bool:
        return _wire_equal(self, other, encode_response)

    __hash__ = None


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
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate JSON object key")
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
    except (RecursionError, ValueError, UnicodeError, json.JSONDecodeError):
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
    if type(value) is not dict or set(value) != _REQUEST_FIELDS:
        raise ValueError("gateway request schema mismatch")
    try:
        request = TopologyGatewayRequestV1(
            TopologyQueryName(value["query"]),
            value["parameters"],
            value["deadline"],
            value["max_records"],
        )
        if len(encode_request(request)) != len(payload):
            raise ValueError("gateway request is not canonical")
        return request
    except (TypeError, ValueError, KeyError):
        raise ValueError("invalid gateway request") from None


def encode_response(value: TopologyGatewayResponseV1) -> bytes:
    if type(value) is TopologyGatewaySuccessV1:
        payload = {"ok": True, "rows": [dict(row) for row in value.rows]}
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
            response = TopologyGatewaySuccessV1(tuple(value["rows"]))
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
