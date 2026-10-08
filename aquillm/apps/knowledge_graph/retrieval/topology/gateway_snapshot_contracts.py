"""Closed V2 snapshot transport; V1 remains an independently pinned protocol."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from math import isfinite
from typing import Final

from .contracts import TopologyScalar
from .gateway_contracts import (
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    GatewayRequestSizeError,
    TopologyGatewayFailureV1,
    _canonical,
    _load,
    _mapping,
    _safe_text,
)
from .gateway_contracts import (
    SCHEMA_CHECKSUM as V1_CHECKSUM,
)
from .gateway_contracts import (
    decode_response as decode_failure,
)
from .gateway_contracts import (
    encode_response as encode_failure,
)

SCHEMA_VERSION: Final = "topology-gateway-v2"
SNAPSHOT_PATH: Final = "/v2/topology/snapshot"
MAX_SNAPSHOT_BYTES: Final = 2_000_000
MAX_MANIFESTS: Final = 128
_MANIFEST_FIELDS = frozenset(
    {
        "collection_key",
        "generation_key",
        "projection_key",
        "active_artifact_key",
        "graph_checksum",
        "membership_checksum",
    }
)
SCHEMA_DESCRIPTOR_V2: Final = (
    ("version", SCHEMA_VERSION),
    ("route", ("POST", SNAPSHOT_PATH)),
    ("scalar_canonical_and_failure_rules", V1_CHECKSUM),
    ("request_fields", ("deadline", "parameters")),
    ("deadline", "exact positive finite absolute monotonic float"),
    ("parameters", "bounded immutable scalar mapping, same as V1"),
    ("snapshot_fields", ("delivery", "manifests", "ok", "snapshot_json")),
    ("snapshot_discriminator", (True, "snapshot")),
    ("family_fields", ("delivery", "manifests", "ok")),
    ("family_discriminator", (True, "use_family_transport")),
    ("family_selection", "only valid complete snapshot exceeding encoded wire cap"),
    ("family_continuation", "original three families, same parameters and deadline"),
    ("manifests", ("fresh each request", "exact tuple", 1, MAX_MANIFESTS)),
    ("manifest_fields", tuple(sorted(_MANIFEST_FIELDS))),
    ("manifest_values", "exact bounded strings"),
    ("snapshot_json", "exact full canonical projected snapshot UTF-8 string"),
    ("max_snapshot_bytes", MAX_SNAPSHOT_BYTES),
    ("max_request_bytes", MAX_REQUEST_BYTES),
    ("max_response_bytes", MAX_RESPONSE_BYTES),
)
SCHEMA_CHECKSUM: Final = sha256(_canonical(SCHEMA_DESCRIPTOR_V2)).hexdigest()


@dataclass(frozen=True, slots=True)
class TopologyGatewayRequestV2:
    parameters: Mapping[str, TopologyScalar] = field(repr=False)
    deadline: float

    def __post_init__(self):
        if (
            type(self.deadline) is not float
            or not isfinite(self.deadline)
            or self.deadline <= 0
        ):
            raise ValueError("deadline must be a positive finite monotonic float")
        object.__setattr__(
            self,
            "parameters",
            _mapping(
                self.parameters,
                "parameters",
                MAX_REQUEST_BYTES,
                request=True,
            ),
        )
        encode_request(self)


def _manifests(rows):
    if type(rows) is not tuple or not 1 <= len(rows) <= MAX_MANIFESTS:
        raise ValueError("snapshot manifests exceed their cap")
    result = []
    for row in rows:
        mapped = _mapping(row, "manifest", MAX_RESPONSE_BYTES)
        if set(mapped) != _MANIFEST_FIELDS or any(
            type(v) is not str for v in mapped.values()
        ):
            raise ValueError("snapshot manifest schema mismatch")
        result.append(mapped)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class TopologyGatewaySnapshotV2:
    manifests: tuple[Mapping[str, str], ...] = field(repr=False)
    snapshot_json: str = field(repr=False)

    def __post_init__(self):
        object.__setattr__(self, "manifests", _manifests(self.manifests))
        _safe_text(
            self.snapshot_json,
            "snapshot_json",
            MAX_SNAPSHOT_BYTES,
            size_error=OverflowError,
        )
        if len(self.snapshot_json.encode("utf-8")) > MAX_SNAPSHOT_BYTES:
            raise OverflowError("snapshot exceeds its byte cap")


@dataclass(frozen=True, slots=True)
class TopologyGatewayFamilyTransportV2:
    manifests: tuple[Mapping[str, str], ...] = field(repr=False)

    def __post_init__(self):
        object.__setattr__(self, "manifests", _manifests(self.manifests))


def encode_request(value: TopologyGatewayRequestV2) -> bytes:
    if type(value) is not TopologyGatewayRequestV2:
        raise ValueError("invalid snapshot request")
    encoded = _canonical(
        {"parameters": dict(value.parameters), "deadline": value.deadline}
    )
    if len(encoded) > MAX_REQUEST_BYTES:
        raise GatewayRequestSizeError("gateway request exceeds its byte cap")
    return encoded


def decode_request(payload: bytes) -> TopologyGatewayRequestV2:
    value = _load(payload, MAX_REQUEST_BYTES)
    if type(value) is not dict or set(value) != {"parameters", "deadline"}:
        raise ValueError("snapshot request schema mismatch")
    try:
        return TopologyGatewayRequestV2(value["parameters"], value["deadline"])
    except (TypeError, ValueError):
        raise ValueError("invalid snapshot request") from None


def encode_response(value) -> bytes:
    if type(value) is TopologyGatewayFailureV1:
        return encode_failure(value)
    if type(value) not in {TopologyGatewaySnapshotV2, TopologyGatewayFamilyTransportV2}:
        raise ValueError("invalid snapshot response")
    payload = {"ok": True, "manifests": [dict(row) for row in value.manifests]}
    if type(value) is TopologyGatewaySnapshotV2:
        payload.update(delivery="snapshot", snapshot_json=value.snapshot_json)
    else:
        payload.update(delivery="use_family_transport")
    encoded = _canonical(payload)
    if len(encoded) > MAX_RESPONSE_BYTES:
        raise OverflowError("gateway response exceeds its byte cap")
    return encoded


def decode_response(payload: bytes):
    value = _load(payload, MAX_RESPONSE_BYTES)
    if type(value) is not dict or type(value.get("ok")) is not bool:
        raise ValueError("invalid snapshot response")
    if value["ok"] is False:
        return decode_failure(payload)
    try:
        if type(value.get("manifests")) is not list:
            raise ValueError
        if (
            set(value) == {"ok", "delivery", "manifests", "snapshot_json"}
            and value["delivery"] == "snapshot"
        ):
            return TopologyGatewaySnapshotV2(
                tuple(value["manifests"]), value["snapshot_json"]
            )
        if (
            set(value) == {"ok", "delivery", "manifests"}
            and value["delivery"] == "use_family_transport"
        ):
            return TopologyGatewayFamilyTransportV2(tuple(value["manifests"]))
    except (TypeError, ValueError, OverflowError):
        pass
    raise ValueError("invalid snapshot response")
