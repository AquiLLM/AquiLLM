"""Derived content-addressed records, subordinate to projection authority."""

import json
from dataclasses import asdict, dataclass
from hashlib import sha256

VERSION = "canonical-frontier-index-v2"
MARKER_FIELDS = (
    "frontier_version",
    "frontier_root_key",
    "frontier_root_checksum",
    "frontier_record_count",
    "frontier_checksum",
    "validated_frontier_checksum",
)
MAX_INDEX_RECORDS = 2_000_000
MAX_INDEX_RECORD_BYTES = 65_536


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def digest(value):
    return sha256(canonical(value).encode()).hexdigest()


def record_digest(row):
    return digest(asdict(row))


def make_record(generation, kind, payload):
    body = canonical(
        {
            "generation_key": generation,
            "kind": kind,
            "payload": payload,
            "version": VERSION,
        }
    )
    if len(body.encode()) > MAX_INDEX_RECORD_BYTES:
        raise ValueError("frontier index record exceeds its byte bound")
    checksum = sha256(body.encode()).hexdigest()
    return {
        "generation_key": generation,
        "index_key": checksum,
        "version": VERSION,
        "kind": kind,
        "payload_json": body,
        "payload_checksum": checksum,
    }


def decode_record(row, *, expected_key, generation):
    if type(row) is not dict or set(row) != {
        "generation_key",
        "index_key",
        "version",
        "kind",
        "payload_json",
        "payload_checksum",
    }:
        raise ValueError("frontier index row schema mismatch")
    body = row["payload_json"]
    if type(body) is not str or len(body.encode()) > MAX_INDEX_RECORD_BYTES:
        raise ValueError("frontier index row exceeds its bound")
    checksum = sha256(body.encode()).hexdigest()
    if (
        row["generation_key"],
        row["index_key"],
        row["version"],
        row["payload_checksum"],
    ) != (generation, expected_key, VERSION, checksum) or checksum != expected_key:
        raise ValueError("frontier content link changed")
    value = json.loads(body)
    if type(value) is not dict or set(value) != {
        "generation_key",
        "kind",
        "payload",
        "version",
    }:
        raise ValueError("frontier payload schema mismatch")
    if (
        body != canonical(value)
        or value["generation_key"] != generation
        or value["version"] != VERSION
        or value["kind"] != row["kind"]
    ):
        raise ValueError("frontier payload binding changed")
    return value["kind"], value["payload"]


def marker_present(marker):
    present = {name for name in MARKER_FIELDS if name in marker}
    if not present:
        return False
    if present != set(MARKER_FIELDS):
        raise ValueError("partially present frontier attestation")
    if (
        marker["frontier_version"] != VERSION
        or type(marker["frontier_record_count"]) is not int
        or not 1 <= marker["frontier_record_count"] <= MAX_INDEX_RECORDS
    ):
        raise ValueError("unsupported frontier attestation")
    for name in MARKER_FIELDS[1:]:
        if name == "frontier_record_count":
            continue
        value = marker[name]
        if (
            type(value) is not str
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("invalid frontier attestation digest")
    if (
        marker["frontier_root_key"] != marker["frontier_root_checksum"]
        or marker["frontier_checksum"] != marker["validated_frontier_checksum"]
    ):
        raise ValueError("unvalidated frontier attestation")
    return True


@dataclass(frozen=True, slots=True)
class FrontierIndexBuildV2:
    records: tuple[dict, ...]
    marker: dict
