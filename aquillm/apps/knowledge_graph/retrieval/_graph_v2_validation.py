"""Exact, bounded scalar and canonical hashing helpers for private V2 DTOs."""

from __future__ import annotations

import json
import re
from dataclasses import fields, is_dataclass
from hashlib import sha256
from math import isfinite
from uuid import UUID

RELATION_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,127}")


def checksum(value, name):
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")


def relation(value):
    if type(value) is not str:
        raise TypeError("relation must be an exact string")
    if RELATION_PATTERN.fullmatch(value) is None:
        raise ValueError("relation must be an ontology token")


def count(value, name, maximum):
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if not 0 <= value <= maximum:
        raise ValueError(f"{name} exceeds its bound")


def score(value, name, *, maximum=None):
    if type(value) is not float:
        raise TypeError(f"{name} must be an exact float")
    if not isfinite(value) or value < 0.0 or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be finite and within its bound")


def ordered_strings(value, name, validator, maximum, *, required=False):
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an exact tuple")
    if len(value) > maximum or (required and not value):
        raise ValueError(f"{name} violates its bound")
    for item in value:
        validator(item)
    if value != tuple(sorted(set(value))):
        raise ValueError(f"{name} must be unique and canonically sorted")


def reference_keys(value):
    for name in ("chunk_key", "graph_generation_key", "authority_group_checksum"):
        checksum(getattr(value, name), name)
    generation = value.representation_generation_key
    if generation is not None and type(generation) is not UUID:
        raise TypeError("representation_generation_key must be exact UUID or None")
    ordered_strings(
        value.evidence_provenance_keys,
        "evidence_provenance_keys",
        lambda k: checksum(k, "evidence key"),
        3000,
        required=True,
    )
    ordered_strings(value.relation_types, "relation_types", relation, 128)


def canonical_bytes(domain, value):
    def encode(item):
        if is_dataclass(item):
            return {f.name: encode(getattr(item, f.name)) for f in fields(item)}
        if type(item) is tuple:
            return [encode(v) for v in item]
        if type(item) is UUID:
            return str(item)
        if type(item) is float:
            return item.hex()
        if item is None or type(item) in (str, int, bool):
            return item
        raise TypeError("unsupported canonical V2 value")

    return json.dumps(
        {"domain": domain, "value": encode(value)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def digest(domain, value):
    return sha256(canonical_bytes(domain, value)).hexdigest()
