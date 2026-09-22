"""Exact scalar and normalized-key validation for canonical resolution."""

from __future__ import annotations

import json
import re
from hashlib import sha256

from .normalization import normalize_entity_label

_ACRONYM_PATTERN = re.compile(r"[A-Z][A-Z0-9-]{1,11}")
def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or not 1 <= value <= 2**63 - 1:
        raise ValueError(f"{label} must be a positive database integer")
    return value


def _bounded_text(
    value: object,
    label: str,
    *,
    maximum: int,
    allow_empty: bool = False,
) -> str:
    if type(value) is not str or value != value.strip() or "\x00" in value:
        raise ValueError(f"{label} must be an exact trimmed string")
    if (not value and not allow_empty) or len(value) > maximum:
        emptiness = "possibly empty" if allow_empty else "nonempty"
        raise ValueError(f"{label} must be a bounded {emptiness} string")
    return value


def _hash_payload(payload: object) -> str:
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _normalized_key(value: str) -> str:
    return normalize_entity_label(value).key


def _is_acronym(value: str) -> bool:
    return bool(_ACRONYM_PATTERN.fullmatch(value))
