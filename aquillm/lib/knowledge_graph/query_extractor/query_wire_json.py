"""Bounded canonical JSON decoding for query extraction wire payloads."""

from __future__ import annotations

import json
from collections.abc import Callable


def parse_canonical_payload(
    data: bytes,
    fields: frozenset[str],
    maximum: int,
    *,
    optional: frozenset[str] = frozenset(),
    canonical: Callable[[object], bytes],
) -> dict[str, object]:
    if type(data) is not bytes:
        raise TypeError("wire data must be exact bytes")
    if len(data) > maximum:
        raise ValueError("wire data exceeds its byte cap")
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("wire data must be valid JSON") from error
    if type(value) is not dict or not fields <= set(value) <= fields | optional:
        raise ValueError("wire object has an invalid field set")
    try:
        canonical = canonical(value)
    except UnicodeEncodeError as error:
        raise ValueError("wire object must contain valid UTF-8") from error
    if data != canonical:
        raise ValueError("wire object must use canonical JSON")
    return value
