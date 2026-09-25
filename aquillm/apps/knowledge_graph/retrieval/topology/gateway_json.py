"""Strict canonical JSON loading for topology gateway payloads."""

from __future__ import annotations

import json
from collections.abc import Callable

INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    if len({key for key, _ in pairs}) != len(pairs):
        raise ValueError("duplicate JSON object key")
    return dict(pairs)


def load_gateway_json(
    payload: bytes, maximum: int, canonical: Callable[[object], bytes]
) -> object:
    if type(payload) is not bytes:
        raise ValueError("gateway payload must be bytes")
    if len(payload) > maximum:
        raise ValueError("gateway payload exceeds its byte cap")

    def _parse_int(value: str) -> int:
        if len(value) > 19 + value.startswith("-"):
            raise ValueError("signed 64-bit integer has too many digits")
        parsed = int(value)
        if INT64_MIN <= parsed <= INT64_MAX:
            return parsed
        raise ValueError("integer exceeds signed 64-bit range")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=int,
            parse_int=_parse_int,
        )
        if canonical(value) != payload:
            raise ValueError("gateway JSON is not canonical")
        return value
    except (RecursionError, ValueError, UnicodeError):
        raise ValueError("malformed gateway payload") from None
