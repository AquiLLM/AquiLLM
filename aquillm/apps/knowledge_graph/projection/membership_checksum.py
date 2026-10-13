"""Canonical membership hashing without recursive serialization of scalar rows."""

import json
from collections.abc import Callable, Sequence
from dataclasses import fields
from hashlib import sha256

from .records import AutomaticCanonicalMembershipV1
from .serialization import _encode, _provider_types

_FIELDS = tuple(field.name for field in fields(AutomaticCanonicalMembershipV1))
_SCALARS = frozenset({str, int, bool})
_BATCH_SIZE = 256


def _checkpoint(check: Callable[[], None] | None) -> None:
    if check is not None:
        check()


def _ordered_rows(assignments, check):
    _checkpoint(check)
    if isinstance(assignments, (str, bytes)):
        raise TypeError("assignments must be projection membership rows")
    rows = []
    for item in assignments:
        if len(rows) % _BATCH_SIZE == 0:
            _checkpoint(check)
        rows.append(item)
    _checkpoint(check)
    if any(type(row) is not AutomaticCanonicalMembershipV1 for row in rows):
        raise TypeError("assignments must contain exact membership rows")
    rows.sort(key=lambda row: row.entity_key)
    _checkpoint(check)
    keys = tuple(row.entity_key for row in rows)
    if len(set(keys)) != len(keys):
        raise ValueError("projection record tuple must be unique")
    _checkpoint(check)
    return rows


def _payload(row):
    payload = {name: getattr(row, name) for name in _FIELDS}
    if all(value is None or type(value) in _SCALARS for value in payload.values()):
        return payload
    # Preserve the generic serializer's accepted values and rejection behavior
    # even for a frozen record defensively altered outside its constructor.
    provider_types, _ = _provider_types()
    return _encode(row, provider_types)


def membership_decision_checksum(
    assignments: Sequence[AutomaticCanonicalMembershipV1],
    *,
    check: Callable[[], None] | None = None,
) -> str:
    """Hash the same canonical array; an optional original caller guards batches."""
    rows = _ordered_rows(assignments, check)
    digest = sha256(b"[")
    for start in range(0, len(rows), _BATCH_SIZE):
        _checkpoint(check)
        payload = [_payload(row) for row in rows[start : start + _BATCH_SIZE]]
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        _checkpoint(check)
        if start:
            digest.update(b",")
        digest.update(encoded[1:-1])
    digest.update(b"]")
    _checkpoint(check)
    return digest.hexdigest()
