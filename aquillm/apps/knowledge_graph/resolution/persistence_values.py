"""Bounded row and canonical value helpers for document persistence."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from itertools import islice
from math import isfinite
from uuid import UUID


_QUERY_ITERATOR_BATCH_SIZE = 1_000


def _record_value(record: object, field: str) -> object:
    if isinstance(record, Mapping):
        return record.get(field)
    return getattr(record, field, None)


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("mention fingerprint values must be finite")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("mention metadata keys must be strings")
        return {key: _json_value(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    raise ValueError(f"unsupported mention fingerprint value: {type(value).__name__}")


def _bounded_rows(values, maximum: int, label: str) -> tuple[object, ...]:
    if type(maximum) is not int or maximum < 1:
        raise ValueError("resolution row cap must be a positive integer")
    query_count = getattr(values, "count", None)
    query_iterator = getattr(values, "iterator", None)
    if callable(query_count) and callable(query_iterator):
        count = query_count()
        if type(count) is not int or count < 0:
            raise ValueError(f"{label} count is invalid")
        if count > maximum:
            raise ValueError(f"{label} cap exceeded ({maximum})")
        records = tuple(
            islice(
                query_iterator(chunk_size=min(maximum, _QUERY_ITERATOR_BATCH_SIZE)),
                maximum + 1,
            )
        )
        if len(records) > maximum:
            raise ValueError(f"{label} cap exceeded ({maximum})")
        return records
    records = tuple(islice(iter(values), maximum + 1))
    if len(records) > maximum:
        raise ValueError(f"{label} cap exceeded ({maximum})")
    return records
