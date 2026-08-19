"""Small deterministic query-envelope helpers for fixture ownership checks."""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable

from .fixture_seed_contract import FixtureSeedError


def bounded_rows(queryset, expected_count: int, *, order_by=("pk",)) -> list:
    rows = list(queryset.order_by(*order_by)[: expected_count + 1])
    if len(rows) > expected_count:
        raise FixtureSeedError("fixture database topology has surplus rows")
    return rows


def require_exact_unique_rows(
    rows: Iterable,
    expected_keys: Collection,
    *,
    key: Callable,
) -> dict:
    materialized = list(rows)
    keys = [key(row) for row in materialized]
    if len(materialized) != len(expected_keys) or len(set(keys)) != len(keys):
        raise FixtureSeedError("fixture database topology is not exact")
    if set(keys) != set(expected_keys):
        raise FixtureSeedError("fixture database topology is not exact")
    return dict(zip(keys, materialized, strict=True))
