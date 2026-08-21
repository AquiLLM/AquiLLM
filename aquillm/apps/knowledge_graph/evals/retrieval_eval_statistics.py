"""Small deterministic statistics helpers for retrieval evaluation."""

from __future__ import annotations

from collections.abc import Sequence


def mean(values: Sequence[float]) -> float:
    return 0.0 if not values else sum(values) / len(values)


def percentile_95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, (95 * len(ordered) + 99) // 100 - 1)
    return ordered[index]


__all__ = ["mean", "percentile_95"]
