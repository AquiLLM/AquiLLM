"""Small deterministic statistics helpers for retrieval evaluation."""

from __future__ import annotations

import random
from collections.abc import Sequence
from functools import lru_cache
from math import isfinite, sqrt


def mean(values: Sequence[float]) -> float:
    return 0.0 if not values else sum(values) / len(values)


def percentile_95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, (95 * len(ordered) + 99) // 100 - 1)
    return ordered[index]


__all__ = ["mean", "percentile_95"]


def wilson_95(correct: int, total: int):
    """Two-sided 95% Wilson score interval; no observations remain unknown."""
    if type(correct) is not int or type(total) is not int or not 0 <= correct <= total:
        raise ValueError("invalid scored counts")
    if not total:
        return {"lower": None, "upper": None}
    z = 1.959963984540054
    p, z2 = correct / total, z * z
    centre = p + z2 / (2 * total)
    radius = z * sqrt(p * (1 - p) / total + z2 / (4 * total * total))
    return {
        "lower": (centre - radius) / (1 + z2 / total),
        "upper": (centre + radius) / (1 + z2 / total),
    }


@lru_cache(maxsize=256)
def _paired_interval(deltas):
    rng = random.Random(20261010)
    samples = sorted(
        sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(2000)
    )
    return samples[49], samples[1949]


def paired_graph_interval(deltas):
    """Preregistered paired percentile bootstrap: seed 20261010, 2000 draws."""
    values = tuple(deltas)
    if any(type(v) not in (int, float) or not isfinite(v) for v in values):
        raise ValueError("invalid paired metric")
    low, high = _paired_interval(values) if values else (None, None)
    return dict(
        count=len(values),
        mean=sum(values) / len(values) if values else None,
        lower=low,
        upper=high,
        confidence=0.95,
        seed=20261010,
        resamples=2000,
        method="paired-percentile-mean-v1",
    )
