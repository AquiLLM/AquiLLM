"""Finite numeric validation for graph evidence and ranking."""

from __future__ import annotations

from collections.abc import Iterable
from math import fsum, isfinite


def _unit_float(value: object, field_name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{field_name} must be a finite number in [0, 1]")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{field_name} must be a finite number in [0, 1]") from error
    if not isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be a finite number in [0, 1]")
    return number


def _positive_float(value: object, field_name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{field_name} must be finite and positive")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{field_name} must be finite and positive") from error
    if not isfinite(number) or number <= 0.0:
        raise ValueError(f"{field_name} must be finite and positive")
    return number


def _safe_fsum(values: Iterable[float], field_name: str) -> float:
    try:
        total = fsum(values)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{field_name} produced non-finite math") from error
    if not isfinite(total) or total < 0.0:
        raise ValueError(f"{field_name} produced non-finite math")
    return total
