"""Exact scalar argument validation for bounded projection repository calls."""

from __future__ import annotations

from math import isfinite

from .identifiers import OpaqueProjectionKey

_MAX_PAGE = 5_000


def _size(value: object, name: str) -> int:
    if type(value) is not int or not 1 <= value <= _MAX_PAGE:
        raise ValueError(f"{name} must be an integer in 1..5000")
    return value


def _timeout(value: object) -> float:
    if type(value) is not float or not isfinite(value) or value <= 0.0:
        raise ValueError("timeout_seconds must be a finite positive float")
    return value


def _key(value: object) -> str:
    if type(value) is not OpaqueProjectionKey:
        raise TypeError("generation key must be an exact opaque key")
    return value.value
