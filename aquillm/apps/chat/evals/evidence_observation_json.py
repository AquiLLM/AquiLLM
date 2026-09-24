"""One JSON representation for SDK observations, identities and saved reports."""

import math
from dataclasses import fields, is_dataclass
from enum import Enum
from uuid import UUID

from pydantic import BaseModel


def _normalize(value, unsupported):
    if isinstance(value, Enum):
        return _normalize(value.value, unsupported)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, BaseModel):
        try:
            shaped = value.model_dump(mode="json")
        except (TypeError, ValueError):
            return unsupported()
        return _normalize(shaped, unsupported)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _normalize(getattr(value, f.name), unsupported)
            for f in fields(value)
        }
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: _normalize(v, unsupported) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_normalize(v, unsupported) for v in value]
    return unsupported()


def normalize(value):
    def unsupported():
        raise TypeError("unsupported observation representation")

    return _normalize(value, unsupported)


def normalize_observation(row):
    failed = []

    def unsupported():
        failed.append(True)
        return {"unsupported_observation": True}

    result = _normalize(row, unsupported)
    if failed:
        result.update(
            observation_normalization_failed=True,
            provenance_complete=False,
            dispatch_accounting_complete=False,
        )
    return result
