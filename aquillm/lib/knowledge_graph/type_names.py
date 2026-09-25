"""Stable type identifiers shared by schema validation and extraction providers."""

from __future__ import annotations

import re
from typing import Any

TYPE_NAME_MAX_LENGTH = 64
TYPE_NAME_PATTERN = r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$"
# GLiNER2 places this envelope key alongside relation type keys.
PROVIDER_RESERVED_TYPE_NAMES = frozenset({"entities"})


class TypeNameValidationError(ValueError):
    """A type identifier cannot be represented safely by storage or providers."""


def validate_type_name(value: Any, label: str = "type name") -> str:
    if (
        type(value) is not str
        or len(value) > TYPE_NAME_MAX_LENGTH
        or not re.fullmatch(TYPE_NAME_PATTERN, value)
    ):
        raise TypeNameValidationError(
            f"{label} must be canonical snake_case, "
            f"at most {TYPE_NAME_MAX_LENGTH} characters"
        )
    if value in PROVIDER_RESERVED_TYPE_NAMES:
        raise TypeNameValidationError(
            f"{label} is reserved by the extraction provider: {value}"
        )
    return value
