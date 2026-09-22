"""Bounded, exact input validation for document coreference."""

from __future__ import annotations

import unicodedata
from math import isfinite
from uuid import UUID

_MAX_SOURCE_TEXT_CHARACTERS = 1_000_000
_MAX_UNIQUE_SOURCE_CONTEXT_CHARACTERS = 2_000_000
_MAX_IDENTIFIER_CHARACTERS = 2_048
_MAX_SOURCE_KEY_CHARACTERS = 512
_MAX_MENTION_ID_CHARACTERS = 128
_MAX_ENTITY_TYPE_CHARACTERS = 128
_MAX_DB_INTEGER = 2**63 - 1
def _require_string(value: object, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field_name} must be a nonempty string")
    return value


def _contains_unsafe_control(
    value: str,
    *,
    allow_text_whitespace: bool,
    allow_format_controls: bool = False,
) -> bool:
    allowed = {"\t", "\n", "\r"} if allow_text_whitespace else set()
    for character in value:
        category = unicodedata.category(character)
        if character not in allowed and category in {"Cc", "Cs"}:
            return True
        if not allow_format_controls and category == "Cf":
            return True
    return False


def _mention_key(value: object) -> str:
    if isinstance(value, bool) or value is None:
        raise ValueError("mention_id must be a stable nonempty scalar")
    try:
        key = str(value).strip()
    except (OverflowError, ValueError) as exc:
        raise ValueError("mention_id must be a stable nonempty scalar") from exc
    if not key:
        raise ValueError("mention_id must be a stable nonempty scalar")
    if len(key) > _MAX_MENTION_ID_CHARACTERS:
        raise ValueError(
            f"mention_id exceeds the {_MAX_MENTION_ID_CHARACTERS}-character limit"
        )
    if _contains_unsafe_control(key, allow_text_whitespace=False):
        raise ValueError("mention_id contains an unsafe control character")
    return key


def _confidence(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("mention confidence must be a finite confidence in [0, 1]")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(
            "mention confidence must be a finite confidence in [0, 1]"
        ) from exc
    if not isfinite(converted) or not 0 <= converted <= 1:
        raise ValueError("mention confidence must be a finite confidence in [0, 1]")
    return converted


def _validated_source_text(value: object) -> str:
    if type(value) is not str:
        raise ValueError("source text must be a string")
    if len(value) > _MAX_SOURCE_TEXT_CHARACTERS:
        raise ValueError(
            f"source text exceeds the {_MAX_SOURCE_TEXT_CHARACTERS}-character limit"
        )
    if _contains_unsafe_control(
        value,
        allow_text_whitespace=True,
        allow_format_controls=True,
    ):
        raise ValueError("source text contains an unsafe control character")
    return value


def _validated_identifier(value: object) -> str:
    if type(value) is not str:
        raise ValueError("identifier must be a string")
    if len(value) > _MAX_IDENTIFIER_CHARACTERS:
        raise ValueError(
            f"identifier exceeds the {_MAX_IDENTIFIER_CHARACTERS}-character limit"
        )
    if _contains_unsafe_control(value, allow_text_whitespace=False):
        raise ValueError("identifier contains an unsafe control character")
    return value


def _validated_source_key(value: object) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError("source key must be a nonempty string")
    if len(value) > _MAX_SOURCE_KEY_CHARACTERS:
        raise ValueError(
            f"source key exceeds the {_MAX_SOURCE_KEY_CHARACTERS}-character limit"
        )
    if _contains_unsafe_control(value, allow_text_whitespace=False):
        raise ValueError("source key contains an unsafe control character")
    return value


def _validated_entity_type(value: object) -> str:
    entity_type = _require_string(value, "entity_type")
    if len(entity_type) > _MAX_ENTITY_TYPE_CHARACTERS:
        raise ValueError(
            f"entity_type exceeds the {_MAX_ENTITY_TYPE_CHARACTERS}-character limit"
        )
    if _contains_unsafe_control(entity_type, allow_text_whitespace=False):
        raise ValueError("entity_type contains an unsafe control character")
    return entity_type


def _canonical_uuid(value: object, field_name: str) -> str:
    if type(value) not in {str, UUID}:
        raise ValueError(f"{field_name} must be a UUID")
    if type(value) is str and len(value) > 64:
        raise ValueError(f"{field_name} exceeds the 64-character UUID limit")
    try:
        parsed = value if type(value) is UUID else UUID(value)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a UUID") from exc
    return str(parsed)


def _validated_db_integer(
    value: object,
    field_name: str,
    *,
    minimum: int,
) -> int:
    if type(value) is not int:
        raise ValueError(f"{field_name} must be an exact integer")
    if not minimum <= value <= _MAX_DB_INTEGER:
        raise ValueError(
            f"{field_name} must be between {minimum} and {_MAX_DB_INTEGER}"
        )
    return value


def _validated_span(start: object, end: object) -> tuple[int, int]:
    validated_start = _validated_db_integer(start, "start", minimum=0)
    validated_end = _validated_db_integer(end, "end", minimum=1)
    if validated_end <= validated_start:
        raise ValueError("end must be greater than start")
    return validated_start, validated_end


def _validated_coordinate_basis(
    position_basis: object,
    content_object_id: object,
) -> tuple[str, str]:
    if type(position_basis) is not str or position_basis not in {
        "document_global",
        "chunk_content",
    }:
        raise ValueError("position_basis must be document_global or chunk_content")
    if position_basis == "document_global":
        if content_object_id is not None:
            raise ValueError(
                "document_global coordinate basis requires content_object_id=None"
            )
        return position_basis, ""
    if content_object_id is None:
        raise ValueError(
            "chunk_content coordinate basis requires a UUID content_object_id"
        )
    return position_basis, _canonical_uuid(content_object_id, "content_object_id")
