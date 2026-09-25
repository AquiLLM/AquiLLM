"""Validated cursor and row values for bounded graph recovery."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

_DOCUMENT_PHASE = "documents"
_COLLECTION_PHASE = "collections"
_PHASES = frozenset({_DOCUMENT_PHASE, _COLLECTION_PHASE})
_MAX_PAGE_SIZE = 500
@dataclass(frozen=True, slots=True)
class DocumentRecoveryRow:
    document_id: uuid.UUID
    source_hash: str


@dataclass(frozen=True, slots=True)
class GraphRecoveryCursor:
    phase: str = _DOCUMENT_PHASE
    document_model_index: int = 0
    last_pk: int = 0

    def __post_init__(self) -> None:
        if self.phase not in _PHASES:
            raise ValueError("graph recovery cursor phase is invalid")
        if type(self.document_model_index) is not int or self.document_model_index < 0:
            raise ValueError("graph recovery document model cursor is invalid")
        if type(self.last_pk) is not int or self.last_pk < 0:
            raise ValueError("graph recovery primary-key cursor is invalid")
        if self.phase == _COLLECTION_PHASE and self.document_model_index != 0:
            raise ValueError("collection recovery cursor cannot name a document model")

    def as_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "document_model_index": self.document_model_index,
            "last_pk": self.last_pk,
        }


def _source_hash(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("document recovery source hash must be lowercase SHA-256")
    return value


def _document_id(value: object) -> uuid.UUID:
    if type(value) is not uuid.UUID or value.version is None:
        raise ValueError("document recovery id must be an exact RFC 4122 UUID")
    return value


def _page_size(value: object) -> int:
    if type(value) is not int or not 1 <= value <= _MAX_PAGE_SIZE:
        raise ValueError(f"graph recovery page size must be 1..{_MAX_PAGE_SIZE}")
    return value


def _cursor(value: object) -> GraphRecoveryCursor:
    if value is None:
        return GraphRecoveryCursor()
    if type(value) is not dict or set(value) != {
        "phase",
        "document_model_index",
        "last_pk",
    }:
        raise ValueError("graph recovery cursor must be an exact mapping")
    return GraphRecoveryCursor(
        phase=value["phase"],
        document_model_index=value["document_model_index"],
        last_pk=value["last_pk"],
    )
