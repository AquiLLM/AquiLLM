"""Bounded chunk windows and lossless mention span remapping.

Extractor offsets are Python half-open string offsets into ``TextChunk.content``.
Text evidence is remapped by adding ``TextChunk.start_position`` and is accepted
only when the corresponding ``Document.full_text`` slice is equal after NFC
normalization.  NFC is deliberately the only normalization used for source
validation; case, punctuation, and whitespace remain evidence-bearing.

Image/figure chunk offsets never enter the document-global coordinate system.
They remain local to the chunk content and carry the exact concrete document
content type and object UUID.
"""

from __future__ import annotations

import unicodedata
import uuid
from dataclasses import dataclass, replace

from lib.knowledge_graph.types import EntityCandidate


class SpanMappingError(ValueError):
    """Raised when provider evidence cannot be mapped to persisted source text."""


@dataclass(frozen=True, slots=True)
class ExtractionWindow:
    """One independently extracted persisted chunk with remapping provenance."""

    chunk_id: int
    document_id: uuid.UUID
    content: str
    start_position: int
    modality: str
    content_object_type_id: int | None = None
    content_object_app_label: str | None = None
    content_object_model: str | None = None
    content_object_id: uuid.UUID | None = None

    def __post_init__(self) -> None:
        if self.chunk_id < 0:
            raise ValueError("chunk_id must be nonnegative")
        if self.start_position < 0:
            raise ValueError("start_position must be nonnegative")
        if self.modality not in {"text", "image"}:
            raise ValueError("modality must be text or image")
        provenance = (
            self.content_object_type_id,
            self.content_object_app_label,
            self.content_object_model,
            self.content_object_id,
        )
        if self.modality == "text" and any(value is not None for value in provenance):
            raise ValueError("text windows cannot carry image provenance")
        if self.modality == "image" and any(value is None for value in provenance):
            raise ValueError("image windows require exact content-object provenance")


@dataclass(frozen=True, slots=True)
class MentionObservation:
    """One provider observation contributing to a deduplicated mention."""

    chunk_id: int
    local_start: int
    local_end: int
    confidence: float
    modality: str
    position_basis: str
    start: int
    end: int
    content_object_type_id: int | None
    content_object_id: uuid.UUID | None


@dataclass(frozen=True, slots=True)
class MappedEntityEvidence:
    """A persistence-ready mention plus all overlapping chunk observations."""

    document_id: uuid.UUID
    chunk_id: int
    start: int
    end: int
    position_basis: str
    raw_text: str
    normalized_text: str
    entity_type: str
    confidence: float
    content_object_type_id: int | None
    content_object_id: uuid.UUID | None
    observations: tuple[MentionObservation, ...]

    @property
    def identity_key(self) -> tuple[object, ...]:
        common = (
            self.document_id,
            self.position_basis,
            self.start,
            self.end,
            self.entity_type,
            self.normalized_text,
        )
        if self.position_basis == "document_global":
            return common
        # Chunk-local coordinates are not comparable between distinct chunks.
        return (*common, self.chunk_id, self.content_object_id)


def normalize_source_text(value: str) -> str:
    """Normalize source slices for equality using Unicode NFC and nothing else."""

    return unicodedata.normalize("NFC", value)


def batch_extraction_windows(
    windows: tuple[ExtractionWindow, ...],
    *,
    max_count: int,
    max_characters: int,
) -> tuple[tuple[ExtractionWindow, ...], ...]:
    """Group independent chunk inputs under provider count and character guards."""

    if max_count <= 0:
        raise ValueError("max_count must be positive")
    if max_characters <= 0:
        raise ValueError("max_characters must be positive")

    batches: list[tuple[ExtractionWindow, ...]] = []
    current: list[ExtractionWindow] = []
    character_count = 0
    for window in windows:
        size = len(window.content)
        if size > max_characters:
            raise ValueError(
                "one extraction window exceeds the provider character guard"
            )
        if current and (
            len(current) >= max_count or character_count + size > max_characters
        ):
            batches.append(tuple(current))
            current = []
            character_count = 0
        current.append(window)
        character_count += size
    if current:
        batches.append(tuple(current))
    return tuple(batches)


def map_entity_candidate(
    window: ExtractionWindow,
    candidate: EntityCandidate,
    *,
    full_text: str,
) -> MappedEntityEvidence:
    """Map one candidate without confusing text and image coordinate systems."""

    if candidate.end > len(window.content):
        raise SpanMappingError("candidate span exceeds chunk content")
    if window.content[candidate.start : candidate.end] != candidate.text:
        raise SpanMappingError("candidate surface does not match chunk source slice")

    if window.modality == "text":
        start = window.start_position + candidate.start
        end = window.start_position + candidate.end
        if not 0 <= start < end <= len(full_text):
            raise SpanMappingError("mapped text span exceeds document bounds")
        source_slice = full_text[start:end]
        if normalize_source_text(source_slice) != normalize_source_text(candidate.text):
            raise SpanMappingError(
                "document source slice does not match extracted text"
            )
        position_basis = "document_global"
        content_object_type_id = None
        content_object_id = None
    else:
        start = candidate.start
        end = candidate.end
        position_basis = "chunk_content"
        content_object_type_id = window.content_object_type_id
        content_object_id = window.content_object_id

    observation = MentionObservation(
        chunk_id=window.chunk_id,
        local_start=candidate.start,
        local_end=candidate.end,
        confidence=candidate.confidence,
        modality=window.modality,
        position_basis=position_basis,
        start=start,
        end=end,
        content_object_type_id=content_object_type_id,
        content_object_id=content_object_id,
    )
    return MappedEntityEvidence(
        document_id=window.document_id,
        chunk_id=window.chunk_id,
        start=start,
        end=end,
        position_basis=position_basis,
        raw_text=candidate.text,
        normalized_text=normalize_source_text(candidate.text),
        entity_type=candidate.entity_type,
        confidence=candidate.confidence,
        content_object_type_id=content_object_type_id,
        content_object_id=content_object_id,
        observations=(observation,),
    )


def deduplicate_mapped_entities(
    entities: tuple[MappedEntityEvidence, ...],
) -> tuple[MappedEntityEvidence, ...]:
    """Deduplicate overlapping global mentions while retaining every observation."""

    deduplicated: dict[tuple[object, ...], MappedEntityEvidence] = {}
    for entity in entities:
        existing = deduplicated.get(entity.identity_key)
        if existing is None:
            deduplicated[entity.identity_key] = entity
            continue
        observations = tuple(
            sorted(
                (*existing.observations, *entity.observations),
                key=lambda item: (item.chunk_id, item.local_start, item.local_end),
            )
        )
        representative = entity if entity.confidence > existing.confidence else existing
        deduplicated[entity.identity_key] = replace(
            representative,
            observations=observations,
        )
    return tuple(
        sorted(
            deduplicated.values(),
            key=lambda item: (
                str(item.document_id),
                item.position_basis,
                item.start,
                item.end,
                item.entity_type,
                item.chunk_id,
            ),
        )
    )


__all__ = [
    "ExtractionWindow",
    "MappedEntityEvidence",
    "MentionObservation",
    "SpanMappingError",
    "batch_extraction_windows",
    "deduplicate_mapped_entities",
    "map_entity_candidate",
    "normalize_source_text",
]
