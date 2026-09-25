"""Evidence values, endpoint matching, and bounded deduplication."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING

from ..resolution.normalization import normalize_entity_label
from .windows import MappedEntityEvidence

if TYPE_CHECKING:
    from lib.knowledge_graph.extractors.base import OntologyDefinition
    from lib.knowledge_graph.types import RelationCandidate


class StructuralExtractionError(RuntimeError):
    """Raised when a provider omits a required per-window output section."""


class ExtractionCapacityCode(StrEnum):
    """Stable, non-sensitive failure codes for bounded extraction limits."""

    CHUNK_LIMIT = "extraction_chunk_limit"
    CHARACTER_LIMIT = "extraction_character_limit"
    ENTITY_LIMIT = "extraction_entity_limit"
    RELATION_LIMIT = "extraction_relation_limit"
    OBSERVATION_LIMIT = "extraction_observation_limit"


class ExtractionCapacityError(StructuralExtractionError):
    """Raised when complete extraction cannot fit a published document bound."""

    def __init__(self, code: ExtractionCapacityCode, message: str) -> None:
        if type(code) is not ExtractionCapacityCode:
            raise TypeError("capacity error code must be an ExtractionCapacityCode")
        super().__init__(message)
        self._capacity_code = code

    @property
    def code(self) -> ExtractionCapacityCode:
        return self._capacity_code
@dataclass(frozen=True, slots=True)
class RelationObservation:
    chunk_id: int
    confidence: float
    modality: str
    head_local_start: int
    head_local_end: int
    tail_local_start: int
    tail_local_end: int


@dataclass(frozen=True, slots=True)
class MappedRelationEvidence:
    document_id: object
    chunk_id: int
    relation_type: str
    head_identity: tuple[object, ...]
    tail_identity: tuple[object, ...]
    confidence: float
    observations: tuple[RelationObservation, ...]

    @property
    def identity_key(self) -> tuple[object, ...]:
        return (self.relation_type, self.head_identity, self.tail_identity)


@dataclass(frozen=True, slots=True)
class ExtractedDocumentEvidence:
    entities: tuple[MappedEntityEvidence, ...]
    relations: tuple[MappedRelationEvidence, ...]
    diagnostic_counts: dict[str, int]
    window_count: int
    batch_count: int


def _resolvable_entities(entities, diagnostic_counts):
    """Exclude provider labels that cannot become resolved graph identities."""

    accepted = []
    for entity in entities:
        try:
            normalize_entity_label(entity.raw_text)
        except ValueError:
            diagnostic_counts["unresolvable_entity_label"] += 1
            continue
        accepted.append(entity)
    return tuple(accepted)


def serialize_entity_observations(
    entity: MappedEntityEvidence,
) -> list[dict[str, str | int | float | bool | None]]:
    """Return immutable observation provenance containing JSON scalar values only."""

    return [
        {
            "chunk_id": observation.chunk_id,
            "confidence": observation.confidence,
            "modality": observation.modality,
            "position_basis": observation.position_basis,
            "start": observation.start,
            "end": observation.end,
            "local_start": observation.local_start,
            "local_end": observation.local_end,
            "content_object_type_id": observation.content_object_type_id,
            "content_object_id": (
                str(observation.content_object_id)
                if observation.content_object_id is not None
                else None
            ),
        }
        for observation in entity.observations
    ]


def _definition_value(definition: object, field_name: str) -> object:
    if isinstance(definition, dict):
        return definition.get(field_name)
    return getattr(definition, field_name, None)


def _allowed_endpoint_types(
    ontology: OntologyDefinition,
    relation_type: str,
    endpoint: str,
) -> frozenset[str]:
    relation_definition = ontology.relations.get(relation_type)
    if relation_definition is None:
        return frozenset()
    raw_types = _definition_value(relation_definition, f"allowed_{endpoint}_types")
    if not isinstance(raw_types, (tuple, list)):
        return frozenset()
    return frozenset(value for value in raw_types if isinstance(value, str))


def _endpoint_entities(
    relation: RelationCandidate,
    endpoint: str,
    allowed_endpoint: str,
    mapped_entities: tuple[MappedEntityEvidence, ...],
    ontology: OntologyDefinition,
) -> tuple[MappedEntityEvidence, ...]:
    surface = getattr(relation, f"{endpoint}_text")
    start = getattr(relation, f"{endpoint}_start")
    end = getattr(relation, f"{endpoint}_end")
    allowed = _allowed_endpoint_types(
        ontology, relation.relation_type, allowed_endpoint
    )
    matches = [
        entity
        for entity in mapped_entities
        if entity.observations[0].local_start == start
        and entity.observations[0].local_end == end
        and entity.raw_text == surface
        and entity.entity_type in allowed
    ]
    return tuple(matches)


def _accumulate_entity(
    accumulated: dict[
        tuple[object, ...], tuple[MappedEntityEvidence, list[object]]
    ],
    entity: MappedEntityEvidence,
    *,
    max_entities: int,
) -> None:
    existing = accumulated.get(entity.identity_key)
    if existing is None:
        if len(accumulated) >= max_entities:
            raise ExtractionCapacityError(
                ExtractionCapacityCode.ENTITY_LIMIT,
                "deduplicated entity cap exceeded",
            )
        accumulated[entity.identity_key] = (entity, list(entity.observations))
        return
    representative, observations = existing
    observations.extend(entity.observations)
    if entity.confidence > representative.confidence:
        accumulated[entity.identity_key] = (entity, observations)


def _finalize_entities(
    accumulated: dict[
        tuple[object, ...], tuple[MappedEntityEvidence, list[object]]
    ],
) -> tuple[MappedEntityEvidence, ...]:
    entities = (
        replace(
            representative,
            observations=tuple(
                sorted(
                    observations,
                    key=lambda item: (
                        item.chunk_id,
                        item.local_start,
                        item.local_end,
                    ),
                )
            ),
        )
        for representative, observations in accumulated.values()
    )
    return tuple(
        sorted(
            entities,
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


def _accumulate_relation(
    accumulated: dict[
        tuple[object, ...], tuple[MappedRelationEvidence, list[RelationObservation]]
    ],
    relation: MappedRelationEvidence,
    *,
    max_relations: int,
) -> None:
    existing = accumulated.get(relation.identity_key)
    if existing is None:
        if len(accumulated) >= max_relations:
            raise ExtractionCapacityError(
                ExtractionCapacityCode.RELATION_LIMIT,
                "deduplicated relation cap exceeded",
            )
        accumulated[relation.identity_key] = (relation, list(relation.observations))
        return
    representative, observations = existing
    observations.extend(relation.observations)
    if relation.confidence > representative.confidence:
        accumulated[relation.identity_key] = (relation, observations)


def _finalize_relations(
    accumulated: dict[
        tuple[object, ...], tuple[MappedRelationEvidence, list[RelationObservation]]
    ],
) -> tuple[MappedRelationEvidence, ...]:
    relations = (
        replace(
            representative,
            observations=tuple(
                sorted(
                    observations,
                    key=lambda item: (
                        item.chunk_id,
                        item.head_local_start,
                        item.tail_local_start,
                    ),
                )
            ),
        )
        for representative, observations in accumulated.values()
    )
    return tuple(
        sorted(
            relations,
            key=lambda item: (
                item.relation_type,
                repr(item.head_identity),
                repr(item.tail_identity),
            ),
        )
    )
