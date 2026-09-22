# ruff: noqa: E501,E701,E702
"""Validated direct-seed scope and candidate records."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID


# fmt: off
@dataclass(frozen=True, slots=True, repr=False)
class DirectSeedScopeV1:
    ready_bundle_checksum: str
    selected_collection_ids: tuple[int, ...]
    selected_artifact_ids: tuple[int, ...]
    selected_document_ids: tuple[UUID, ...]
    selected_document_artifact_ids: tuple[int, ...]
    generation_keys_by_artifact: tuple[tuple[int, str], ...]
    generation_ids_by_artifact: tuple[tuple[int, UUID], ...]
    ontology_checksum: str
    resolver_version: str
    expected_embedding_signature: str = "embed-v1"

    def __post_init__(self) -> None:
        for values, name in (
            (self.selected_collection_ids, "selected_collection_ids"),
            (self.selected_artifact_ids, "selected_artifact_ids"),
            (self.selected_document_artifact_ids, "selected_document_artifact_ids"),
        ):
            if (
                type(values) is not tuple
                or not values
                or any(type(value) is not int or value <= 0 for value in values)
                or values != tuple(sorted(set(values)))
            ):
                raise ValueError(f"{name} must be a sorted unique exact tuple")
        if (
            type(self.selected_document_ids) is not tuple
            or not self.selected_document_ids
            or any(type(value) is not UUID for value in self.selected_document_ids)
            or self.selected_document_ids
            != tuple(sorted(set(self.selected_document_ids), key=str))
        ):
            raise ValueError("selected_document_ids must be sorted and unique")
        if (
            tuple(row[0] for row in self.generation_keys_by_artifact)
            != self.selected_artifact_ids
        ):
            raise ValueError("generation mapping must cover selected artifacts")
        if tuple(
            row[0] for row in self.generation_ids_by_artifact
        ) != self.selected_artifact_ids or any(
            type(row[1]) is not UUID for row in self.generation_ids_by_artifact
        ):
            raise ValueError(
                "raw generation UUID mapping must cover selected artifacts"
            )


@dataclass(frozen=True, slots=True)
class DirectSeedCandidateRow:
    entity_id: int; artifact_id: int; ontology_type: str; automatic_canonical_entity_id: int | None; similarity: float
    link_outcome: str = "automatic"

    def __post_init__(self):
        canonical = self.automatic_canonical_entity_id
        if canonical is not None and (type(canonical) is not int or canonical <= 0):
            raise ValueError("automatic canonical entity must be a positive database PK")
