"""Immutable representation scope values shared by ingestion and retrieval."""
from dataclasses import dataclass
from uuid import UUID

from lib.parsers.source_contracts import SourceArtifactRefV1


class SourceScopeUnavailable(ValueError):
    """Exact source identity cannot be proven for the captured scope."""


@dataclass(frozen=True)
class DocumentRefV1:
    model_label: str
    pkid: int
    id: UUID
    collection_id: int
    source_hash: str


@dataclass(frozen=True)
class RepresentationSelectionV1:
    document: DocumentRefV1
    generation_key: UUID | None
    canonical_sha256: str
    source_revision_sha256: str | None


@dataclass(frozen=True)
class RepresentationSnapshotV1:
    selections: tuple[RepresentationSelectionV1, ...]


@dataclass(frozen=True)
class BuildLeaseV1:
    generation_key: UUID
    lease_token: UUID


@dataclass(frozen=True)
class CitationSourceV1:
    document: DocumentRefV1
    representation_key: UUID | None
    canonical_text: str
    start: int
    end: int
    artifact: SourceArtifactRefV1 | None
