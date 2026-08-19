"""Immutable value types for the provider-neutral fixture manifest."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from uuid import UUID


class FixtureValidationError(ValueError):
    """Raised when a logical or resolved fixture is not exact."""


@dataclass(frozen=True, slots=True)
class FixtureCollectionBinding:
    symbol: str
    collection_id: int
    rebuild_request_id: UUID | None
    authorized: bool


@dataclass(frozen=True, slots=True)
class FixtureDocumentBinding:
    symbol: str
    document_id: UUID
    collection_symbol: str
    collection_id: int
    full_text_sha256: str


@dataclass(frozen=True, slots=True)
class FixtureChunkBinding:
    symbol: str
    chunk_id: int
    document_symbol: str
    collection_symbol: str
    collection_id: int
    chunk_number: int
    start: int
    end: int
    content_sha256: str
    embedding_sha256: str


@dataclass(frozen=True, slots=True)
class FixtureCanonicalIdentityAssertion:
    source_chunk_symbol: str
    target_chunk_symbol: str
    expected_outcome: str


@dataclass(frozen=True, slots=True)
class FixtureInaccessibleNeighborAssertion:
    source_chunk_symbol: str
    target_chunk_symbol: str


@dataclass(frozen=True, slots=True)
class FixtureEmbeddingBinding:
    model: str
    checkpoint: str
    dimensions: int
    input_type: str
    endpoint_signature: str


@dataclass(frozen=True, slots=True)
class ResolvedFixtureManifest:
    fixture_id: str
    fixture_checksum: str
    embedding: FixtureEmbeddingBinding
    authorized_scope: tuple[tuple[int, UUID], ...]
    collections: Mapping[str, FixtureCollectionBinding]
    documents: Mapping[str, FixtureDocumentBinding]
    chunks: Mapping[str, FixtureChunkBinding]
    canonical_identity_assertions: tuple[FixtureCanonicalIdentityAssertion, ...]
    inaccessible_neighbor_assertions: tuple[FixtureInaccessibleNeighborAssertion, ...]
    manifest_checksum: str

    def chunk(self, symbol: str) -> FixtureChunkBinding:
        try:
            return self.chunks[symbol]
        except KeyError as error:
            raise FixtureValidationError(
                f"fixture manifest has no chunk symbol {symbol!r}"
            ) from error
