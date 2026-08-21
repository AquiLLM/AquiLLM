from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol
from uuid import UUID

from .chunk_reference_store import (
    DjangoChunkReferenceStore,
    private_row,
    validate_current_chunk_row,
)
from .django_projection_source import DjangoProjectionRowSource
from .identifiers import OpaqueProjectionKey, ProjectionIdentifierDomain
from .records import (
    CollectionGraphProjectionBundleV1,
    PrivateProjectionChunkReferenceV1,
)
from .serialization import private_chunk_mapping_checksum, projection_checksum

_MAX_PAGE = 5_000


def _page_size(value: object, name: str = "batch_size") -> int:
    if type(value) is not int or not 1 <= value <= _MAX_PAGE:
        raise ValueError(f"{name} must be an integer in 1..5000")
    return value


def _projection_id(value: object) -> UUID:
    if type(value) is not UUID:
        raise TypeError("projection_id must be an exact UUID")
    return value


class ProjectionRowSource(Protocol):
    def load_projection_rows(
        self, *, projection_id: UUID, batch_size: int, purpose: str = "build"
    ) -> Mapping[str, object]: ...

    def load_private_chunk_rows(
        self, *, projection_id: UUID, batch_size: int
    ) -> tuple[PrivateProjectionChunkReferenceV1, ...]: ...


class ChunkReferenceStore(Protocol):
    def load(
        self, *, projection_id: UUID, keys: tuple[str, ...] | None = None
    ) -> tuple[object, ...]: ...

    def create(
        self,
        *,
        projection_id: UUID,
        rows: tuple[PrivateProjectionChunkReferenceV1, ...],
        batch_size: int,
    ) -> None: ...

    def fence(self, *, projection_id: UUID, checksum: str, row_count: int) -> None: ...


def _requested_keys(chunk_keys: tuple[OpaqueProjectionKey, ...]) -> tuple[str, ...]:
    if type(chunk_keys) is not tuple or any(
        type(key) is not OpaqueProjectionKey
        or key.domain is not ProjectionIdentifierDomain.CHUNK
        for key in chunk_keys
    ):
        raise TypeError("chunk_keys must contain exact opaque chunk keys")
    values = tuple(key.value for key in chunk_keys)
    if values != tuple(sorted(set(values))):
        raise ValueError("chunk_keys must be sorted and unique")
    return values


class PostgresProjectionRepository:
    def __init__(
        self,
        *,
        using: str = "default",
        source: ProjectionRowSource | None = None,
        chunk_store: ChunkReferenceStore | None = None,
    ) -> None:
        if type(using) is not str or not using:
            raise ValueError("using must be a nonempty database alias")
        self._source = (
            source if source is not None else DjangoProjectionRowSource(using)
        )
        self._chunk_store = (
            chunk_store if chunk_store is not None else DjangoChunkReferenceStore(using)
        )

    def load_projection_bundle(
        self,
        *,
        projection_id: UUID,
        batch_size: int,
        purpose: str = "build",
    ) -> CollectionGraphProjectionBundleV1:
        identifier = _projection_id(projection_id)
        size = _page_size(batch_size)
        if type(purpose) is not str or purpose not in {"build", "audit", "prune"}:
            raise ValueError("projection load purpose must be build, audit, or prune")
        rows = self._source.load_projection_rows(
            projection_id=identifier,
            batch_size=size,
            purpose=purpose,
        )
        required = {
            "generation",
            "entities",
            "automatic_memberships",
            "documents",
            "chunks",
            "relations",
            "evidence",
            "artifact_provenance",
            "counts",
        }
        if type(rows) is not dict or set(rows) != required:
            raise ValueError("projection source returned an incomplete row family")
        bundle = CollectionGraphProjectionBundleV1(**rows)  # type: ignore[arg-type]
        projection_checksum(bundle)
        return bundle

    def persist_chunk_references(
        self,
        *,
        projection_id: UUID,
        rows: Sequence[PrivateProjectionChunkReferenceV1],
        batch_size: int,
    ) -> str:
        identifier = _projection_id(projection_id)
        size = _page_size(batch_size)
        requested = tuple(rows)
        if any(type(row) is not PrivateProjectionChunkReferenceV1 for row in requested):
            raise TypeError("rows must contain exact private chunk references")
        requested = tuple(
            sorted(
                requested,
                key=lambda row: (
                    row.projection_chunk_key,
                    row.integer_chunk_pk,
                    row.document_uuid,
                    row.chunk_number,
                ),
            )
        )
        expected_checksum = private_chunk_mapping_checksum(requested)
        existing = tuple(
            private_row(row) for row in self._chunk_store.load(projection_id=identifier)
        )
        by_key = {row.projection_chunk_key: row for row in existing}
        coordinates = {(row.document_uuid, row.chunk_number): row for row in existing}
        missing = []
        for row in requested:
            observed = by_key.get(row.projection_chunk_key)
            coordinate = coordinates.get((row.document_uuid, row.chunk_number))
            if observed is not None and observed != row:
                raise ValueError(
                    "projection chunk key conflicts with persisted mapping"
                )
            if coordinate is not None and coordinate != row:
                raise ValueError(
                    "projection chunk coordinate conflicts with persisted mapping"
                )
            if observed is None:
                missing.append(row)
        if missing:
            self._chunk_store.create(
                projection_id=identifier,
                rows=tuple(missing),
                batch_size=size,
            )
        persisted_objects = self._chunk_store.load(projection_id=identifier)
        for row in persisted_objects:
            validate_current_chunk_row(row)
        persisted = tuple(
            sorted(
                (private_row(row) for row in persisted_objects),
                key=lambda row: row.projection_chunk_key,
            )
        )
        if (
            persisted != requested
            or private_chunk_mapping_checksum(persisted) != expected_checksum
        ):
            raise ValueError(
                "persisted private chunk mapping is incomplete or conflicting"
            )
        self._chunk_store.fence(
            projection_id=identifier,
            checksum=expected_checksum,
            row_count=len(requested),
        )
        return expected_checksum

    def load_private_chunk_references(
        self, *, projection_id: UUID, batch_size: int
    ) -> tuple[PrivateProjectionChunkReferenceV1, ...]:
        identifier = _projection_id(projection_id)
        size = _page_size(batch_size)
        rows = self._source.load_private_chunk_rows(
            projection_id=identifier, batch_size=size
        )
        if type(rows) is not tuple or any(
            type(row) is not PrivateProjectionChunkReferenceV1 for row in rows
        ):
            raise TypeError("projection source returned invalid private chunk rows")
        private_chunk_mapping_checksum(rows)
        return rows

    def resolve_projection_chunk_references(
        self,
        *,
        projection_id: UUID,
        chunk_keys: tuple[OpaqueProjectionKey, ...],
        authorized_document_ids: frozenset[UUID],
    ) -> tuple[PrivateProjectionChunkReferenceV1, ...]:
        identifier = _projection_id(projection_id)
        keys = _requested_keys(chunk_keys)
        if type(authorized_document_ids) is not frozenset or any(
            type(value) is not UUID for value in authorized_document_ids
        ):
            raise TypeError("authorized_document_ids must be an exact UUID frozenset")
        observed = self._chunk_store.load(projection_id=identifier, keys=keys)
        if (
            tuple(sorted(getattr(row, "projection_chunk_key") for row in observed))
            != keys
        ):
            raise ValueError("projection chunk mapping is stale or deleted")
        resolved = []
        for row in observed:
            validate_current_chunk_row(row)
            if row.document_uuid in authorized_document_ids:
                resolved.append(private_row(row))
        return tuple(sorted(resolved, key=lambda row: row.projection_chunk_key))
