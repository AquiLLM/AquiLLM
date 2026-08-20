"""Current-authorized reversal of opaque projected chunk keys."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable
from uuid import UUID

from apps.collections.services.retrieval_authorization import (
    RetrievalAuthorizationContext,
    revalidate_retrieval_authorization_context,
)
from apps.knowledge_graph.projection.identifiers import (
    OpaqueProjectionKey,
    ProjectionIdentifierDomain,
)
from apps.knowledge_graph.projection.records import PrivateProjectionChunkReferenceV1


@runtime_checkable
class PrivateChunkMapRepository(Protocol):
    def load_private_chunk_map(
        self,
        *,
        projection_id: UUID,
        chunk_keys: tuple[str, ...],
        database_alias: str,
    ) -> tuple[PrivateProjectionChunkReferenceV1, ...]: ...

    def load_chunk_objects(
        self, *, integer_chunk_pks: tuple[int, ...], database_alias: str
    ) -> Mapping[int, object]: ...


@dataclass(frozen=True, slots=True)
class MaterializedGraphChunkV1:
    chunk_key: str
    integer_chunk_pk: int
    document_uuid: UUID
    chunk_number: int
    candidate_object: object = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.chunk_key) is not str or len(self.chunk_key) != 64:
            raise ValueError("chunk_key must be an opaque projection key")
        if type(self.integer_chunk_pk) is not int or self.integer_chunk_pk < 1:
            raise ValueError("integer_chunk_pk must be a positive exact integer")
        if type(self.document_uuid) is not UUID:
            raise TypeError("document_uuid must be exact")
        if type(self.chunk_number) is not int or self.chunk_number < 0:
            raise ValueError("chunk_number must be a nonnegative exact integer")
        if self.candidate_object is None:
            raise TypeError("candidate_object must be non-null")


def _requested_keys(
    chunk_keys: tuple[OpaqueProjectionKey, ...],
) -> tuple[str, ...]:
    if type(chunk_keys) is not tuple or not chunk_keys or len(chunk_keys) > 20:
        raise ValueError("chunk_keys must be a nonempty exact tuple within its cap")
    if any(
        type(key) is not OpaqueProjectionKey
        or key.domain is not ProjectionIdentifierDomain.CHUNK
        for key in chunk_keys
    ):
        raise TypeError("chunk_keys must contain exact opaque chunk keys")
    values = tuple(key.value for key in chunk_keys)
    if len(set(values)) != len(values):
        raise ValueError("duplicate requested chunk keys")
    return values


def _private_rows(
    rows: object, requested: tuple[str, ...]
) -> dict[str, PrivateProjectionChunkReferenceV1]:
    if type(rows) is not tuple or any(
        type(row) is not PrivateProjectionChunkReferenceV1 for row in rows
    ):
        raise TypeError("private chunk map must contain exact reference rows")
    by_key: dict[str, PrivateProjectionChunkReferenceV1] = {}
    coordinates: dict[int, tuple[str, str, int]] = {}
    for row in rows:
        if row.projection_chunk_key in by_key:
            raise ValueError("duplicate private chunk map row")
        by_key[row.projection_chunk_key] = row
        coordinate = (
            row.projection_chunk_key,
            row.document_uuid,
            row.chunk_number,
        )
        previous = coordinates.setdefault(row.integer_chunk_pk, coordinate)
        if previous != coordinate:
            raise ValueError("conflict in private chunk map coordinates")
    if set(by_key) != set(requested):
        raise ValueError("stale private chunk map coverage")
    return by_key


def materialize_projected_chunks(
    *,
    projection_id: UUID,
    chunk_keys: tuple[OpaqueProjectionKey, ...],
    authorization: RetrievalAuthorizationContext,
    repository: PrivateChunkMapRepository,
) -> tuple[MaterializedGraphChunkV1, ...]:
    """Reverse exact keys, reauthorize, then load only still-authorized chunks."""

    if type(projection_id) is not UUID:
        raise TypeError("projection_id must be an exact UUID")
    if type(authorization) is not RetrievalAuthorizationContext:
        raise TypeError("authorization must be an exact retrieval context")
    if not isinstance(repository, PrivateChunkMapRepository):
        raise TypeError("repository must implement PrivateChunkMapRepository")
    requested = _requested_keys(chunk_keys)
    by_key = _private_rows(
        repository.load_private_chunk_map(
            projection_id=projection_id,
            chunk_keys=requested,
            database_alias=authorization.database_alias,
        ),
        requested,
    )
    current = revalidate_retrieval_authorization_context(context=authorization)
    allowed = set(current.document_ids)
    retained = tuple(
        by_key[key] for key in requested if UUID(by_key[key].document_uuid) in allowed
    )
    pks = tuple(row.integer_chunk_pk for row in retained)
    objects = repository.load_chunk_objects(
        integer_chunk_pks=pks,
        database_alias=authorization.database_alias,
    )
    if not isinstance(objects, Mapping) or set(objects) != set(pks):
        raise ValueError("stale chunk object materialization")
    if any(type(pk) is not int or objects[pk] is None for pk in objects):
        raise ValueError("conflict in chunk object materialization")
    return tuple(
        MaterializedGraphChunkV1(
            row.projection_chunk_key,
            row.integer_chunk_pk,
            UUID(row.document_uuid),
            row.chunk_number,
            objects[row.integer_chunk_pk],
        )
        for row in retained
    )


__all__ = ["MaterializedGraphChunkV1", "materialize_projected_chunks"]
