"""Current-authorized reversal of opaque projected chunk keys."""

from __future__ import annotations

import re
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
        expected_private_mapping_checksum: str,
        database_alias: str,
    ) -> tuple[str, tuple[PrivateProjectionChunkReferenceV1, ...]]: ...

    def load_chunk_objects(
        self,
        *,
        chunk_predicates: tuple[tuple[int, UUID, int], ...],
        authorized_document_ids: tuple[UUID, ...],
        database_alias: str,
    ) -> tuple[object, ...]: ...


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
    loaded: object, requested: tuple[str, ...], expected_checksum: str
) -> dict[str, PrivateProjectionChunkReferenceV1]:
    if (
        type(loaded) is not tuple
        or len(loaded) != 2
        or type(loaded[0]) is not str
        or type(loaded[1]) is not tuple
    ):
        raise TypeError("private chunk map load envelope is not exact")
    checksum, rows = loaded
    if checksum != expected_checksum:
        raise ValueError("private mapping checksum mismatch")
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


def _chunk_objects(
    objects: object, predicates: tuple[tuple[int, UUID, int], ...]
) -> dict[int, object]:
    if type(objects) is not tuple:
        raise TypeError("chunk object materialization must be an exact tuple")
    expected = {pk: (document_id, number) for pk, document_id, number in predicates}
    by_pk: dict[int, object] = {}
    for candidate in objects:
        pk = getattr(candidate, "pk", None)
        document_id = getattr(candidate, "doc_id", None)
        chunk_number = getattr(candidate, "chunk_number", None)
        if type(pk) is not int or pk not in expected:
            raise ValueError("corrupt chunk object pk")
        if pk in by_pk:
            raise ValueError("duplicate chunk object materialization")
        if type(document_id) is not UUID or document_id != expected[pk][0]:
            raise ValueError("corrupt chunk object document")
        if type(chunk_number) is not int or chunk_number != expected[pk][1]:
            raise ValueError("corrupt chunk object chunk_number")
        by_pk[pk] = candidate
    if set(by_pk) != set(expected):
        raise ValueError("missing chunk object materialization")
    return by_pk


def materialize_projected_chunks(
    *,
    projection_id: UUID,
    expected_private_mapping_checksum: str,
    chunk_keys: tuple[OpaqueProjectionKey, ...],
    authorization: RetrievalAuthorizationContext,
    repository: PrivateChunkMapRepository,
) -> tuple[MaterializedGraphChunkV1, ...]:
    """Reverse exact keys, reauthorize, then load only still-authorized chunks."""

    if type(projection_id) is not UUID:
        raise TypeError("projection_id must be an exact UUID")
    if (
        type(expected_private_mapping_checksum) is not str
        or re.fullmatch(r"[0-9a-f]{64}", expected_private_mapping_checksum) is None
    ):
        raise ValueError("expected private mapping checksum must be lowercase SHA-256")
    if type(authorization) is not RetrievalAuthorizationContext:
        raise TypeError("authorization must be an exact retrieval context")
    if not isinstance(repository, PrivateChunkMapRepository):
        raise TypeError("repository must implement PrivateChunkMapRepository")
    requested = _requested_keys(chunk_keys)
    current = revalidate_retrieval_authorization_context(context=authorization)
    if (
        frozenset(current.collection_ids) != authorization.selected_collection_ids
        or frozenset(current.document_ids) != authorization.selected_document_ids
    ):
        return ()
    by_key = _private_rows(
        repository.load_private_chunk_map(
            projection_id=projection_id,
            chunk_keys=requested,
            expected_private_mapping_checksum=expected_private_mapping_checksum,
            database_alias=authorization.database_alias,
        ),
        requested,
        expected_private_mapping_checksum,
    )
    allowed = set(current.document_ids)
    retained = tuple(by_key[key] for key in requested)
    if any(UUID(row.document_uuid) not in allowed for row in retained):
        raise ValueError("private chunk map contains an unauthorized document")
    predicates = tuple(
        (row.integer_chunk_pk, UUID(row.document_uuid), row.chunk_number)
        for row in retained
    )
    objects = _chunk_objects(
        repository.load_chunk_objects(
            chunk_predicates=predicates,
            authorized_document_ids=current.document_ids,
            database_alias=authorization.database_alias,
        ),
        predicates,
    )
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
