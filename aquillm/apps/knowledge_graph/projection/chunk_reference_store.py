from __future__ import annotations

from uuid import UUID

from django.db import transaction

from apps.knowledge_graph.models import (
    CollectionGraphProjection,
    ProjectionChunkReference,
)

from .records import PrivateProjectionChunkReferenceV1
from .serialization import private_chunk_mapping_checksum

_MAX_PAGE = 5_000


def private_row(row: object) -> PrivateProjectionChunkReferenceV1:
    return PrivateProjectionChunkReferenceV1(
        projection_chunk_key=getattr(row, "projection_chunk_key"),
        integer_chunk_pk=getattr(row, "integer_chunk_pk"),
        document_uuid=str(getattr(row, "document_uuid")),
        chunk_number=getattr(row, "chunk_number"),
    )


def validate_current_chunk_row(row: object) -> None:
    if getattr(row, "chunk_id", None) is None:
        raise ValueError("projection chunk mapping is stale or deleted")
    chunk = getattr(row, "chunk", None)
    if (
        chunk is None
        or chunk.pk != getattr(row, "integer_chunk_pk")
        or chunk.doc_id != getattr(row, "document_uuid")
        or chunk.chunk_number != getattr(row, "chunk_number")
    ):
        raise ValueError("projection chunk mapping conflicts with current chunk")


class DjangoChunkReferenceStore:
    def __init__(self, using: str) -> None:
        self.using = using

    def load(
        self, *, projection_id: UUID, keys: tuple[str, ...] | None = None
    ) -> tuple[ProjectionChunkReference, ...]:
        query = ProjectionChunkReference.objects.using(self.using).filter(
            projection_id=projection_id
        )
        if keys is not None:
            query = query.filter(projection_chunk_key__in=keys)
        return tuple(
            query.select_related("chunk")
            .order_by("projection_chunk_key")
            .iterator(chunk_size=_MAX_PAGE)
        )

    def create(
        self,
        *,
        projection_id: UUID,
        rows: tuple[PrivateProjectionChunkReferenceV1, ...],
        batch_size: int,
    ) -> None:
        values = [
            ProjectionChunkReference(
                projection_id=projection_id,
                projection_chunk_key=row.projection_chunk_key,
                chunk_id=row.integer_chunk_pk,
                integer_chunk_pk=row.integer_chunk_pk,
                document_uuid=UUID(row.document_uuid),
                chunk_number=row.chunk_number,
            )
            for row in rows
        ]
        with transaction.atomic(using=self.using):
            ProjectionChunkReference.objects.using(self.using).bulk_create(
                values, batch_size=batch_size, ignore_conflicts=True
            )

    def fence(self, *, projection_id: UUID, checksum: str, row_count: int) -> None:
        empty_checksum = private_chunk_mapping_checksum(())
        with transaction.atomic(using=self.using):
            projection = (
                CollectionGraphProjection.objects.using(self.using)
                .select_for_update()
                .get(pk=projection_id)
            )
            if (
                projection.state != "building"
                or projection.collection_id is None
                or projection.artifact_id is None
                or projection.private_mapping_checksum not in {empty_checksum, checksum}
            ):
                raise ValueError("projection private mapping fence is stale")
            stored = tuple(
                ProjectionChunkReference.objects.using(self.using)
                .select_for_update()
                .filter(projection_id=projection_id)
                .order_by("projection_chunk_key")
            )
            for row in stored:
                validate_current_chunk_row(row)
            persisted = tuple(
                sorted(
                    (private_row(row) for row in stored),
                    key=lambda row: row.projection_chunk_key,
                )
            )
            if (
                len(persisted) != row_count
                or private_chunk_mapping_checksum(persisted) != checksum
                or (row_count > 0 and checksum == empty_checksum)
            ):
                raise ValueError("projection private mapping fence is incomplete")
            updated = (
                CollectionGraphProjection.objects.using(self.using)
                .filter(
                    pk=projection_id,
                    state="building",
                    collection__isnull=False,
                    artifact__isnull=False,
                    private_mapping_checksum__in=(empty_checksum, checksum),
                )
                .update(private_mapping_checksum=checksum)
            )
            if updated != 1:
                raise ValueError("projection private mapping fence was lost")
