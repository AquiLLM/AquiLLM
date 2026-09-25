from __future__ import annotations

from .identifiers import ProjectionIdentifierCodec, ProjectionIdentifierDomain
from .records import PrivateProjectionChunkReferenceV1


def encode_private_rows(
    *, snapshot: dict, codec: ProjectionIdentifierCodec
) -> tuple[PrivateProjectionChunkReferenceV1, ...]:
    generation = snapshot["projection"]["generation_key"]
    rows = tuple(
        PrivateProjectionChunkReferenceV1(
            codec.encode(
                ProjectionIdentifierDomain.CHUNK,
                generation=generation,
                source=row["id"],
            ).value,
            row["id"],
            str(row["document_id"]),
            row["chunk_number"],
        )
        for row in snapshot["chunks"]
    )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                row.projection_chunk_key,
                row.integer_chunk_pk,
                row.document_uuid,
                row.chunk_number,
            ),
        )
    )
