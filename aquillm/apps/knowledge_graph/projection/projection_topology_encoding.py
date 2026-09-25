from __future__ import annotations

from .identifiers import ProjectionIdentifierDomain
from .records import (
    ProjectedEntityMentionEvidenceV1,
    ProjectedRelationSemanticsV1,
)


def encode_relation_semantics(snapshot, key, artifact_key):
    values = {
        (row["relation_type"], row["direction"]) for row in snapshot["relations"]
    }
    if len({relation_type for relation_type, _direction in values}) != len(values):
        raise ValueError("relation type direction is inconsistent")
    return tuple(
        ProjectedRelationSemanticsV1(
            key(ProjectionIdentifierDomain.RELATION, f"semantics:{relation_type}"),
            artifact_key,
            relation_type,
            direction,
        )
        for relation_type, direction in sorted(values)
    )


def encode_entity_mentions(snapshot, key, entity_keys, chunk_keys, document_keys):
    rows = []
    for row in snapshot["entity_mentions"]:
        source = f"mention:{row['mention_id']}:chunk:{row['chunk_id']}"
        rows.append(
            ProjectedEntityMentionEvidenceV1(
                key(ProjectionIdentifierDomain.ENTITY_MENTION, source),
                key(ProjectionIdentifierDomain.EVIDENCE, f"entity-{source}"),
                entity_keys[row["entity_id"]],
                chunk_keys[row["chunk_id"]],
                document_keys[row["document_id"]],
                row["chunk_number"],
                float(row["confidence"]),
            )
        )
    return tuple(sorted(rows, key=lambda row: (row.entity_key, row.provenance_key)))
