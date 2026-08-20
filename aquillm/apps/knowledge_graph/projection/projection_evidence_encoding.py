from __future__ import annotations

import json
from hashlib import sha256

from .identifiers import ProjectionIdentifierDomain
from .records import ProjectedRelationEvidenceV1


def encode_evidence(
    snapshot, key, relation_keys, chunk_keys, document_keys, artifact_keys
):
    rows = []
    for row in snapshot["evidence"]:
        semantic = sha256(
            json.dumps(
                row,
                default=str,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        rows.append(
            ProjectedRelationEvidenceV1(
                key(ProjectionIdentifierDomain.EVIDENCE, row["id"]),
                relation_keys[row["relation_id"]],
                key(
                    ProjectionIdentifierDomain.RELATION_MENTION,
                    row["relation_mention_id"],
                ),
                chunk_keys[row["chunk_id"]],
                document_keys[row["document_id"]],
                row["chunk_number"],
                float(row["confidence"]),
                artifact_keys[row["artifact_id"]],
                document_keys[row["document_id"]],
                key(
                    ProjectionIdentifierDomain.ENTITY_MENTION,
                    row["head_mention_id"],
                ),
                key(
                    ProjectionIdentifierDomain.ENTITY_MENTION,
                    row["tail_mention_id"],
                ),
                key(
                    ProjectionIdentifierDomain.ENTITY_MAPPING,
                    row["head_mapping_id"],
                ),
                key(
                    ProjectionIdentifierDomain.ENTITY_MAPPING,
                    row["tail_mapping_id"],
                ),
                row["orientation"],
                row["relation_type"],
                row["ontology_checksum"],
                row["assembly_config_checksum"],
                key(ProjectionIdentifierDomain.EVIDENCE, f"provenance:{semantic}"),
                semantic,
            )
        )
    return tuple(sorted(rows, key=lambda row: row.evidence_key))
