from __future__ import annotations

from .identifiers import ProjectionIdentifierDomain
from .memberships import (
    MEMBERSHIP_REGISTRY_GENERATION,
    membership_decision_checksum,
    null_membership_decision_checksum,
)
from .projection_evidence_encoding import encode_evidence
from .records import (
    AutomaticCanonicalMembershipV1,
    ProjectedArtifactProvenanceV1,
    ProjectedChunkMembershipV1,
    ProjectedDocumentMembershipV1,
    ProjectedEntityV1,
    ProjectedPhysicalRelationV1,
    ProjectionCountsV1,
    ProjectionGenerationMarkerV1,
)


def encode_projection_snapshot(*, snapshot: dict, codec):
    projection = snapshot["projection"]
    generation = projection["generation_key"]

    def key(domain, source):
        return codec.encode(domain, generation=generation, source=source).value

    generation_key = key(ProjectionIdentifierDomain.COLLECTION, generation)
    collection_key = key(
        ProjectionIdentifierDomain.COLLECTION, projection["collection_id"]
    )
    artifact_key = key(ProjectionIdentifierDomain.ARTIFACT, projection["artifact_id"])
    marker = ProjectionGenerationMarkerV1(
        generation_key,
        collection_key,
        artifact_key,
        projection["schema_version"],
        projection["projection_version"],
        projection["identifier_key_version"],
        projection["membership_epoch"],
        projection["membership_checksum"],
    )
    entities, entity_keys = _entities(snapshot, key, marker)
    memberships = _memberships(
        snapshot, codec, generation, projection, entities, entity_keys
    )
    documents, document_keys = _documents(snapshot, key, generation_key)
    chunks, chunk_keys = _chunks(snapshot, key, document_keys)
    relations, relation_keys = _relations(snapshot, key, artifact_key, entity_keys)
    provenance, artifact_keys = _provenance(snapshot, key, collection_key)
    evidence = encode_evidence(
        snapshot,
        key,
        relation_keys,
        chunk_keys,
        document_keys,
        artifact_keys,
    )
    counts = ProjectionCountsV1(
        len(entities),
        len(memberships),
        len(documents),
        len(chunks),
        len(relations),
        len(evidence),
        len(provenance),
    )
    return {
        "generation": marker,
        "entities": entities,
        "automatic_memberships": memberships,
        "documents": documents,
        "chunks": chunks,
        "relations": relations,
        "evidence": evidence,
        "artifact_provenance": provenance,
        "counts": counts,
    }


def _entities(snapshot, key, marker):
    keys = {
        row["id"]: key(ProjectionIdentifierDomain.ENTITY, row["id"])
        for row in snapshot["entities"]
    }
    rows = tuple(
        sorted(
            (
                ProjectedEntityV1(
                    keys[row["id"]],
                    marker.generation_key,
                    marker.artifact_key,
                    marker.collection_key,
                    row["entity_type"],
                    key(
                        ProjectionIdentifierDomain.ENTITY,
                        f"cluster:{row['cluster_key']}",
                    ),
                    float(row["retrieval_utility"]),
                )
                for row in snapshot["entities"]
            ),
            key=lambda row: row.entity_key,
        )
    )
    return rows, keys


def _memberships(snapshot, codec, generation, projection, entities, entity_keys):
    automatic = {}
    for row in snapshot["memberships"]:
        if (row["outcome"], row["status"], row["canonical_status"]) != (
            "automatic",
            "active",
            "active",
        ):
            continue
        if row["entity_id"] in automatic:
            raise ValueError("automatic membership is not unique")
        automatic[row["entity_id"]] = (
            codec.encode(
                ProjectionIdentifierDomain.AUTOMATIC_CANONICAL_IDENTITY,
                generation=generation,
                source=row["canonical_entity_id"],
            ).value,
            row["decision_checksum"],
        )
    artifact = next(
        row for row in snapshot["artifacts"] if row["id"] == projection["artifact_id"]
    )
    source_ids = sorted(entity_keys, key=lambda value: entity_keys[value])
    audit_rows = []
    for entity, source_id in zip(entities, source_ids, strict=True):
        assignment = automatic.get(source_id)
        audit_entity_key = codec.encode(
            ProjectionIdentifierDomain.ENTITY,
            generation=MEMBERSHIP_REGISTRY_GENERATION,
            source=source_id,
        ).value
        audit_rows.append(
            AutomaticCanonicalMembershipV1(
                audit_entity_key,
                None if assignment is None else assignment[0],
                assignment[1]
                if assignment is not None
                else null_membership_decision_checksum(
                    audit_entity_key,
                    artifact["resolver_version"],
                    artifact["resolution_config_checksum"],
                ),
                artifact["resolver_version"],
                artifact["resolution_config_checksum"],
            )
        )
    audit = tuple(audit_rows)
    if membership_decision_checksum(audit) != projection["membership_checksum"]:
        raise ValueError("projection membership audit checksum is stale")
    return tuple(
        AutomaticCanonicalMembershipV1(
            entity.entity_key,
            audit_row.automatic_membership_key,
            projection["membership_checksum"],
            audit_row.resolver_version,
            audit_row.resolution_config_checksum,
        )
        for entity, audit_row in zip(entities, audit, strict=True)
    )


def _documents(snapshot, key, generation_key):
    keys = {
        row["document_id"]: key(ProjectionIdentifierDomain.DOCUMENT, row["document_id"])
        for row in snapshot["documents"]
    }
    rows = tuple(
        sorted(
            (
                ProjectedDocumentMembershipV1(value, generation_key)
                for value in keys.values()
            ),
            key=lambda row: row.document_key,
        )
    )
    return rows, keys


def _chunks(snapshot, key, document_keys):
    keys = {
        row["id"]: key(ProjectionIdentifierDomain.CHUNK, row["id"])
        for row in snapshot["chunks"]
    }
    rows = tuple(
        sorted(
            (
                ProjectedChunkMembershipV1(
                    keys[row["id"]],
                    document_keys[row["document_id"]],
                    row["chunk_number"],
                )
                for row in snapshot["chunks"]
            ),
            key=lambda row: (row.document_key, row.chunk_number, row.chunk_key),
        )
    )
    return rows, keys


def _relations(snapshot, key, artifact_key, entity_keys):
    keys = {
        row["id"]: key(ProjectionIdentifierDomain.RELATION, row["id"])
        for row in snapshot["relations"]
    }
    rows = tuple(
        sorted(
            (
                ProjectedPhysicalRelationV1(
                    keys[row["id"]],
                    artifact_key,
                    entity_keys[row["source_id"]],
                    row["relation_type"],
                    entity_keys[row["target_id"]],
                )
                for row in snapshot["relations"]
            ),
            key=lambda row: row.relation_key,
        )
    )
    return rows, keys


def _provenance(snapshot, key, collection_key):
    keys = {
        row["id"]: key(ProjectionIdentifierDomain.ARTIFACT, row["id"])
        for row in snapshot["artifacts"]
    }
    rows = []
    for row in snapshot["artifacts"]:
        scope_key = (
            collection_key
            if row["scope_type"] == "collection"
            else key(ProjectionIdentifierDomain.DOCUMENT, row["scope_id"])
        )
        rebuild = row["rebuild_request_id"]
        rows.append(
            ProjectedArtifactProvenanceV1(
                keys[row["id"]],
                row["scope_type"],
                scope_key,
                collection_key,
                None
                if rebuild is None
                else key(ProjectionIdentifierDomain.ARTIFACT, f"rebuild:{rebuild}"),
                row["evaluation_only"],
                row["build_key"],
                row["build_generation"],
                row["orchestration_version"],
                row["source_hash"],
                row["ontology_version"],
                row["ontology_checksum"],
                row["extractor_version"],
                row["resolver_version"],
                row["resolution_config_checksum"],
                row["filter_policy_version"],
                row["filter_policy_checksum"],
                row["embedding_model_signature"],
                row["assembly_version"],
                row["assembly_config_checksum"],
            )
        )
    return tuple(
        sorted(rows, key=lambda row: (row.scope_type, row.scope_key, row.artifact_key))
    ), keys
