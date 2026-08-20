"""Normalize a legacy authorized snapshot into closed opaque-key topology."""

from __future__ import annotations

from hashlib import sha256

from apps.knowledge_graph.projection.identifiers import (
    ProjectionIdentifierCodec,
    ProjectionIdentifierDomain,
)

from .expansion import AuthorizedGraphSnapshot
from .ppr import canonical_algorithm_json
from .projected_snapshot_codec import (
    audit_order,
    encode_opaque,
    identity_source,
    old_evidence_key,
    projected_evidence,
    projected_fallback,
    projected_provenance,
)
from .projected_types import (
    ProjectedAlgorithmSignatureV1,
    ProjectedAllowedScopeV1,
    ProjectedAuthorizedGraphSnapshotV1,
    ProjectedAutomaticMembershipAuditV1,
    ProjectedFallbackMentionAuditV1,
    ProjectedIdentityMentionV1,
    ProjectedPhysicalRelationAuditV1,
    ProjectedRelationEvidenceAuditV1,
    ProjectedRelationGroupV1,
    ProjectedRetrievalDirectionV1,
    ProjectedSeedIdentityV1,
    ProjectedSnapshotCapsV1,
)


def project_legacy_authorized_snapshot_v1(
    *, snapshot: AuthorizedGraphSnapshot, codec: ProjectionIdentifierCodec
) -> ProjectedAuthorizedGraphSnapshotV1:
    """Normalize one already-authorized legacy snapshot without retaining IDs."""

    if type(snapshot) is not AuthorizedGraphSnapshot:
        raise TypeError("snapshot must be an exact AuthorizedGraphSnapshot")
    if not callable(getattr(codec, "encode", None)):
        raise TypeError("codec must implement ProjectionIdentifierCodec")
    domain = ProjectionIdentifierDomain
    generation = snapshot.scope_version_signature
    collections = {
        value: encode_opaque(codec, domain.COLLECTION, generation, value)
        for value in snapshot.allowed_collection_ids
    }
    documents = {
        value: encode_opaque(codec, domain.DOCUMENT, generation, value)
        for value in snapshot.allowed_doc_ids
    }
    identities = {
        value: encode_opaque(
            codec,
            domain.AUTOMATIC_CANONICAL_IDENTITY,
            generation,
            identity_source(value),
        )
        for value in snapshot.identity_keys
    }
    audits: list[object] = []
    evidence_by_legacy_key = {}
    for hop, kind, values in snapshot.raw_audit_rows:
        if kind == "canonical_link" and len(values) == 5:
            identity = (
                ("canonical", values[2])
                if type(values[2]) is int
                else ("local", values[2])
            )
            audits.append(
                ProjectedAutomaticMembershipAuditV1(
                    hop,
                    encode_opaque(codec, domain.ENTITY, generation, values[1]),
                    identities[identity],
                    values[3],
                    values[4],
                )
            )
        elif kind == "physical_relation" and len(values) == 5:
            audits.append(
                ProjectedPhysicalRelationAuditV1(
                    hop,
                    encode_opaque(codec, domain.RELATION, generation, values[0]),
                    encode_opaque(codec, domain.ARTIFACT, generation, values[1]),
                    encode_opaque(codec, domain.ENTITY, generation, values[2]),
                    values[3],
                    encode_opaque(codec, domain.ENTITY, generation, values[4]),
                )
            )
        elif kind == "relation_evidence" and len(values) == 17:
            signature, evidence = projected_evidence(codec, generation, values)
            audits.append(ProjectedRelationEvidenceAuditV1(hop, signature))
            evidence_by_legacy_key[old_evidence_key(values)] = evidence
        elif kind == "fallback_mention" and len(values) == 6:
            identity, evidence = projected_fallback(
                codec, generation, values, identities
            )
            audits.append(ProjectedFallbackMentionAuditV1(hop, identity, evidence))
        else:
            raise ValueError("legacy audit rows are not a closed projected family")
    groups = tuple(
        sorted(
            (
                ProjectedRelationGroupV1(
                    identities[row.source_key],
                    row.relation_type,
                    identities[row.target_key],
                    ProjectedRetrievalDirectionV1(row.direction.value),
                    row.raw_weight,
                    row.admission_hop,
                    tuple(
                        sorted(
                            (
                                evidence_by_legacy_key[item.provenance_key]
                                for item in row.evidence
                            ),
                            key=lambda item: (item.provenance_key, item.chunk_key),
                        )
                    ),
                )
                for row in snapshot.relation_groups
            ),
            key=lambda row: (
                row.source_identity_key,
                row.relation_type,
                row.target_identity_key,
                row.direction.value,
            ),
        )
    )
    mentions = tuple(
        sorted(
            (
                ProjectedIdentityMentionV1(
                    identities[row.identity_key],
                    projected_fallback(
                        codec,
                        generation,
                        (
                            row.identity_key,
                            row.evidence.chunk_id,
                            str(row.evidence.document_id),
                            row.evidence.chunk_number,
                            row.evidence.confidence.hex(),
                            row.evidence.provenance_key,
                        ),
                        identities,
                    )[1],
                )
                for row in snapshot.mentions
            ),
            key=lambda row: (row.identity_key, row.evidence.provenance_key),
        )
    )
    config = snapshot.config
    algorithm_signature = sha256(
        b"ppr_projected_v1\0" + canonical_algorithm_json(config)
    ).hexdigest()
    return ProjectedAuthorizedGraphSnapshotV1(
        ProjectedAlgorithmSignatureV1(
            "ppr_projected_v1",
            "ppr_transition_v1",
            "ppr_evidence_v1",
            "rrf_seed_v1",
            algorithm_signature,
        ),
        ProjectedSnapshotCapsV1(
            config.max_seeds,
            config.max_scope_documents,
            config.max_scope_collections,
            config.max_hops,
            config.max_nodes,
            config.max_edges,
            config.max_evidence_rows,
            config.max_evidence_per_edge,
            config.max_mentions_per_entity,
        ),
        snapshot.load_max_hops,
        ProjectedAllowedScopeV1(
            tuple(sorted(documents.values())),
            tuple(sorted(collections.values())),
            snapshot.scope_version_signature,
        ),
        tuple(sorted(identities.values())),
        tuple(
            sorted(
                (
                    ProjectedSeedIdentityV1(
                        encode_opaque(
                            codec, domain.CHUNK, generation, row.seed_chunk_id
                        ),
                        identities[row.identity_key],
                    )
                    for row in snapshot.seed_identities
                ),
                key=lambda row: (row.seed_chunk_key, row.identity_key),
            )
        ),
        groups,
        mentions,
        projected_provenance(snapshot, codec, generation, collections, documents),
        tuple(sorted(audits, key=audit_order)),
    )


__all__ = ["project_legacy_authorized_snapshot_v1"]
