# ruff: noqa: E501, E701, E702, I001
# fmt: off
"""Explicit legacy PostgreSQL snapshot to opaque projected parity normalizer."""
from __future__ import annotations
import json
from dataclasses import fields
from enum import StrEnum
from hashlib import sha256
from uuid import UUID
from apps.knowledge_graph.projection.identifiers import OpaqueProjectionKey, ProjectionIdentifierCodec, ProjectionIdentifierDomain as Domain
from .expansion import AuthorizedGraphSnapshot
from .ppr import canonical_algorithm_json
from .projected_types import ProjectedAlgorithmSignatureV1, ProjectedAllowedScopeV1, ProjectedArtifactProvenanceV1, ProjectedAuthorizedGraphSnapshotV1, ProjectedAutomaticMembershipAuditV1, ProjectedChunkEvidenceV1, ProjectedEvidenceOrientationV1, ProjectedEvidenceSignatureV1, ProjectedFallbackMentionAuditV1, ProjectedIdentityMentionV1, ProjectedPhysicalRelationAuditV1, ProjectedRelationEvidenceAuditV1, ProjectedRelationGroupV1, ProjectedRetrievalDirectionV1, ProjectedScopeTypeV1, ProjectedSeedIdentityV1, ProjectedSnapshotCapsV1
def _encode(codec, domain: Domain, generation: str, source) -> str:
    key = codec.encode(
        domain,
        generation=None if domain is Domain.AUTOMATIC_CANONICAL_IDENTITY else generation,
        source=source,
    )
    if type(key) is not OpaqueProjectionKey or key.domain is not domain:
        raise TypeError("codec returned an invalid opaque projection key")
    return key.value
def _identity_source(identity) -> str: return f"{identity[0]}:{identity[1]}"
def _canonical(value):
    if hasattr(value, "__dataclass_fields__"):
        return {
            item.name: _canonical(getattr(value, item.name)) for item in fields(value)
        }
    if type(value) is tuple:
        return [_canonical(item) for item in value]
    if isinstance(value, StrEnum):
        return value.value
    if type(value) is float:
        return value.hex()
    return value
def _audit_order(row) -> tuple:
    return row.discovery_hop, row.kind, json.dumps(_canonical(row), sort_keys=True, separators=(",", ":"))
def _old_evidence_key(values: tuple[object, ...]) -> str:
    decoded = list(values)
    decoded[6] = float.fromhex(decoded[6]) if type(decoded[6]) is str else decoded[6]
    encoded = json.dumps(
        decoded, default=str, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return sha256(encoded).hexdigest()
def _projected_evidence(codec, generation, values):
    evidence_key = _encode(codec, Domain.EVIDENCE, generation, values[0])
    relation_key = _encode(codec, Domain.RELATION, generation, values[1])
    chunk_key = _encode(codec, Domain.CHUNK, generation, values[3])
    document_key = _encode(codec, Domain.DOCUMENT, generation, values[4])
    confidence = (
        float.fromhex(values[6]) if type(values[6]) is str else float(values[6])
    )
    signature = ProjectedEvidenceSignatureV1(
        evidence_key,
        relation_key,
        _encode(codec, Domain.RELATION_MENTION, generation, values[2]),
        chunk_key,
        document_key,
        int(values[5]),
        confidence,
        _encode(codec, Domain.ARTIFACT, generation, values[7]),
        _encode(codec, Domain.DOCUMENT, generation, values[8]),
        _encode(codec, Domain.ENTITY_MENTION, generation, values[9]),
        _encode(codec, Domain.ENTITY_MENTION, generation, values[10]),
        values[11],
        _encode(codec, Domain.ENTITY_MAPPING, generation, values[12]),
        _encode(codec, Domain.ENTITY_MAPPING, generation, values[13]),
        ProjectedEvidenceOrientationV1(values[14]),
        values[15],
        values[16],
    )
    return signature, ProjectedChunkEvidenceV1(
        chunk_key, document_key, int(values[5]), confidence, evidence_key
    )
def _fallback(codec, generation, values, identities):
    identity = identities[tuple(values[0])]
    confidence = (
        float.fromhex(values[4]) if type(values[4]) is str else float(values[4])
    )
    evidence = ProjectedChunkEvidenceV1(
        _encode(codec, Domain.CHUNK, generation, values[1]),
        _encode(codec, Domain.DOCUMENT, generation, values[2]),
        int(values[3]),
        confidence,
        _encode(codec, Domain.EVIDENCE, generation, values[5]),
    )
    return identity, evidence
def _provenance(snapshot, codec, generation, collections, documents):
    rows = []
    for row in snapshot.artifact_provenance:
        scope_type = ProjectedScopeTypeV1(row.scope_type)
        scope_key = (
            collections[row.collection_id]
            if scope_type is ProjectedScopeTypeV1.COLLECTION
            else documents[UUID(row.scope_id)]
        )
        rows.append(
            ProjectedArtifactProvenanceV1(
                _encode(codec, Domain.ARTIFACT, generation, row.artifact_id),
                scope_type,
                scope_key,
                collections[row.collection_id],
                None
                if row.rebuild_request_id is None
                else _encode(
                    codec,
                    Domain.CANONICAL_LINK_DECISION,
                    generation,
                    row.rebuild_request_id,
                ),
                row.evaluation_only,
                row.build_key,
                row.build_generation,
                row.orchestration_version,
                row.source_hash,
                row.ontology_version,
                row.ontology_checksum,
                row.extractor_version,
                row.resolver_version,
                row.resolution_config_checksum,
                row.filter_policy_version,
                row.filter_policy_checksum,
                row.embedding_model_signature,
                row.assembly_version,
                row.assembly_config_checksum,
            )
        )
    return tuple(
        sorted(
            rows,
            key=lambda row: (row.scope_type.value, row.scope_key, row.artifact_key),
        )
    )
def project_legacy_authorized_snapshot_v1(
    *, snapshot: AuthorizedGraphSnapshot, codec: ProjectionIdentifierCodec
) -> ProjectedAuthorizedGraphSnapshotV1:
    """Normalize one already-authorized legacy snapshot without retaining IDs."""
    if type(snapshot) is not AuthorizedGraphSnapshot:
        raise TypeError("snapshot must be an exact AuthorizedGraphSnapshot")
    if not callable(getattr(codec, "encode", None)):
        raise TypeError("codec must implement ProjectionIdentifierCodec")
    generation = snapshot.scope_version_signature
    collections = {
        value: _encode(codec, Domain.COLLECTION, generation, value)
        for value in snapshot.allowed_collection_ids
    }
    documents = {
        value: _encode(codec, Domain.DOCUMENT, generation, value)
        for value in snapshot.allowed_doc_ids
    }
    identities = {
        value: _encode(
            codec,
            Domain.AUTOMATIC_CANONICAL_IDENTITY,
            generation,
            _identity_source(value),
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
                    _encode(codec, Domain.ENTITY, generation, values[1]),
                    identities[identity],
                    values[3],
                    values[4],
                )
            )
        elif kind == "physical_relation" and len(values) == 5:
            audits.append(
                ProjectedPhysicalRelationAuditV1(
                    hop,
                    _encode(codec, Domain.RELATION, generation, values[0]),
                    _encode(codec, Domain.ARTIFACT, generation, values[1]),
                    _encode(codec, Domain.ENTITY, generation, values[2]),
                    values[3],
                    _encode(codec, Domain.ENTITY, generation, values[4]),
                )
            )
        elif kind == "relation_evidence" and len(values) == 17:
            signature, evidence = _projected_evidence(codec, generation, values)
            audits.append(ProjectedRelationEvidenceAuditV1(hop, signature))
            evidence_by_legacy_key[_old_evidence_key(values)] = evidence
        elif kind == "fallback_mention" and len(values) == 6:
            identity, evidence = _fallback(codec, generation, values, identities)
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
                    _fallback(
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
                        _encode(codec, Domain.CHUNK, generation, row.seed_chunk_id),
                        identities[row.identity_key],
                    )
                    for row in snapshot.seed_identities
                ),
                key=lambda row: (row.seed_chunk_key, row.identity_key),
            )
        ),
        groups,
        mentions,
        _provenance(snapshot, codec, generation, collections, documents),
        tuple(sorted(audits, key=_audit_order)),
    )
__all__ = ["project_legacy_authorized_snapshot_v1"]
