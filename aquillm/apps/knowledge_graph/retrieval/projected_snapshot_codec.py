"""Opaque-key and audit conversion helpers for legacy snapshot projection."""

from __future__ import annotations

import json
from dataclasses import fields
from enum import StrEnum
from hashlib import sha256
from uuid import UUID

from apps.knowledge_graph.projection.identifiers import (
    OpaqueProjectionKey,
    ProjectionIdentifierCodec,
    ProjectionIdentifierDomain,
)

from .projected_types import (
    ProjectedArtifactProvenanceV1,
    ProjectedChunkEvidenceV1,
    ProjectedEvidenceOrientationV1,
    ProjectedEvidenceSignatureV1,
    ProjectedScopeTypeV1,
)


def encode_opaque(
    codec: ProjectionIdentifierCodec,
    domain: ProjectionIdentifierDomain,
    generation: str,
    source: object,
) -> str:
    key = codec.encode(
        domain,
        generation=(
            None
            if domain is ProjectionIdentifierDomain.AUTOMATIC_CANONICAL_IDENTITY
            else generation
        ),
        source=source,
    )
    if type(key) is not OpaqueProjectionKey or key.domain is not domain:
        raise TypeError("codec returned an invalid opaque projection key")
    return key.value


def identity_source(identity) -> str:
    return f"{identity[0]}:{identity[1]}"


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


def audit_order(row) -> tuple:
    return (
        row.discovery_hop,
        row.kind,
        json.dumps(_canonical(row), sort_keys=True, separators=(",", ":")),
    )


def old_evidence_key(values: tuple[object, ...]) -> str:
    decoded = list(values)
    if type(decoded[6]) is str:
        decoded[6] = float.fromhex(decoded[6])
    encoded = json.dumps(
        decoded,
        default=str,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return sha256(encoded).hexdigest()


def projected_evidence(codec, generation, values):
    domain = ProjectionIdentifierDomain
    evidence_key = encode_opaque(codec, domain.EVIDENCE, generation, values[0])
    relation_key = encode_opaque(codec, domain.RELATION, generation, values[1])
    chunk_key = encode_opaque(codec, domain.CHUNK, generation, values[3])
    document_key = encode_opaque(codec, domain.DOCUMENT, generation, values[4])
    confidence = (
        float.fromhex(values[6]) if type(values[6]) is str else float(values[6])
    )
    signature = ProjectedEvidenceSignatureV1(
        evidence_key,
        relation_key,
        encode_opaque(codec, domain.RELATION_MENTION, generation, values[2]),
        chunk_key,
        document_key,
        int(values[5]),
        confidence,
        encode_opaque(codec, domain.ARTIFACT, generation, values[7]),
        encode_opaque(codec, domain.DOCUMENT, generation, values[8]),
        encode_opaque(codec, domain.ENTITY_MENTION, generation, values[9]),
        encode_opaque(codec, domain.ENTITY_MENTION, generation, values[10]),
        values[11],
        encode_opaque(codec, domain.ENTITY_MAPPING, generation, values[12]),
        encode_opaque(codec, domain.ENTITY_MAPPING, generation, values[13]),
        ProjectedEvidenceOrientationV1(values[14]),
        values[15],
        values[16],
    )
    return signature, ProjectedChunkEvidenceV1(
        chunk_key,
        document_key,
        int(values[5]),
        confidence,
        evidence_key,
    )


def projected_fallback(codec, generation, values, identities):
    domain = ProjectionIdentifierDomain
    identity = identities[tuple(values[0])]
    confidence = (
        float.fromhex(values[4]) if type(values[4]) is str else float(values[4])
    )
    evidence = ProjectedChunkEvidenceV1(
        encode_opaque(codec, domain.CHUNK, generation, values[1]),
        encode_opaque(codec, domain.DOCUMENT, generation, values[2]),
        int(values[3]),
        confidence,
        encode_opaque(codec, domain.EVIDENCE, generation, values[5]),
    )
    return identity, evidence


def projected_provenance(snapshot, codec, generation, collections, documents):
    domain = ProjectionIdentifierDomain
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
                encode_opaque(codec, domain.ARTIFACT, generation, row.artifact_id),
                scope_type,
                scope_key,
                collections[row.collection_id],
                (
                    None
                    if row.rebuild_request_id is None
                    else encode_opaque(
                        codec,
                        domain.CANONICAL_LINK_DECISION,
                        generation,
                        row.rebuild_request_id,
                    )
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


__all__ = [
    "audit_order",
    "encode_opaque",
    "identity_source",
    "old_evidence_key",
    "projected_evidence",
    "projected_fallback",
    "projected_provenance",
]
