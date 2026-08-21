"""Encoding helpers for projected topology snapshot assembly."""

from __future__ import annotations

from hashlib import sha256

from apps.knowledge_graph.retrieval import projected_types as t
from apps.knowledge_graph.retrieval.ppr import (
    PPRAlgorithmConfig,
    canonical_algorithm_json,
)


def provenance(row):
    return t.ProjectedArtifactProvenanceV1(
        row.artifact_key,
        t.ProjectedScopeTypeV1(row.scope_type),
        row.scope_key,
        row.collection_key,
        row.rebuild_request_key,
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


def chunk_evidence(row, provenance_key: str | None = None):
    return t.ProjectedChunkEvidenceV1(
        row.chunk_key,
        row.document_key,
        row.chunk_number,
        row.confidence,
        row.evidence_key if provenance_key is None else provenance_key,
    )


def evidence_signature(row):
    return t.ProjectedEvidenceSignatureV1(
        row.evidence_key,
        row.relation_key,
        row.relation_mention_key,
        row.chunk_key,
        row.document_key,
        row.chunk_number,
        row.confidence,
        row.artifact_key,
        row.source_document_key,
        row.head_mention_key,
        row.tail_mention_key,
        row.relation_type,
        row.head_mapping_key,
        row.tail_mapping_key,
        t.ProjectedEvidenceOrientationV1(row.orientation),
        row.ontology_checksum,
        row.assembly_config_checksum,
    )


def algorithm(caps, resolver_version: str):
    config = PPRAlgorithmConfig(
        canonical_resolver_version=resolver_version,
        max_seeds=caps.max_seeds,
        max_hops=caps.max_depth,
        max_nodes=caps.max_nodes,
        max_edges=caps.max_edges,
        max_candidates=caps.max_results,
        max_per_document=min(3, caps.max_results),
    )
    signature = sha256(
        b"ppr_projected_v1\0" + canonical_algorithm_json(config)
    ).hexdigest()
    return (
        t.ProjectedAlgorithmSignatureV1(
            "ppr_projected_v1",
            "ppr_transition_v1",
            "ppr_evidence_v1",
            "rrf_seed_v1",
            signature,
        ),
        t.ProjectedSnapshotCapsV1(
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
    )


__all__ = ["algorithm", "chunk_evidence", "evidence_signature", "provenance"]
