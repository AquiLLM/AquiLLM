"""Fixtures shared by projected PPR behavior tests."""

from __future__ import annotations

from hashlib import sha256

from apps.knowledge_graph.retrieval import projected_types as types
from apps.knowledge_graph.retrieval.expansion import AuthorizedArtifactProvenance
from apps.knowledge_graph.retrieval.ppr import (
    PPRAlgorithmConfig,
    canonical_algorithm_json,
)


def key(label: str) -> str:
    return sha256(label.encode()).hexdigest()


def provenance(scope: types.ProjectedScopeTypeV1):
    collection = key("collection")
    is_collection = scope is types.ProjectedScopeTypeV1.COLLECTION
    return types.ProjectedArtifactProvenanceV1(
        key(f"artifact-{scope}"),
        scope,
        collection if is_collection else key("document"),
        collection,
        None,
        False,
        key(f"build-{scope}"),
        1,
        1,
        key(f"source-{scope}"),
        "ontology-v1",
        key("ontology"),
        "extractor-v1",
        "resolver-v1",
        key("resolution"),
        "filter-v1",
        key("filter"),
        "embed-v1" if is_collection else "",
        "assembly-v1",
        key("assembly"),
    )


def projected_snapshot(*, edges: tuple[tuple[str, str], ...] = ()):
    identities = tuple(
        sorted(
            {key(name) for edge in edges for name in edge}
            | {key("a"), key("b"), key("c")}
        )
    )
    entity_by_identity = {
        identity: key(f"entity-{identity}") for identity in identities
    }
    memberships = tuple(
        types.ProjectedAutomaticMembershipAuditV1(
            0,
            entity_by_identity[identity],
            identity,
            key(f"decision-{identity}"),
            "resolver-v1",
        )
        for identity in identities
    )
    groups = []
    audits: list[object] = list(memberships)
    for number, (source_name, target_name) in enumerate(edges, start=1):
        source, target = key(source_name), key(target_name)
        relation = key(f"relation-{number}")
        evidence = key(f"evidence-{number}")
        chunk = types.ProjectedChunkEvidenceV1(
            key(f"chunk-{number}"), key("document"), number, 1.0, evidence
        )
        signature = types.ProjectedEvidenceSignatureV1(
            evidence,
            relation,
            key(f"mention-{number}"),
            chunk.chunk_key,
            chunk.document_key,
            number,
            1.0,
            key(f"artifact-{types.ProjectedScopeTypeV1.DOCUMENT}"),
            chunk.document_key,
            key(f"head-{number}"),
            key(f"tail-{number}"),
            "related_to",
            key(f"head-map-{number}"),
            key(f"tail-map-{number}"),
            types.ProjectedEvidenceOrientationV1.HEAD_TO_TAIL,
            key("ontology"),
            key("assembly"),
        )
        groups.append(
            types.ProjectedRelationGroupV1(
                source,
                "related_to",
                target,
                types.ProjectedRetrievalDirectionV1.FORWARD,
                1.0,
                1,
                (chunk,),
            )
        )
        audits.extend(
            (
                types.ProjectedPhysicalRelationAuditV1(
                    1,
                    relation,
                    key(f"artifact-{types.ProjectedScopeTypeV1.COLLECTION}"),
                    entity_by_identity[source],
                    "related_to",
                    entity_by_identity[target],
                ),
                types.ProjectedRelationEvidenceAuditV1(1, signature),
            )
        )
    audits.sort(key=lambda row: (row.discovery_hop, row.kind, types._canonical(row)))
    groups.sort(
        key=lambda row: (
            row.source_identity_key,
            row.relation_type,
            row.target_identity_key,
            row.direction.value,
        )
    )
    config = PPRAlgorithmConfig(
        canonical_resolver_version="resolver-v1",
        ppr_iterations=2,
    )
    return types.ProjectedAuthorizedGraphSnapshotV1(
        types.ProjectedAlgorithmSignatureV1(
            "ppr_projected_v1",
            "ppr_transition_v1",
            "ppr_evidence_v1",
            "rrf_seed_v1",
            sha256(
                b"ppr_projected_v1\0" + canonical_algorithm_json(config)
            ).hexdigest(),
        ),
        types.ProjectedSnapshotCapsV1(64, 10_000, 128, 2, 200, 1_000, 3_000, 3, 2),
        2,
        types.ProjectedAllowedScopeV1(
            (key("document"),),
            (key("collection"),),
            key("scope"),
        ),
        identities,
        (),
        tuple(groups),
        (),
        (
            provenance(types.ProjectedScopeTypeV1.COLLECTION),
            provenance(types.ProjectedScopeTypeV1.DOCUMENT),
        ),
        tuple(audits),
    ), config


def legacy_provenance(scope_type: str, scope_id: str, collection_id: int):
    is_collection = scope_type == "collection"
    return AuthorizedArtifactProvenance(
        10 if is_collection else 20,
        scope_type,
        scope_id,
        collection_id,
        None,
        False,
        key(f"legacy-build-{scope_type}"),
        1,
        1,
        key(f"legacy-source-{scope_type}"),
        "ontology-v1",
        key("legacy-ontology"),
        "extractor-v1",
        "resolver-v1",
        key("legacy-resolution"),
        "filter-v1",
        key("legacy-filter"),
        "embed-v1" if is_collection else "",
        "assembly-v1",
        key("legacy-assembly"),
    )
