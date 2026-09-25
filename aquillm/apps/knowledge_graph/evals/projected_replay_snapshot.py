"""Synthetic authorized snapshots shared by offline evaluators and tests."""

from __future__ import annotations

from hashlib import sha256

from apps.knowledge_graph.retrieval import projected_types as types
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


def projected_snapshot(
    *,
    edges=(),
    nodes=None,
    iterations=2,
    max_nodes=200,
):
    """Build synthetic, authorized opaque snapshots for regression replay only."""
    if nodes is None:
        nodes = tuple({name for edge in edges for name in edge[:2]} | {"a", "b", "c"})
    weighted_edges = tuple(
        (edge[0], edge[1], float(edge[2]) if len(edge) == 3 else 1.0) for edge in edges
    )
    identities = tuple(sorted(key(name) for name in nodes))
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
    for number, (source_name, target_name, weight) in enumerate(
        weighted_edges, start=1
    ):
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
                weight,
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
        ppr_iterations=iterations,
        max_nodes=max_nodes,
        max_edges=min(1000, max_nodes * 10),
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
        types.ProjectedSnapshotCapsV1(
            64, 10_000, 128, 2, max_nodes, config.max_edges, 3_000, 3, 2
        ),
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
