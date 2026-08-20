# ruff: noqa: E501, E701, E702, I001
# fmt: off
from __future__ import annotations
import json
from dataclasses import replace
from hashlib import sha256
from uuid import UUID
import pytest
from apps.knowledge_graph.projection.identifiers import HmacSha256ProjectionIdentifierCodec, ProjectionIdentifierDomain
from apps.knowledge_graph.retrieval import projected_types as types
from apps.knowledge_graph.retrieval.expansion import AuthorizedArtifactProvenance, AuthorizedChunkEvidence, AuthorizedGraphSnapshot, AuthorizedIdentityMention, AuthorizedRelationGroup, AuthorizedSeedIdentity
from apps.knowledge_graph.retrieval.ppr import PPRAlgorithmConfig, RetrievalDirection, personalized_pagerank
from apps.knowledge_graph.retrieval.projected_ppr import ppr_projected_v1
from apps.knowledge_graph.retrieval.projected_snapshot import project_legacy_authorized_snapshot_v1
from apps.knowledge_graph.retrieval.topology.contracts import ProjectedSeedV1
def _key(label: str) -> str: return sha256(label.encode()).hexdigest()
def _provenance(scope: types.ProjectedScopeTypeV1):
    collection = _key("collection")
    is_collection = scope is types.ProjectedScopeTypeV1.COLLECTION
    return types.ProjectedArtifactProvenanceV1(
        _key(f"artifact-{scope}"),
        scope,
        collection if is_collection else _key("document"),
        collection,
        None,
        False,
        _key(f"build-{scope}"),
        1,
        1,
        _key(f"source-{scope}"),
        "ontology-v1",
        _key("ontology"),
        "extractor-v1",
        "resolver-v1",
        _key("resolution"),
        "filter-v1",
        _key("filter"),
        "embed-v1" if is_collection else "",
        "assembly-v1",
        _key("assembly"),
    )
def _projected_snapshot(*, edges: tuple[tuple[str, str], ...] = ()):
    identities = tuple(
        sorted(
            {_key(name) for edge in edges for name in edge}
            | {_key("a"), _key("b"), _key("c")}
        )
    )
    entity_by_identity = {
        identity: _key(f"entity-{identity}") for identity in identities
    }
    memberships = tuple(
        types.ProjectedAutomaticMembershipAuditV1(
            0,
            entity_by_identity[identity],
            identity,
            _key(f"decision-{identity}"),
            "resolver-v1",
        )
        for identity in identities
    )
    groups = []
    audits: list[object] = list(memberships)
    for number, (source_name, target_name) in enumerate(edges, start=1):
        source, target = _key(source_name), _key(target_name)
        relation, evidence = _key(f"relation-{number}"), _key(f"evidence-{number}")
        chunk = types.ProjectedChunkEvidenceV1(
            _key(f"chunk-{number}"), _key("document"), number, 1.0, evidence
        )
        signature = types.ProjectedEvidenceSignatureV1(
            evidence,
            relation,
            _key(f"mention-{number}"),
            chunk.chunk_key,
            chunk.document_key,
            number,
            1.0,
            _key(f"artifact-{types.ProjectedScopeTypeV1.DOCUMENT}"),
            chunk.document_key,
            _key(f"head-{number}"),
            _key(f"tail-{number}"),
            "related_to",
            _key(f"head-map-{number}"),
            _key(f"tail-map-{number}"),
            types.ProjectedEvidenceOrientationV1.HEAD_TO_TAIL,
            _key("ontology"),
            _key("assembly"),
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
                    _key(f"artifact-{types.ProjectedScopeTypeV1.COLLECTION}"),
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
    config = PPRAlgorithmConfig(canonical_resolver_version="resolver-v1", ppr_iterations=2)
    return types.ProjectedAuthorizedGraphSnapshotV1(
        types.ProjectedAlgorithmSignatureV1(
            "ppr_projected_v1",
            "ppr_transition_v1",
            "ppr_evidence_v1",
            "rrf_seed_v1",
            _key("algorithm"),
        ),
        types.ProjectedSnapshotCapsV1(64, 10_000, 128, 2, 200, 1_000, 3_000, 3, 2),
        2,
        types.ProjectedAllowedScopeV1(
            (_key("document"),), (_key("collection"),), _key("scope")
        ),
        identities,
        (),
        tuple(groups),
        (),
        (
            _provenance(types.ProjectedScopeTypeV1.COLLECTION),
            _provenance(types.ProjectedScopeTypeV1.DOCUMENT),
        ),
        tuple(audits),
    ), config
def test_projected_ppr_hand_recurrence_zero_hop_and_relation_hops() -> None:
    snapshot, config = _projected_snapshot(edges=(("a", "b"), ("b", "c")))
    result = ppr_projected_v1(
        snapshot=snapshot, seeds=(ProjectedSeedV1(_key("a"), 1.0),), config=config
    )
    assert dict(result.scores) == {
        _key("a"): pytest.approx(0.2),
        _key("b"): pytest.approx(0.16),
        _key("c"): pytest.approx(0.64),
    }
    permuted, _ = _projected_snapshot(edges=(("b", "c"), ("a", "b")))
    assert (
        ppr_projected_v1(
            snapshot=permuted,
            seeds=(ProjectedSeedV1(_key("a"), 1.0),),
            config=config,
        ).trace_bytes
        == result.trace_bytes
    )
    isolated, config = _projected_snapshot()
    zero_hop = ppr_projected_v1(
        snapshot=isolated, seeds=(ProjectedSeedV1(_key("a"), 1.0),), config=config
    )
    assert dict(zero_hop.scores) == {_key("a"): 1.0, _key("b"): 0.0, _key("c"): 0.0}
def test_projected_ppr_uses_opaque_order_for_ties_and_stable_trace() -> None:
    snapshot, config = _projected_snapshot()
    first = ppr_projected_v1(
        snapshot=snapshot, seeds=(ProjectedSeedV1(_key("a"), 1.0),), config=config
    )
    second = ppr_projected_v1(
        snapshot=replace(snapshot),
        seeds=(ProjectedSeedV1(_key("a"), 1.0),),
        config=config,
    )
    assert first == second
    tied = tuple(key for key in first.ranked_identity_keys if key != _key("a"))
    assert tied == tuple(sorted((_key("b"), _key("c"))))
    assert first.trace_bytes == second.trace_bytes
    assert sha256(first.trace_bytes).hexdigest() == (
        "fb60307c60bdc5e26ec890858c97412ebfb4fd4590ce3e564dccb9557a5c8b9e"
    )
def _legacy_provenance(scope_type: str, scope_id: str, collection_id: int):
    is_collection = scope_type == "collection"
    return AuthorizedArtifactProvenance(
        10 if is_collection else 20,
        scope_type,
        scope_id,
        collection_id,
        None,
        False,
        _key(f"legacy-build-{scope_type}"),
        1,
        1,
        _key(f"legacy-source-{scope_type}"),
        "ontology-v1",
        _key("legacy-ontology"),
        "extractor-v1",
        "resolver-v1",
        _key("legacy-resolution"),
        "filter-v1",
        _key("legacy-filter"),
        "embed-v1" if is_collection else "",
        "assembly-v1",
        _key("legacy-assembly"),
    )
def test_legacy_projection_is_closed_opaque_and_does_not_use_db_order() -> None:
    document = UUID("11111111-1111-4111-8111-111111111111")
    config = PPRAlgorithmConfig(
        canonical_resolver_version="resolver-v1", ppr_iterations=2
    )
    snapshot = AuthorizedGraphSnapshot(
        config,
        2,
        (document,),
        (1,),
        _key("legacy-scope"),
        (("canonical", 7), ("local", _key("component-b"))),
        (
            AuthorizedSeedIdentity(99, ("canonical", 7)),
            AuthorizedSeedIdentity(100, ("local", _key("component-b"))),
        ),
        (),
        (),
        (
            _legacy_provenance("collection", "1", 1),
            _legacy_provenance("document", str(document), 1),
        ),
        (
            (0, "canonical_link", (91, 101, 7, _key("decision-a"), "resolver-v1")),
            (
                0,
                "canonical_link",
                (92, 102, _key("component-b"), _key("decision-b"), "resolver-v1"),
            ),
        ),
    )
    codec = HmacSha256ProjectionIdentifierCodec(b"key0", key_version="key-v1")
    projected = project_legacy_authorized_snapshot_v1(snapshot=snapshot, codec=codec)
    neutral = types.canonical_projected_snapshot_bytes(projected)
    assert b'"101"' not in neutral and b'"102"' not in neutral
    assert str(document).encode() not in neutral
    assert len(projected.identity_keys) == 2
    seeds = tuple(
        sorted(
            (
                ProjectedSeedV1(row.identity_key, 0.5)
                for row in projected.seed_identities
            ),
            key=lambda row: row.identity_key,
        )
    )
    result = ppr_projected_v1(snapshot=projected, seeds=seeds, config=config)
    assert tuple(result.ranked_identity_keys) == tuple(sorted(projected.identity_keys))
    legacy = personalized_pagerank(
        {("canonical", 7): 0.5, ("local", _key("component-b")): 0.5},
        {},
        restart_probability=config.ppr_restart,
        iterations=config.ppr_iterations,
    )
    assert sorted(dict(result.scores).values()) == sorted(legacy.values())
    canonical_key = codec.encode(
        ProjectionIdentifierDomain.AUTOMATIC_CANONICAL_IDENTITY,
        source="canonical:7",
    ).value
    assert result.ranked_identity_keys[0] != canonical_key
    signature = (
        301, 201, 401, 501, document, 2, 0.75, 20, document, 601, 602,
        "related_to", 701, 702, "head_to_tail", _key("legacy-ontology"),
        _key("legacy-assembly"),
    )
    provenance_key = sha256(json.dumps(
        signature, default=str, ensure_ascii=True, separators=(",", ":"),
        allow_nan=False,
    ).encode()).hexdigest()
    evidence = AuthorizedChunkEvidence(501, document, 2, 0.75, provenance_key)
    fallback = AuthorizedChunkEvidence(502, document, 3, 0.25, _key("fallback"))
    graph_snapshot = replace(
        snapshot,
        relation_groups=(AuthorizedRelationGroup(
            ("canonical", 7), "related_to", ("local", _key("component-b")),
            RetrievalDirection.FORWARD, 1.0, 1, (evidence,),
        ),),
        mentions=(AuthorizedIdentityMention(("local", _key("component-b")), fallback),),
        raw_audit_rows=(*snapshot.raw_audit_rows,
            (0, "fallback_mention", (("local", _key("component-b")), 502,
             str(document), 3, 0.25.hex(), _key("fallback"))),
            (1, "physical_relation", (201, 10, 101, "related_to", 102)),
            (1, "relation_evidence", signature),
        ),
    )
    normalized = project_legacy_authorized_snapshot_v1(
        snapshot=graph_snapshot, codec=codec
    )
    assert len(normalized.relation_groups) == 1
    assert len(normalized.mentions) == 1
