"""Projected opaque-key PPR and legacy normalization tests."""

from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from uuid import UUID

import pytest

from apps.knowledge_graph.projection.identifiers import (
    HmacSha256ProjectionIdentifierCodec,
    ProjectionIdentifierDomain,
)
from apps.knowledge_graph.retrieval import projected_types as types
from apps.knowledge_graph.retrieval.expansion import (
    AuthorizedChunkEvidence,
    AuthorizedGraphSnapshot,
    AuthorizedIdentityMention,
    AuthorizedRelationGroup,
    AuthorizedSeedIdentity,
)
from apps.knowledge_graph.retrieval.ppr import (
    PPRAlgorithmConfig,
    RetrievalDirection,
    personalized_pagerank,
)
from apps.knowledge_graph.retrieval.projected_ppr import ppr_projected_v1
from apps.knowledge_graph.retrieval.projected_snapshot import (
    project_legacy_authorized_snapshot_v1,
)
from apps.knowledge_graph.retrieval.topology.contracts import ProjectedSeedV1
from apps.knowledge_graph.tests.projected_ppr_fixtures import (
    key,
    legacy_provenance,
    projected_snapshot,
)


def test_projected_ppr_hand_recurrence_zero_hop_and_relation_hops() -> None:
    snapshot, config = projected_snapshot(edges=(("a", "b"), ("b", "c")))
    result = ppr_projected_v1(
        snapshot=snapshot,
        seeds=(ProjectedSeedV1(key("a"), 1.0),),
        config=config,
    )
    assert dict(result.scores) == {
        key("a"): pytest.approx(0.2),
        key("b"): pytest.approx(0.16),
        key("c"): pytest.approx(0.64),
    }
    permuted, _ = projected_snapshot(edges=(("b", "c"), ("a", "b")))
    assert (
        ppr_projected_v1(
            snapshot=permuted,
            seeds=(ProjectedSeedV1(key("a"), 1.0),),
            config=config,
        ).trace_bytes
        == result.trace_bytes
    )
    isolated, config = projected_snapshot()
    zero_hop = ppr_projected_v1(
        snapshot=isolated,
        seeds=(ProjectedSeedV1(key("a"), 1.0),),
        config=config,
    )
    assert dict(zero_hop.scores) == {
        key("a"): 1.0,
        key("b"): 0.0,
        key("c"): 0.0,
    }


def test_projected_ppr_uses_opaque_order_for_ties_and_stable_trace() -> None:
    snapshot, config = projected_snapshot()
    first = ppr_projected_v1(
        snapshot=snapshot,
        seeds=(ProjectedSeedV1(key("a"), 1.0),),
        config=config,
    )
    second = ppr_projected_v1(
        snapshot=replace(snapshot),
        seeds=(ProjectedSeedV1(key("a"), 1.0),),
        config=config,
    )
    assert first == second
    tied = tuple(item for item in first.ranked_identity_keys if item != key("a"))
    assert tied == tuple(sorted((key("b"), key("c"))))
    assert first.trace_bytes == second.trace_bytes
    assert sha256(first.trace_bytes).hexdigest() == (
        "fb60307c60bdc5e26ec890858c97412ebfb4fd4590ce3e564dccb9557a5c8b9e"
    )


def test_legacy_projection_is_closed_opaque_and_does_not_use_db_order() -> None:
    document = UUID("11111111-1111-4111-8111-111111111111")
    config = PPRAlgorithmConfig(
        canonical_resolver_version="resolver-v1",
        ppr_iterations=2,
    )
    snapshot = AuthorizedGraphSnapshot(
        config,
        2,
        (document,),
        (1,),
        key("legacy-scope"),
        (("canonical", 7), ("local", key("component-b"))),
        (
            AuthorizedSeedIdentity(99, ("canonical", 7)),
            AuthorizedSeedIdentity(100, ("local", key("component-b"))),
        ),
        (),
        (),
        (
            legacy_provenance("collection", "1", 1),
            legacy_provenance("document", str(document), 1),
        ),
        (
            (0, "canonical_link", (91, 101, 7, key("decision-a"), "resolver-v1")),
            (
                0,
                "canonical_link",
                (92, 102, key("component-b"), key("decision-b"), "resolver-v1"),
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
        {("canonical", 7): 0.5, ("local", key("component-b")): 0.5},
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
        301,
        201,
        401,
        501,
        document,
        2,
        0.75,
        20,
        document,
        601,
        602,
        "related_to",
        701,
        702,
        "head_to_tail",
        key("legacy-ontology"),
        key("legacy-assembly"),
    )
    provenance_key = sha256(
        json.dumps(
            signature,
            default=str,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    evidence = AuthorizedChunkEvidence(501, document, 2, 0.75, provenance_key)
    fallback = AuthorizedChunkEvidence(502, document, 3, 0.25, key("fallback"))
    graph_snapshot = replace(
        snapshot,
        relation_groups=(
            AuthorizedRelationGroup(
                ("canonical", 7),
                "related_to",
                ("local", key("component-b")),
                RetrievalDirection.FORWARD,
                1.0,
                1,
                (evidence,),
            ),
        ),
        mentions=(
            AuthorizedIdentityMention(
                ("local", key("component-b")),
                fallback,
            ),
        ),
        raw_audit_rows=(
            *snapshot.raw_audit_rows,
            (
                0,
                "fallback_mention",
                (
                    ("local", key("component-b")),
                    502,
                    str(document),
                    3,
                    (0.25).hex(),
                    key("fallback"),
                ),
            ),
            (1, "physical_relation", (201, 10, 101, "related_to", 102)),
            (1, "relation_evidence", signature),
        ),
    )
    normalized = project_legacy_authorized_snapshot_v1(
        snapshot=graph_snapshot,
        codec=codec,
    )
    assert len(normalized.relation_groups) == 1
    assert len(normalized.mentions) == 1
