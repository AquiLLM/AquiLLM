"""Provider-neutral projected topology and PPR parity proofs."""

from __future__ import annotations

from dataclasses import replace

from apps.knowledge_graph.projection.topology_adapter import (
    Neo4jProjectedTopologyQueryAdapter,
)
from apps.knowledge_graph.projection.topology_snapshot import (
    build_projected_topology_snapshot,
)
from apps.knowledge_graph.retrieval import projected_types as t
from apps.knowledge_graph.retrieval.ppr import PPRAlgorithmConfig, personalized_pagerank
from apps.knowledge_graph.retrieval.projected_ppr import ppr_projected_v1
from apps.knowledge_graph.retrieval.topology.contracts import ProjectedSeedV1
from apps.knowledge_graph.retrieval.topology.memgraph import (
    MemgraphProjectedTopologyLoader,
)
from apps.knowledge_graph.retrieval.topology.postgres import (
    PostgresProjectedTopologyLoader,
    _make_test_postgres_parity_capability,
)
from apps.knowledge_graph.tests.test_projected_topology_adapter import (
    ProjectionDriver,
    _caps,
    _ready,
)
from apps.knowledge_graph.tests.test_projection_records import _bundle


class _PostgresSource:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.calls = []

    def load(self, **kwargs):
        self.calls.append(kwargs)
        return self.snapshot


def _tied_groups(scores) -> frozenset[frozenset[object]]:
    grouped: dict[float, set[object]] = {}
    for identity, score in scores.items():
        grouped.setdefault(score, set()).add(identity)
    return frozenset(
        frozenset(group) for group in grouped.values() if len(group) > 1
    )


def _isolated_bundle():
    bundle = _bundle()
    counts = replace(
        bundle.counts,
        relation_semantics_count=0,
        relation_count=0,
        evidence_count=0,
        entity_mention_count=0,
    )
    return replace(
        bundle,
        relation_semantics=(),
        relations=(),
        evidence=(),
        entity_mentions=(),
        counts=counts,
    )


def test_memgraph_and_postgres_match_bytes_scores_traces_ties_and_ranks() -> None:
    bundle = _isolated_bundle()
    ready, caps = _ready(bundle), _caps()
    seeds = tuple(
        sorted(
            (
                ProjectedSeedV1(bundle.entities[0].entity_key, 0.5),
                ProjectedSeedV1(
                    bundle.automatic_memberships[1].automatic_membership_key,
                    0.5,
                ),
            ),
            key=lambda row: row.identity_key,
        )
    )
    expected = build_projected_topology_snapshot(
        ready=ready,
        seeds=seeds,
        caps=caps,
        bundles=(bundle,),
    )
    memgraph = MemgraphProjectedTopologyLoader(
        Neo4jProjectedTopologyQueryAdapter(
            ProjectionDriver(bundle),
            clock=lambda: 40.0,
        )
    ).load(ready=ready, seeds=seeds, caps=caps, deadline=42.5)
    source = _PostgresSource(expected)
    capability = _make_test_postgres_parity_capability(source=source)
    postgres = PostgresProjectedTopologyLoader(source, capability).load(
        capability=capability,
        ready=ready,
        seeds=seeds,
        caps=caps,
        deadline=42.5,
    )

    assert t.canonical_projected_snapshot_bytes(memgraph) == (
        t.canonical_projected_snapshot_bytes(postgres)
    )
    config = PPRAlgorithmConfig(
        canonical_resolver_version="resolver-v1",
        max_seeds=caps.max_seeds,
    )
    memgraph_result = ppr_projected_v1(
        snapshot=memgraph,
        seeds=seeds,
        config=config,
    )
    postgres_result = ppr_projected_v1(
        snapshot=postgres,
        seeds=seeds,
        config=config,
    )
    assert dict(memgraph_result.scores) == dict(postgres_result.scores)
    assert memgraph_result.trace_bytes == postgres_result.trace_bytes
    assert _tied_groups(dict(memgraph_result.scores)) == _tied_groups(
        dict(postgres_result.scores)
    )
    assert memgraph_result.ranked_identity_keys == (
        postgres_result.ranked_identity_keys
    )
    assert _tied_groups(dict(memgraph_result.scores))
    assert source.calls == [
        {"ready": ready, "seeds": seeds, "caps": caps, "deadline": 42.5}
    ]


def test_legacy_parity_compares_logical_scores_and_ties_not_db_order() -> None:
    left, right = ("canonical", 9), ("local", "f" * 64)
    first = personalized_pagerank(
        {left: 0.5, right: 0.5},
        {},
        restart_probability=0.2,
        iterations=2,
    )
    second = personalized_pagerank(
        {right: 0.5, left: 0.5},
        {},
        restart_probability=0.2,
        iterations=2,
    )

    assert first == second
    assert _tied_groups(first) == _tied_groups(second) == {
        frozenset((left, right))
    }
