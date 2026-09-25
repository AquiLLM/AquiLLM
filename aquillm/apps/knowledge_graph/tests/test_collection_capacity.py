"""Operational collection envelopes must preserve every admitted input."""

import tracemalloc
from dataclasses import replace

import pytest

from apps.knowledge_graph.graph.assembly import (
    AssemblyConfig,
    validate_assembly_projection,
)
from apps.knowledge_graph.graph.assembly_plan import CollectionAssemblyPlan
from apps.knowledge_graph.resolution.collection import (
    CollectionResolutionConfig,
    resolve_collection_entities,
)
from apps.knowledge_graph.tests.test_collection_resolution import (
    _document_entity,
    _ontology,
    _session,
    _snapshot,
)


def test_large_collection_preserves_every_distinct_input_and_provenance():
    count = 89_052
    config = CollectionResolutionConfig(max_entities=100_000)
    # Unknown ontology types are valid isolated nodes and require no embeddings.
    entities = (
        _document_entity(index, f"entity {index}", entity_type="unmapped_type")
        for index in range(1, count + 1)
    )
    session, _backend = _session({})
    result = resolve_collection_entities(
        _snapshot(config=config),
        entities,
        _ontology(),
        config=config,
        embedding_session=session,
    )
    assert len(result.clusters) == count
    assert {
        member for cluster in result.clusters for member in cluster.document_entity_ids
    } == set(range(1, count + 1))
    assert all(cluster.document_artifact_ids == (201,) for cluster in result.clusters)
    assert result.audit.embedded_entity_count == 0
    active = frozenset(range(1, count + 1))
    stats = validate_assembly_projection(
        CollectionAssemblyPlan((), (), "a" * 64),
        active_entity_ids=active,
        provenanced_entity_ids=active,
        config=AssemblyConfig(max_entities=100_000, max_orphan_entities=100_000),
    )
    assert stats.entity_count == stats.orphan_count == count


def test_large_collection_rejects_overflow_before_consuming_unbounded_input():
    config = CollectionResolutionConfig(max_entities=100_000)
    consumed = 0

    def inputs():
        nonlocal consumed
        while True:
            consumed += 1
            yield None

    session, _backend = _session({})
    with pytest.raises(ValueError, match="entity input exceeds configured limit"):
        resolve_collection_entities(
            _snapshot(config=config),
            inputs(),
            _ontology(),
            config=config,
            embedding_session=session,
        )
    assert consumed == 100_001
    with pytest.raises(ValueError, match="max_entities"):
        replace(config, max_entities=100_001)
    with pytest.raises(ValueError, match="max_entities"):
        AssemblyConfig(max_entities=100_001)


def test_candidate_iteration_releases_each_roots_pool_before_next_root():
    from apps.knowledge_graph.resolution.candidate_pool import iter_candidate_pools

    count = 10_000
    roots = tuple(range(count))
    pools = iter_candidate_pools(
        roots,
        component_keys={root: f"{root:064x}" for root in roots},
        lexical_keys=lambda _root: ("common",),
        exact_scan_limit=512,
        pool_limit=128,
    )
    observed = 0
    tracemalloc.start()
    try:
        for root, candidates in pools:
            assert len(candidates) == 128
            assert root not in candidates
            if root == 0:
                assert candidates == set(range(1, 129))
            observed += len(candidates)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert observed == 1_280_000
    # Retaining 10k independent 128-member sets exceeds this by several times.
    assert peak < 16 * 1024 * 1024


def test_candidate_iteration_preserves_block_priority_and_wraparound():
    from apps.knowledge_graph.resolution.candidate_pool import iter_candidate_pools

    keys = {1: ("a", "b"), 2: ("b",), 3: ("a",), 4: ("a", "b")}
    pools = dict(
        iter_candidate_pools(
            (1, 2, 3, 4),
            component_keys={1: "d", 2: "c", 3: "b", 4: "a"},
            lexical_keys=keys.__getitem__,
            exact_scan_limit=2,
            pool_limit=2,
        )
    )
    assert pools == {1: {3, 4}, 2: {1, 4}, 3: {1, 4}, 4: {1, 3}}
