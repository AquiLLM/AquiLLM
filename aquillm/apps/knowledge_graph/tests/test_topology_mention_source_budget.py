"""Raw mention hydration has a separate bound from final top-two selection."""

from dataclasses import replace
from time import monotonic

import pytest

from apps.knowledge_graph.projection.memgraph_driver import Neo4jMemgraphDriver
from apps.knowledge_graph.projection.memgraph_repository import (
    MemgraphProjectionRepository,
)
from apps.knowledge_graph.projection.topology_adapter import (
    Neo4jProjectedTopologyQueryAdapter,
)
from apps.knowledge_graph.retrieval.topology.contracts import (
    ProjectedSeedV1,
    TopologyFailureReason,
)
from apps.knowledge_graph.retrieval.topology.failures import TopologyLoadError
from apps.knowledge_graph.retrieval.topology.memgraph import (
    MemgraphProjectedTopologyLoader,
)
from apps.knowledge_graph.tests.memgraph_test_support import (
    isolated_memgraph_container as _isolated_memgraph_container,
)
from apps.knowledge_graph.tests.test_memgraph_projection_repository import (
    _expected_manifest,
)
from apps.knowledge_graph.tests.test_projected_topology_adapter import _caps, _ready
from apps.knowledge_graph.tests.test_projection_records import _bundle


@pytest.fixture(scope="module")
def isolated_memgraph_container():
    yield from _isolated_memgraph_container.__wrapped__()


@pytest.fixture(scope="module")
def mention_graph(isolated_memgraph_container):
    target = isolated_memgraph_container
    driver = Neo4jMemgraphDriver(target["uri"], "", "", database=target["database"])
    original = _bundle()
    counts = (1501, 5000)
    families = tuple(
        tuple(
            replace(
                original.entity_mentions[0],
                entity_key=entity.entity_key,
                mention_key=f"{family * 10000 + index + 1:064x}",
                provenance_key=f"{family * 10000 + index + 1:064x}",
                confidence=1.0
                if index == count - 1
                else 0.9
                if index == count - 2
                else 0.1,
            )
            for index in range(count)
        )
        for family, (entity, count) in enumerate(
            zip(original.entities, counts, strict=True)
        )
    )
    bundle = replace(
        original,
        relation_semantics=(),
        relations=(),
        evidence=(),
        entity_mentions=tuple(row for family in families for row in family),
        counts=replace(
            original.counts,
            relation_semantics_count=0,
            relation_count=0,
            evidence_count=0,
            entity_mention_count=6501,
        ),
    )
    try:
        repository = MemgraphProjectionRepository(driver)
        expected = replace(
            _expected_manifest(bundle), private_mapping_checksum="d" * 64
        )
        repository.write_staging_generation(
            bundle=bundle,
            private_mapping_checksum=expected.private_mapping_checksum,
            batch_size=128,
            timeout_seconds=15.0,
        )
        validation = repository.validate_generation(
            expected=expected, timeout_seconds=15.0
        )
        assert validation.valid
        repository.mark_generation_ready(
            generation_key=repository.opaque_generation_key(
                bundle.generation.generation_key
            ),
            validation_checksum=validation.validation_checksum,
            timeout_seconds=15.0,
        )
        yield driver, bundle, families
    finally:
        if driver._client is not None:
            driver._client.close()


def _load(driver, bundle, identity):
    return MemgraphProjectedTopologyLoader(
        Neo4jProjectedTopologyQueryAdapter(driver)
    ).load(
        ready=_ready(bundle),
        seeds=(ProjectedSeedV1(identity, 1.0),),
        caps=_caps(),
        deadline=monotonic() + 30.0,
    )


@pytest.mark.container
def test_late_page_best_mentions_survive_complete_source_hydration(mention_graph):
    driver, bundle, families = mention_graph
    snapshot = _load(driver, bundle, bundle.entities[0].entity_key)
    assert len(snapshot.identity_keys) == 1
    assert len(snapshot.mentions) == 2
    assert {row.evidence.provenance_key for row in snapshot.mentions} == {
        families[0][-1].provenance_key,
        families[0][-2].provenance_key,
    }
    assert sorted(row.evidence.confidence for row in snapshot.mentions) == [0.9, 1.0]
    assert snapshot.caps.max_nodes == 200
    assert snapshot.caps.max_mentions_per_entity == 2


@pytest.mark.container
def test_5000_raw_mentions_fail_with_branch_local_capacity(mention_graph):
    driver, bundle, _ = mention_graph
    with pytest.raises(TopologyLoadError) as captured:
        _load(driver, bundle, bundle.automatic_memberships[1].automatic_membership_key)
    assert captured.value.reason is TopologyFailureReason.DIRECT_TOPOLOGY_INVALID


@pytest.mark.container
def test_malformed_overflow_sentinel_remains_shared_schema_failure(mention_graph):
    driver, bundle, families = mention_graph
    parameters = {
        "generation_key": bundle.generation.generation_key,
        "mention_key": families[1][-1].mention_key,
    }
    query = (
        "MATCH (n:ProjectedEntityMention {generation_key:$generation_key, "
        "opaque_key:$mention_key}) SET n.confidence=$confidence"
    )
    driver.execute_write(query, {**parameters, "confidence": -1.0}, timeout_seconds=5.0)
    try:
        with pytest.raises(TopologyLoadError) as captured:
            _load(
                driver, bundle, bundle.automatic_memberships[1].automatic_membership_key
            )
        assert captured.value.reason is TopologyFailureReason.BACKEND_SCHEMA_MISMATCH
    finally:
        driver.execute_write(
            query, {**parameters, "confidence": 1.0}, timeout_seconds=5.0
        )
