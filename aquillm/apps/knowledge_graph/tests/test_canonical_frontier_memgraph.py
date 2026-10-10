"""Real owned Memgraph access-plan and bridge proofs."""

import os
from uuid import uuid4

import pytest

from apps.knowledge_graph.projection.memgraph_driver import Neo4jMemgraphDriver


@pytest.fixture
def frontier_memgraph():
    if os.environ.get("RAG_REMEDIATION_TEST") != "1":
        pytest.skip("requires the sanctioned disposable remediation wrapper")
    assert os.environ["KG_MEMGRAPH_URI"] == "bolt://memgraph:7687"
    driver = Neo4jMemgraphDriver(
        "bolt://memgraph:7687",
        "",
        "",
        database="memgraph",
        max_transaction_retry_time=0.0,
    )
    try:
        yield driver
    finally:
        driver._connection().close()


def test_frontier_equality_lookup_stops_before_third_duplicate(frontier_memgraph):
    driver = frontier_memgraph
    value = uuid4().hex
    with driver._connection().session(database="memgraph") as session:
        session.run("CREATE INDEX ON :FrontierIndexV2(index_key)").consume()
    driver.execute_write(
        "UNWIND range(1,201) AS i "
        "CREATE (:FrontierIndexV2 {index_key:$key, fixture:$key})",
        {"key": value},
        timeout_seconds=5.0,
    )
    query = (
        "USING INDEX :FrontierIndexV2(index_key) "
        "MATCH (n:FrontierIndexV2 {index_key:$key}) WITH n LIMIT 2 RETURN n AS record"
    )
    try:
        plan = driver.execute_read(
            "EXPLAIN " + query, {"key": value}, timeout_seconds=5.0, max_records=100
        )
        print("FRONTIER_EXPLAIN", plan)
        # Memgraph PROFILE is an implicit-transaction diagnostic command.
        from neo4j import Query

        with driver._connection().session(database="memgraph") as session:
            profile = tuple(
                session.run(
                    Query("PROFILE " + query, timeout=5.0), {"key": value}
                ).data()
            )
        print("FRONTIER_PROFILE", profile)
        assert not any(
            "ScanAll (" in row["QUERY PLAN"] or "OrderBy" in row["QUERY PLAN"]
            for row in plan
        )
        rows = driver.execute_read(
            query, {"key": value}, timeout_seconds=5.0, max_records=2
        )
        assert len(rows) == 2
        accesses = [r for r in profile if "ScanAllByLabelProperties" in r["OPERATOR"]]
        assert accesses, "PROFILE must demonstrate property equality access"
        assert all(r["ACTUAL HITS"] <= 2 for r in accesses), (
            "equality query examined hidden duplicate rows beyond LIMIT2"
        )
    finally:
        driver.execute_write(
            "MATCH (n:FrontierIndexV2 {fixture:$key}) DELETE n",
            {"key": value},
            timeout_seconds=5.0,
        )


def test_real_index_staging_validates_source_and_raw_equality_plan(frontier_memgraph):
    from apps.knowledge_graph.projection.frontier_cypher_v2 import RAW_LOOKUP
    from apps.knowledge_graph.projection.memgraph_repository import (
        MemgraphProjectionRepository,
    )
    from apps.knowledge_graph.tests.frontier_fixture_support import bridge_bundles
    from apps.knowledge_graph.tests.test_memgraph_projection_repository import (
        _expected_manifest,
    )

    driver = frontier_memgraph
    repository = MemgraphProjectionRepository(driver)
    bundle = bridge_bundles()[0]
    generation = repository.opaque_generation_key(bundle.generation.generation_key)
    repository.delete_generation(generation_key=generation, timeout_seconds=5.0)
    try:
        from dataclasses import replace

        expected = replace(
            _expected_manifest(bundle), private_mapping_checksum="d" * 64
        )
        repository.write_staging_generation(
            bundle=bundle,
            private_mapping_checksum=expected.private_mapping_checksum,
            batch_size=128,
            timeout_seconds=5.0,
            frontier_index=True,
        )
        validation = repository.validate_generation(
            expected=expected, timeout_seconds=5.0
        )
        assert validation.valid
        repository.mark_generation_ready(
            generation_key=generation,
            validation_checksum=validation.validation_checksum,
            timeout_seconds=5.0,
        )
        parameters = {
            "generation_key": generation.value,
            "keys_csv": bundle.entities[0].entity_key,
        }
        plan = driver.execute_read(
            "EXPLAIN " + RAW_LOOKUP, parameters, timeout_seconds=5.0, max_records=100
        )
        print("FRONTIER_RAW_EXPLAIN", plan)
        assert not any(
            "ScanAll (" in r["QUERY PLAN"] or "OrderBy" in r["QUERY PLAN"] for r in plan
        )
        with driver._connection().session(database="memgraph") as session:
            profile = tuple(session.run("PROFILE " + RAW_LOOKUP, parameters).data())
        print("FRONTIER_RAW_PROFILE", profile)
        rows = driver.execute_read(
            RAW_LOOKUP, parameters, timeout_seconds=5.0, max_records=10
        )
        assert len(rows) == 2  # Entity and its automatic membership share opaque key.
        assert {frozenset(r["labels"]) for r in rows} == {
            frozenset({"ProjectedRecord", "ProjectedEntity"}),
            frozenset({"ProjectedRecord", "AutomaticMembership"}),
        }
        driver.execute_write(
            "UNWIND range(1,199) AS i CREATE (:ProjectedRecord:ProjectedEntity "
            "{generation_key:$generation_key,opaque_key:$key})",
            {"generation_key": generation.value, "key": bundle.entities[0].entity_key},
            timeout_seconds=5.0,
        )
        with driver._connection().session(database="memgraph") as session:
            overflow = tuple(session.run("PROFILE " + RAW_LOOKUP, parameters).data())
        print("FRONTIER_RAW_OVERFLOW_PROFILE", overflow)
        accesses = [r for r in overflow if "ScanAllByLabelProperties" in r["OPERATOR"]]
        assert accesses and all(r["ACTUAL HITS"] == 10 for r in accesses)
        assert (
            len(
                driver.execute_read(
                    RAW_LOOKUP, parameters, timeout_seconds=5.0, max_records=10
                )
            )
            == 10
        )
    finally:
        repository.delete_generation(generation_key=generation, timeout_seconds=5.0)


@pytest.mark.parametrize("hub", [False, True])
def test_real_canonical_bridge_uses_original_seed_and_at_most_64_queries(
    frontier_memgraph, hub
):
    from dataclasses import replace
    from time import monotonic

    from apps.knowledge_graph.projection.frontier_adapter_v2 import (
        Neo4jCanonicalFrontierAdapterV2,
    )
    from apps.knowledge_graph.projection.memgraph_repository import (
        MemgraphProjectionRepository,
    )
    from apps.knowledge_graph.retrieval.topology.frontier_loader_v2 import (
        canonical_frontier_loader_v2,
    )
    from apps.knowledge_graph.tests.frontier_fixture_support import FrontierFixture
    from apps.knowledge_graph.tests.frontier_hub_fixture import hub_bundles
    from apps.knowledge_graph.tests.test_memgraph_projection_repository import (
        _expected_manifest,
    )

    bundles, late_answer = hub_bundles() if hub else (None, None)
    fixture = FrontierFixture(bundles)
    repository = MemgraphProjectionRepository(frontier_memgraph)
    generations = [
        repository.opaque_generation_key(b.generation.generation_key)
        for b in fixture.bundles
    ]
    try:
        for bundle, generation in zip(fixture.bundles, generations, strict=True):
            repository.delete_generation(generation_key=generation, timeout_seconds=5.0)
            expected = replace(
                _expected_manifest(bundle), private_mapping_checksum="d" * 64
            )
            repository.write_staging_generation(
                bundle=bundle,
                private_mapping_checksum="d" * 64,
                batch_size=128,
                timeout_seconds=5.0,
                frontier_index=True,
            )
            validation = repository.validate_generation(
                expected=expected, timeout_seconds=5.0
            )
            assert validation.valid
            repository.mark_generation_ready(
                generation_key=generation,
                validation_checksum=validation.validation_checksum,
                timeout_seconds=5.0,
            )
        if not hub:
            from apps.knowledge_graph.projection.topology_adapter import (
                Neo4jProjectedTopologyQueryAdapter,
            )
            from apps.knowledge_graph.retrieval.topology.memgraph import (
                MemgraphProjectedTopologyLoader,
            )

            legacy = MemgraphProjectedTopologyLoader(
                Neo4jProjectedTopologyQueryAdapter(frontier_memgraph)
            ).load(
                ready=fixture.group.scope.ready,
                seeds=fixture.seeds,
                caps=fixture.caps,
                deadline=monotonic() + 10.0,
            )
            assert fixture.shared_identity in legacy.identity_keys
            assert fixture.answer_identity not in legacy.identity_keys

        class Meter:
            calls = 0

            def execute_read(self, *args, **kwargs):
                self.calls += 1
                return frontier_memgraph.execute_read(*args, **kwargs)

        meter = Meter()
        loader = canonical_frontier_loader_v2(
            selection=fixture.selection,
            codec=fixture.codec,
            adapter=Neo4jCanonicalFrontierAdapterV2(meter),
        )
        fixture.deadline = monotonic() + 10.0
        budget = fixture.budget()
        result = loader.load(
            group=fixture.group,
            seeds=fixture.seeds,
            request=fixture.request,
            caps=fixture.caps,
            budget=budget,
            deadline=fixture.deadline,
        )
        print(
            "FRONTIER_REAL_BRIDGE",
            {
                "queries": budget.queries,
                "source_rows": budget.source_rows,
                "identities": budget.identities,
                "edges": budget.edges,
                "coverage": result.coverage.status,
            },
        )
        if hub:
            assert late_answer not in result.snapshot.identity_keys
            assert result.coverage.reasons == ("frontier_budget",)
            assert result.coverage.status == "partial"
        else:
            assert fixture.answer_identity in result.snapshot.identity_keys
        assert budget.queries == meter.calls <= 64
        assert result.coverage.exhaustive is False
    finally:
        for generation in generations:
            repository.delete_generation(generation_key=generation, timeout_seconds=5.0)
