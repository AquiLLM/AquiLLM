"""Batch writes retain selective identities when read indexes coexist."""

from dataclasses import asdict
from hashlib import sha256
from types import SimpleNamespace

import pytest
from neo4j import GraphDatabase

from apps.knowledge_graph.projection.memgraph_driver import Neo4jMemgraphDriver
from apps.knowledge_graph.projection.memgraph_edges import (
    write_parameterized_batches,
    write_topology_edges,
)
from apps.knowledge_graph.projection.memgraph_repository import (
    MemgraphProjectionRepository,
)
from apps.knowledge_graph.tests.memgraph_test_support import (
    isolated_memgraph_container as _isolated_memgraph_container,
)
from apps.knowledge_graph.tests.test_memgraph_projection_repository import _FakeDriver
from apps.knowledge_graph.tests.test_projection_records import _bundle


@pytest.fixture
def isolated_memgraph_container():
    yield from _isolated_memgraph_container.__wrapped__()


@pytest.mark.container
def test_full_membership_batch_uses_composite_endpoints_with_read_indexes(
    isolated_memgraph_container,
):
    target = isolated_memgraph_container
    # Disable automatic retries so a regressed query cannot hide a timeout.
    with GraphDatabase.driver(
        target["uri"], auth=None, max_transaction_retry_time=0
    ) as raw:
        driver = Neo4jMemgraphDriver(
            target["uri"], "", "", database=target["database"], driver=raw
        )
        repository = MemgraphProjectionRepository(driver)
        repository.ensure_schema(timeout_seconds=5.0)
        bundle = _bundle()
        generation = bundle.generation.generation_key
        driver.execute_write(
            "CREATE (:CollectionGeneration "
            "{generation_key:$generation_key, state:'staging'})",
            {"generation_key": generation},
            timeout_seconds=5.0,
        )
        # A large same-generation family makes its generation-only index an
        # expensive alternative to the shared (generation, opaque identity) index.
        driver.execute_write(
            "UNWIND range(0,39999) AS i "
            "CREATE (:ProjectedEntity:ProjectedRecord "
            "{generation_key:$generation_key, opaque_key:toString(i)}), "
            "(:AutomaticMembership:ProjectedRecord "
            "{generation_key:$generation_key, opaque_key:toString(i)})",
            {"generation_key": generation},
            timeout_seconds=5.0,
        )
        template = _FakeDriver()
        MemgraphProjectionRepository(template).write_staging_generation(
            bundle=bundle,
            private_mapping_checksum="d" * 64,
            batch_size=128,
            timeout_seconds=0.3,
        )
        keys = tuple(sha256(f"batch:{i}".encode()).hexdigest() for i in range(128))
        calls = _FakeDriver()
        for label, record in (
            ("ProjectedEntity", bundle.entities[0]),
            ("AutomaticMembership", bundle.automatic_memberships[0]),
        ):
            query = next(q for q, _, _ in template.writes if f"(n:{label}:" in q)
            body = query.split("] AS row ", 1)[1]
            write_parameterized_batches(
                calls,
                body,
                [dict(asdict(record), entity_key=key, opaque_key=key) for key in keys],
                generation_key=generation,
                batch_size=128,
                timeout_seconds=0.3,
            )
        write_topology_edges(
            calls,
            SimpleNamespace(
                generation=bundle.generation,
                automatic_memberships=tuple(
                    SimpleNamespace(entity_key=key) for key in keys
                ),
                chunks=(),
                relations=(),
                relation_semantics=(),
                evidence=(),
                entity_mentions=(),
            ),
            timeout_seconds=0.3,
        )
        assert len(calls.writes) == 3
        for query, parameters, timeout in calls.writes:
            plan = driver.execute_read(
                "EXPLAIN " + query,
                parameters,
                timeout_seconds=5.0,
                max_records=100,
            )
            operators = "\n".join(row["QUERY PLAN"] for row in plan)
            required = 2 if "ENTITY_MEMBERSHIP" in query else 1
            assert (
                operators.count(":ProjectedRecord {generation_key, opaque_key}")
                == required
            )
            assert "ScanAll (" not in operators
            for _ in range(2):
                driver.execute_write(query, parameters, timeout_seconds=timeout)
        count_query = (
            "MATCH ()-[e:ENTITY_MEMBERSHIP {generation_key:$generation_key}]->() "
            "RETURN count(e) AS count"
        )
        assert driver.execute_read(
            count_query,
            {"generation_key": generation},
            timeout_seconds=5.0,
            max_records=1,
        ) == ({"count": 128},)
        driver.execute_write(
            "MATCH (g:CollectionGeneration {generation_key:$generation_key}) "
            "SET g.state='ready' WITH g "
            "MATCH ()-[e:ENTITY_MEMBERSHIP {generation_key:$generation_key}]->() "
            "DELETE e",
            {"generation_key": generation},
            timeout_seconds=5.0,
        )
        query, parameters, timeout = calls.writes[-1]
        driver.execute_write(query, parameters, timeout_seconds=timeout)
        assert driver.execute_read(
            count_query,
            {"generation_key": generation},
            timeout_seconds=5.0,
            max_records=1,
        ) == ({"count": 0},)
