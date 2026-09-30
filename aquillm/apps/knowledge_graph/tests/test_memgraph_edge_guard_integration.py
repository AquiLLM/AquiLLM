from __future__ import annotations

import pytest

from apps.knowledge_graph.projection import memgraph_edge_validation as edges
from apps.knowledge_graph.projection.memgraph_driver import Neo4jMemgraphDriver
from apps.knowledge_graph.tests.memgraph_test_support import (
    isolated_memgraph_container as _isolated_memgraph_container,
)
from apps.knowledge_graph.tests.test_memgraph_edge_pagination_indexes import _SPECS


@pytest.fixture
def isolated_memgraph_container():
    yield from _isolated_memgraph_container.__wrapped__()


@pytest.mark.container
def test_incident_guard_detects_each_endpoint_arm_without_labels(
    isolated_memgraph_container,
):
    target = isolated_memgraph_container
    driver = Neo4jMemgraphDriver(target["uri"], "", "", database=target["database"])
    try:
        driver.ensure_projection_schema(timeout_seconds=5.0)
        with driver._connection().session() as session:
            for query, spec in zip(edges._EDGE_QUERIES, _SPECS, strict=True):
                for source_gen, target_gen, edge_gen in (
                    ("g", "other", "other"),
                    ("other", "g", "other"),
                    ("g", None, None),
                    (None, "g", None),
                ):
                    edge_id = session.run(
                        "CREATE (s {generation_key:$sg})"
                        f"-[e:{spec[0]} {{generation_key:$eg}}]->"
                        "(t {generation_key:$tg}) RETURN id(e) AS edge_id",
                        sg=source_gen,
                        tg=target_gen,
                        eg=edge_gen,
                    ).single()["edge_id"]
                    with pytest.raises(ValueError, match="incident"):
                        tuple(
                            edges._stream_edge_family(driver, "g", query, 0, 100, 0.3)
                        )
                    session.run(
                        "MATCH (s)-[e]->(t) WHERE id(e)=$edge_id DELETE e,s,t",
                        edge_id=edge_id,
                    ).consume()
                assert (
                    tuple(edges._stream_edge_family(driver, "g", query, 0, 100, 0.3))
                    == ()
                )
    finally:
        if driver._client is not None:
            driver._client.close()
