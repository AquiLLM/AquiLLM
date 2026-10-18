"""Real scalar-parameter Cypher checks in the sanctioned local Memgraph."""

from hashlib import sha256
from uuid import uuid4

import pytest

from apps.knowledge_graph.projection.frontier_reads_v2 import QUERIES
from apps.knowledge_graph.tests.test_canonical_frontier_memgraph import (
    frontier_memgraph as _frontier_memgraph,
)


@pytest.fixture
def frontier_memgraph():
    yield from _frontier_memgraph.__wrapped__()


@pytest.mark.parametrize(
    "kind", ["manifest_batch", "index_batch_multi", "raw_batch_multi"]
)
def test_fixed_generation_batch_is_one_statement_with_per_pair_sentinels(
    frontier_memgraph, kind
):
    driver = frontier_memgraph
    keys = [sha256((uuid4().hex + str(i)).encode()).hexdigest() for i in range(3)]
    label, field, cap = {
        "manifest_batch": ("CollectionGeneration", "generation_key", 2),
        "index_batch_multi": ("FrontierIndexV2", "index_key", 2),
        "raw_batch_multi": ("ProjectedRecord", "opaque_key", 10),
    }[kind]
    with driver._connection().session(database="memgraph") as session:
        index = "generation_key, opaque_key" if kind == "raw_batch_multi" else field
        session.run(f"CREATE INDEX ON :{label}({index})").consume()
    try:
        if kind == "manifest_batch":
            pairs = [(k, k) for k in keys]
            properties = "generation_key:k"
        elif kind == "index_batch_multi":
            pairs = [(keys[0], k) for k in keys]
            properties = "index_key:k,generation_key:'wrong'"
        else:
            pairs = [(k, keys[0]) for k in keys]
            properties = "generation_key:k,opaque_key:$lookup"
        driver.execute_write(
            f"UNWIND split($keys_csv, ',') AS k UNWIND range(1,{cap + 1}) AS i "
            f"CREATE (:{label} {{{properties},fixture:$fixture}})",
            {"keys_csv": ",".join(keys), "lookup": keys[0], "fixture": keys[0]},
            timeout_seconds=5.0,
        )
        rows = driver.execute_read(
            QUERIES[kind],
            {"lookups_csv": ",".join(g + ":" + k for g, k in sorted(pairs))},
            timeout_seconds=5.0,
            max_records=cap * 3,
        )
        assert len(rows) == cap * 3
        assert all(
            sum((r["generation_key"], r["lookup_key"]) == pair for r in rows) == cap
            for pair in pairs
        )
        if kind == "index_batch_multi":
            assert all(r["record"]["generation_key"] == "wrong" for r in rows)
        if kind == "raw_batch_multi":
            assert all(
                r["record"]["generation_key"] == r["generation_key"] for r in rows
            )
    finally:
        driver.execute_write(
            f"MATCH (n:{label} {{fixture:$fixture}}) DELETE n",
            {"fixture": keys[0]},
            timeout_seconds=5.0,
        )
