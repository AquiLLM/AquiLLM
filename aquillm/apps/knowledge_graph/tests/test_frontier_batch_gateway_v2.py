"""Closed fixed-query batch transport, including real per-key LIMIT2."""

import json
from hashlib import sha256
from time import monotonic
from uuid import uuid4

import pytest

from apps.knowledge_graph.projection.frontier_reads_v2 import QUERIES
from apps.knowledge_graph.retrieval.topology.frontier_gateway_contracts_v2 import (
    CHECKSUM, FrontierCompletedBudgetFailureV2, FrontierGatewayRequestV2,
    decode_request, encode_request,
)
from apps.knowledge_graph.retrieval.topology.frontier_gateway_service_v2 import read_payload
from apps.knowledge_graph.tests.frontier_scope_batch_support import ScopeBatchFixture
from apps.knowledge_graph.tests.memgraph_test_support import (
    isolated_memgraph_container as _isolated_memgraph_container,
)


@pytest.mark.parametrize("size", [1, 32])
def test_batch_envelope_and_service_use_exact_requested_keys_and_physical_bound(size):
    fixture = ScopeBatchFixture()
    ready = fixture.group.scope.ready
    keys = tuple(sorted(fixture.driver.index)[:size])
    request = FrontierGatewayRequestV2(
        "index_batch", ready.selected_generations[0].generation_key, keys, ready, 1048576,
    )
    assert decode_request(encode_request(request)) == request
    assert request.maximum == 2 * size
    payload = json.loads(read_payload(fixture.driver, request, monotonic() + 5, 1048576))
    assert payload["queries"] == 1 and payload["scanned_rows"] == size
    assert {r["lookup_key"] for r in payload["rows"]} == set(keys)
    assert len(fixture.driver.calls) == 1
    assert fixture.driver.calls[0] == (
        QUERIES["index_batch"], {"generation_key": request.generation, "keys_csv": ",".join(keys)},
    )
    assert fixture.driver.bounds[0][1] == 2 * size


@pytest.mark.parametrize("invalid", ["empty", "33", "duplicate", "unsorted", "wrong_key", "foreign_generation"])
def test_invalid_or_unselected_batch_is_rejected_before_driver_dispatch(invalid):
    fixture = ScopeBatchFixture()
    keys = tuple(sorted(fixture.driver.index)[:2])
    generation = fixture.bundle.generation.generation_key
    if invalid == "empty":
        keys = ()
    elif invalid == "33":
        keys = tuple(sorted(fixture.driver.index)[:33])
    elif invalid == "duplicate":
        keys = keys[:1] * 2
    elif invalid == "unsorted":
        keys = tuple(reversed(keys))
    elif invalid == "wrong_key":
        keys = ("invalid",)
    else:
        generation = "0" * 64
    with pytest.raises(ValueError):
        FrontierGatewayRequestV2("index_batch", generation, keys, fixture.group.scope.ready, 1048576)
    assert fixture.driver.calls == []


def test_batch_wire_overflow_reports_actual_completed_rows_and_requires_new_peer_checksum():
    fixture = ScopeBatchFixture()
    keys = tuple(sorted(fixture.driver.index)[:32])
    request = FrontierGatewayRequestV2(
        "index_batch", fixture.bundle.generation.generation_key, keys, fixture.group.scope.ready, 1,
    )
    with pytest.raises(FrontierCompletedBudgetFailureV2) as captured:
        read_payload(fixture.driver, request, monotonic() + 5, 1048576)
    assert captured.value.scanned_rows == 32 and len(fixture.driver.calls) == 1
    old = sha256(b"frontier-v2:fixed-manifest-index-raw:relative-timeout:raw-body:limit2:raw-limit10").hexdigest()
    assert CHECKSUM != old


@pytest.fixture
def isolated_memgraph_container():
    yield from _isolated_memgraph_container.__wrapped__()


@pytest.mark.container
def test_owned_batch_query_reads_two_rows_per_key_before_any_identity_filter(isolated_memgraph_container):
    from neo4j import GraphDatabase
    from apps.knowledge_graph.projection.memgraph_driver import Neo4jMemgraphDriver

    query = QUERIES.get("index_batch")
    assert query is not None, "fixed batch query is missing"
    target = isolated_memgraph_container
    keys = tuple(sha256((uuid4().hex + str(i)).encode()).hexdigest() for i in range(3))
    with GraphDatabase.driver(target["uri"], auth=None, max_transaction_retry_time=0) as raw:
        driver = Neo4jMemgraphDriver(target["uri"], "", "", database=target["database"], driver=raw)
        with raw.session(database=target["database"]) as session:
            session.run("CREATE INDEX ON :FrontierIndexV2(index_key)").consume()
        try:
            driver.execute_write(
                "UNWIND split($keys_csv, ',') AS k UNWIND range(1,3) AS i "
                "CREATE (:FrontierIndexV2 {index_key:k,generation_key:'wrong',fixture:$fixture})",
                {"keys_csv": ",".join(keys), "fixture": keys[0]}, timeout_seconds=5.0,
            )
            rows = driver.execute_read(
                query, {"generation_key": "f" * 64, "keys_csv": ",".join(keys)},
                timeout_seconds=5.0, max_records=6,
            )
            assert len(rows) == 6
            assert all(sum(r["lookup_key"] == k for r in rows) == 2 for k in keys)
            assert all(r["record"]["generation_key"] == "wrong" for r in rows)
        finally:
            driver.execute_write(
                "MATCH (n:FrontierIndexV2 {fixture:$fixture}) DELETE n",
                {"fixture": keys[0]}, timeout_seconds=5.0,
            )
