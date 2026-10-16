"""Actual indexed query, full route, physical cardinality, and owner proof gate."""

import pytest

from apps.knowledge_graph.projection import frontier_index_store_v2
from apps.knowledge_graph.projection.frontier_index_contracts_v2 import (
    PACKED_VERSION,
    VERSION,
)
from apps.knowledge_graph.projection.frontier_reads_v2 import QUERIES
from apps.knowledge_graph.retrieval.topology.contracts import ProjectedSeedV1
from apps.knowledge_graph.retrieval.topology.frontier_acquisition_v2 import (
    acquire_frontier,
)
from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
    FrontierBudgetExceeded,
)
from apps.knowledge_graph.tests.frontier_scope_memgraph_support import (
    live_scope as _live_scope,
)
from apps.knowledge_graph.tests.test_frontier_copy_feasibility_v2 import GatewayOpener


@pytest.fixture
def live_scope(request, monkeypatch):
    monkeypatch.setattr(frontier_index_store_v2, "_NEW_INDEX_VERSION", request.param)
    yield from _live_scope.__wrapped__()


@pytest.mark.parametrize(
    "live_scope, initialized",
    [(VERSION, (35, 2721)), (PACKED_VERSION, (27, 2101))],
    indirect=["live_scope"],
)
def test_live_original_ledger_scope_copy_and_final_survive_actual_query(
    live_scope, monkeypatch, initialized
):
    fixture, driver = live_scope
    for rows in [1, 1, 1, 7, 1, 264, 1, 7, 1, 1, 1, 1, 1, 1, 5]:
        fixture.budget.reserve_read(rows)
        fixture.budget.charge_read(reserved_rows=rows, scanned_rows=rows)
    transport = GatewayOpener(fixture, monkeypatch, windows=True, complete_scope=True)
    seeds = tuple(
        ProjectedSeedV1(b.relations[0].source_entity_key, 1 / 16)
        for b in sorted(fixture.bundles, key=lambda b: b.generation.generation_key)[:16]
    )
    fixture.session.initialize()
    assert (fixture.budget.queries, fixture.budget.source_rows) == initialized
    fixture.driver.phase = "acquisition"
    try:
        acquire_frontier(
            fixture.session, fixture.assembly, seeds, fixture.caps, fixture.budget
        )
    except FrontierBudgetExceeded:
        pass
    fixture.driver.phase = "final"
    fixture.session.final_attestation()
    assert {s.identity_key for s in seeds} <= fixture.budget._identities
    if initialized == (35, 2721):
        assert len(fixture.assembly.candidates) == 2
        assert (fixture.budget.queries, fixture.budget.source_rows) == (63, 4937)
    else:
        assert len(fixture.assembly.candidates) >= 2
        assert fixture.budget.queries <= 64 and fixture.budget.source_rows <= 5000
    assert fixture.budget.queries == len(fixture.driver.calls) + 15
    assert fixture.budget.source_rows == fixture.driver.scanned + 294
    assert (
        fixture.budget.wire_bytes == sum(f["wire"] for f in transport.frames) <= 1048576
    )
    assert not fixture.budget._uncertain and not fixture.budget._authority_reservations
    assert fixture.budget._window is None and fixture.budget._reserved_rows == 0
    raw = [
        row
        for row in fixture.driver.transcript
        if row["kind"] == "scope_raw_complete_batch"
    ]
    assert [row["rows"] for row in raw] == [249, 249, 249, 248, 96]
    assert all(row["maximum"] == 320 for row in raw)
    print(
        {
            "queries": fixture.budget.queries,
            "rows": fixture.budget.source_rows,
            "wire": fixture.budget.wire_bytes,
            "identities": fixture.budget.identities,
            "candidates": len(fixture.assembly.candidates),
            "frames": transport.frames,
        }
    )
    # Post-operation diagnostics are explicitly outside the invocation ledger.
    parameters = {"lookups_csv": ",".join(g + ":" + key for g, key in raw[0]["pairs"])}
    query = QUERIES["scope_raw_complete_batch"]
    plan = driver.execute_read(
        "EXPLAIN " + query, parameters, timeout_seconds=5.0, max_records=100
    )
    assert any("ScanAllByLabelProperties" in row["QUERY PLAN"] for row in plan)
    assert not any(
        "ScanAll (" in row["QUERY PLAN"] or "OrderBy" in row["QUERY PLAN"]
        for row in plan
    )
    from neo4j import Query

    with driver._connection().session(database="memgraph") as session:
        profile = tuple(
            session.run(Query("PROFILE " + query, timeout=5.0), parameters).data()
        )
    print({"plan": plan, "profile": profile})
    seeks = [row for row in profile if "ScanAllByLabelProperties" in row["OPERATOR"]]
    assert seeks and all(row["ACTUAL HITS"] <= 1280 for row in seeks)
    # ACTUAL HITS includes end-of-stream pulls: 249 rows plus one final pull.
    assert profile[0]["OPERATOR"] == "* Limit" and profile[0]["ACTUAL HITS"] == 250
