"""Complete scope keeps all 31/530 proofs within five bounded raw reads."""

import pytest

from apps.knowledge_graph.tests.frontier_scope_window_support import MixedScopeFixture
from apps.knowledge_graph.tests.frontier_window_test_support import WindowOpener


def test_canonical_scope_uses_five_complete_raw_reads_from_directory(monkeypatch):
    reference = MixedScopeFixture()
    reference.session.initialize()
    assert (
        len([r for r in reference.driver.transcript if r["kind"] == "raw_batch_multi"])
        == 18
    )
    fixture = MixedScopeFixture()
    # Projected remote metadata debits: protocol isolation, not native evidence.
    for rows in [1, 1, 1, 7, 1, 264, 1, 7, 1, 1, 1, 1, 1, 1, 5]:
        fixture.budget.reserve_read(rows)
        fixture.budget.charge_read(reserved_rows=rows, scanned_rows=rows)
    opener = WindowOpener(fixture, monkeypatch, complete_scope=True)
    fixture.session.initialize()
    raw = [
        r for r in fixture.driver.transcript if r["kind"] == "scope_raw_complete_batch"
    ]
    assert [len(r["pairs"]) for r in raw] == [128, 128, 128, 128, 49]
    assert [r["rows"] for r in raw] == [249, 249, 249, 248, 96]
    assert all(r["maximum"] == 320 for r in raw)
    assert fixture.session.scope_rows == reference.session.scope_rows
    assert (
        fixture.session.reads._index_records == reference.session.reads._index_records
    )
    assert (fixture.budget.queries, fixture.budget.source_rows) == (35, 2721)
    assert opener.requests


def test_inherited_sixteen_seeds_reach_candidates_and_fresh_final(monkeypatch):
    from apps.knowledge_graph.retrieval.topology.contracts import ProjectedSeedV1
    from apps.knowledge_graph.retrieval.topology.frontier_acquisition_v2 import (
        acquire_frontier,
    )
    from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
        FrontierBudgetExceeded,
    )
    from apps.knowledge_graph.tests.test_frontier_copy_feasibility_v2 import (
        GatewayOpener,
    )

    fixture = MixedScopeFixture()
    for rows in [1, 1, 1, 7, 1, 264, 1, 7, 1, 1, 1, 1, 1, 1, 5]:
        fixture.budget.reserve_read(rows)
        fixture.budget.charge_read(reserved_rows=rows, scanned_rows=rows)
    transport = GatewayOpener(fixture, monkeypatch, windows=True, complete_scope=True)
    seeds = tuple(
        ProjectedSeedV1(b.relations[0].source_entity_key, 1 / 16)
        for b in sorted(fixture.bundles, key=lambda b: b.generation.generation_key)[:16]
    )
    fixture.session.initialize()
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
    assert len(fixture.assembly.candidates) == 2
    assert (fixture.budget.queries, fixture.budget.source_rows) == (63, 4937)
    assert fixture.budget.queries == len(fixture.driver.calls) + 15
    assert fixture.budget.source_rows == fixture.driver.scanned + 294
    assert (
        fixture.budget.wire_bytes == sum(f["wire"] for f in transport.frames) <= 1048576
    )
    assert not fixture.budget._uncertain and not fixture.budget._authority_reservations
    assert fixture.budget._window is None and fixture.budget._reserved_rows == 0
    print(
        {
            "queries": fixture.budget.queries,
            "rows": fixture.budget.source_rows,
            "wire": fixture.budget.wire_bytes,
            "identities": fixture.budget.identities,
            "edges": fixture.budget.edges,
            "frames": transport.frames,
        }
    )


@pytest.mark.parametrize(
    "queries,rows,seeds_present,final",
    [
        (20, 519, 0, (63, 4968)),
        (15, 309, 16, (58, 4930)),
    ],
)
def test_retained_insufficient_metadata_compositions_still_fail_useful_gate(
    monkeypatch, queries, rows, seeds_present, final
):
    from apps.knowledge_graph.retrieval.topology.contracts import ProjectedSeedV1
    from apps.knowledge_graph.retrieval.topology.frontier_acquisition_v2 import (
        acquire_frontier,
    )
    from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
        FrontierBudgetExceeded,
    )
    from apps.knowledge_graph.tests.test_frontier_copy_feasibility_v2 import (
        GatewayOpener,
    )

    fixture = MixedScopeFixture()
    # Historical aggregate comparisons only, never an actual native SQL claim.
    for debit in [rows] + [0] * (queries - 1):
        fixture.budget.reserve_read(debit)
        fixture.budget.charge_read(reserved_rows=debit, scanned_rows=debit)
    transport = GatewayOpener(fixture, monkeypatch, windows=True, complete_scope=True)
    seeds = tuple(
        ProjectedSeedV1(b.relations[0].source_entity_key, 1 / 16)
        for b in sorted(fixture.bundles, key=lambda b: b.generation.generation_key)[:16]
    )
    fixture.session.initialize()
    with pytest.raises(FrontierBudgetExceeded):
        acquire_frontier(
            fixture.session, fixture.assembly, seeds, fixture.caps, fixture.budget
        )
    fixture.session.final_attestation()
    assert (
        len({s.identity_key for s in seeds} & fixture.budget._identities)
        == seeds_present
    )
    assert not fixture.assembly.candidates
    assert (fixture.budget.queries, fixture.budget.source_rows) == final
    assert fixture.budget.wire_bytes == sum(frame["wire"] for frame in transport.frames)
    assert not fixture.budget._authority_reservations and not fixture.budget._uncertain
