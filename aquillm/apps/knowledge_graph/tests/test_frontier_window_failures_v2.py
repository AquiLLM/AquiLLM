"""Discarded, malformed and uncertain windows cannot replenish the ledger."""

from copy import deepcopy

import pytest

from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
    FrontierBudgetExceeded,
)
from apps.knowledge_graph.retrieval.topology.frontier_window_contracts_v2 import (
    CompletedWindowBudgetFailure,
)
from apps.knowledge_graph.tests.frontier_copy_batch_support import CopyFixture
from apps.knowledge_graph.tests.frontier_window_test_support import WindowOpener


@pytest.mark.parametrize("allowance", [100, 200])
def test_small_wire_failure_receipt_charges_known_work_or_retains_uncertainty(
    monkeypatch, allowance
):
    fixture = CopyFixture()
    transport = WindowOpener(fixture, monkeypatch)
    budget = fixture.ledger
    budget.charge_page(
        scanned_rows=0, wire_bytes=budget.available_wire_bytes() - allowance
    )
    before = budget.queries, budget.source_rows
    with pytest.raises(Exception) as caught:
        fixture.session.prepare_copies(fixture.pairs, relation_heads=True)
    assert len(fixture.driver.calls) == before[0] + 1
    assert len(transport.requests) == 1 and transport.responses[0].closed
    if allowance == 200:
        assert isinstance(caught.value, CompletedWindowBudgetFailure)
        assert budget.queries == len(fixture.driver.calls)
        assert budget.source_rows == fixture.driver.scanned
        assert budget._window is None and not budget._uncertain
        fixture.session.final_attestation()
    else:
        assert budget._window is not None and budget._uncertain
        assert (budget.queries, budget.source_rows) == before
        with pytest.raises(FrontierBudgetExceeded):
            fixture.session.final_attestation()


@pytest.mark.parametrize("mutation", ["digest", "truncated", "later", "kind"])
def test_malformed_window_never_retries_or_releases_unknown_work(monkeypatch, mutation):
    fixture = CopyFixture()
    transport = WindowOpener(fixture, monkeypatch)
    before = fixture.ledger.queries, fixture.ledger.source_rows
    cache = deepcopy(fixture.session.reads._index_records)

    def mutate(body):
        assert body.startswith(b"KGFW0001") and body[40] > 1
        if mutation == "digest":
            return body[:8] + bytes([body[8] ^ 1]) + body[9:]
        if mutation == "truncated":
            return body[:-1]
        if mutation == "kind":
            return body[:41] + b"\x02" + body[42:]
        return body[:-1] + bytes([body[-1] ^ 1])

    transport.mutate = mutate
    with pytest.raises(Exception):
        fixture.session.prepare_copies(fixture.pairs, relation_heads=True)
    assert fixture.ledger._uncertain and fixture.ledger._window is not None
    assert (fixture.ledger.queries, fixture.ledger.source_rows) == before
    assert fixture.session.reads._index_records == cache
    assert all(r.closed for r in transport.responses)
    with pytest.raises(FrontierBudgetExceeded):
        fixture.session.prepare_copies(fixture.pairs, relation_heads=True)
    assert len(transport.requests) == 1
