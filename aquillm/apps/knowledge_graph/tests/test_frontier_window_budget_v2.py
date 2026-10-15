"""Window reservations spend the original ledger and retain uncertainty."""

import pytest

from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
    FrontierBudgetExceeded,
)
from apps.knowledge_graph.tests.frontier_fixture_support import FrontierFixture


def test_window_holds_preserve_two_final_groups_and_settle_only_actual_work():
    ledger = FrontierFixture().budget()
    reserve = getattr(ledger, "reserve_window", None)
    assert callable(reserve), "original ledger needs a finite window reservation"
    one = ledger.reserve_authority(62, 1)
    two = ledger.reserve_authority(4, 1)
    ledger.reserve_authority_wire(one, 100)
    ledger.reserve_authority_wire(two, 200)
    grant = reserve()
    assert grant.planning_rows == 4934
    assert (grant.queries, grant.rows, grant.wire) == (8, 2560, 1048276)
    assert ledger.available_read_rows() == 2374
    assert ledger.available_wire_bytes() == 0
    assert ledger.queries == ledger.source_rows == 0
    with pytest.raises(FrontierBudgetExceeded):
        reserve()
    with pytest.raises(FrontierBudgetExceeded):
        ledger.reserve_read(2, authority=two)
    ledger.charge_page(scanned_rows=0, wire_bytes=91)
    ledger.settle_window(grant, queries=3, scanned_rows=123)
    assert ledger.available_read_rows() == 4811
    assert ledger.available_wire_bytes() == 1048185
    assert (ledger.queries, ledger.source_rows, ledger.wire_bytes) == (3, 123, 91)
    assert len(ledger._authority_reservations) == 2
    with pytest.raises(ValueError):
        ledger.settle_window(grant, queries=3, scanned_rows=123)


@pytest.mark.parametrize("queries,rows", [(0, 0), (9, 1), (1, 321), (True, 1), (1, -1)])
def test_invalid_settlement_keeps_hold_and_poisoned_original_ledger(queries, rows):
    ledger = FrontierFixture().budget()
    reserve = getattr(ledger, "reserve_window", None)
    assert callable(reserve), "original ledger needs a finite window reservation"
    grant = reserve()
    with pytest.raises(ValueError):
        ledger.settle_window(grant, queries=queries, scanned_rows=rows)
    assert ledger._window is grant
    assert ledger.queries == ledger.source_rows == 0
    with pytest.raises(FrontierBudgetExceeded):
        ledger.check_active()


def test_uncertain_window_never_becomes_measured_work_or_releases_hold():
    ledger = FrontierFixture().budget()
    reserve = getattr(ledger, "reserve_window", None)
    assert callable(reserve), "original ledger needs a finite window reservation"
    grant = reserve()
    ledger.stop_uncertain()
    assert ledger._window is grant
    assert ledger.queries == ledger.source_rows == 0
    with pytest.raises(FrontierBudgetExceeded):
        reserve()
