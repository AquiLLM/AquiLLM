"""One cumulative ledger includes discarded work, copies and all groups."""

from importlib.util import find_spec

import pytest

from lib.retrieval.turn_budget import TurnBudget, TurnLimits


def test_frontier_budget_deduplicates_admission_but_never_refunds_physical_work():
    assert (
        find_spec("apps.knowledge_graph.retrieval.topology.frontier_contracts_v2")
        is not None
    ), "request-local canonical frontier accounting is missing"
    from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
        FrontierBudgetV2,
    )

    def clock():
        return 1.0

    budget = FrontierBudgetV2(
        turn_budget=TurnBudget(TurnLimits(), clock=clock),
        deadline=10.0,
        max_identities=2,
        max_edges=1,
        max_source_rows=4,
        max_queries=2,
        max_wire_bytes=100,
        clock=clock,
    )
    assert budget.admit_identity("a" * 64)
    assert budget.admit_identity("a" * 64)
    assert budget.identities == 1
    assert budget.reserve_query() and budget.reserve_query()
    assert not budget.reserve_query()
    assert budget.charge_page(scanned_rows=3, wire_bytes=90)
    assert not budget.charge_page(scanned_rows=2, wire_bytes=11)
    assert (budget.source_rows, budget.wire_bytes, budget.queries) == (5, 101, 2)
    with pytest.raises(AttributeError):
        budget.max_queries = 64
    with pytest.raises(AttributeError):
        budget.deadline = 100.0


def test_completed_oversized_response_charges_actual_rows_without_refund():
    from apps.knowledge_graph.projection.frontier_reads_v2 import FrontierReadsV2
    from apps.knowledge_graph.retrieval.topology.frontier_gateway_contracts_v2 import (
        FrontierCompletedBudgetFailureV2,
    )
    from apps.knowledge_graph.tests.frontier_fixture_support import FrontierFixture

    f = FrontierFixture()
    budget = f.budget()

    class Driver:
        def execute_frontier_read(self, **kwargs):
            raise FrontierCompletedBudgetFailureV2(1)

    reads = FrontierReadsV2(
        Driver(), ready=f.group.scope.ready, budget=budget, deadline=f.deadline
    )
    with pytest.raises(FrontierCompletedBudgetFailureV2):
        reads.read("manifest", f.bundles[0].generation.generation_key)
    assert budget.queries == budget.source_rows == 1
    assert budget._reserved_rows == 0
    assert not budget._uncertain
