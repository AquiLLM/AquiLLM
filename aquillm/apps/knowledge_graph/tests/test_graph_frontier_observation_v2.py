"""Frontier telemetry preserves actual overshoots and incomplete remote accounting."""

from dataclasses import asdict

from apps.knowledge_graph.retrieval.graph_frontier_observation_v2 import (
    capture_frontier_budget_v2,
    publish_frontier_budget_v2,
)
from apps.knowledge_graph.tests.frontier_fixture_support import FrontierFixture
from lib.evidence_observation import observe


def test_observation_is_the_same_budget_and_preserves_overshoot():
    budget = FrontierFixture().budget(max_wire_bytes=20)
    budget.reserve_read(2)
    budget.charge_read(reserved_rows=2, scanned_rows=3)
    budget.charge_page(scanned_rows=0, wire_bytes=21)
    events = []
    with observe(lambda event, data: events.append((event, data))):
        result = capture_frontier_budget_v2(budget, max_depth=2)
        publish_frontier_budget_v2(result, query="test", requested_document_ids=())
    assert (result.queries_used, result.rows_used, result.bytes_used) == (1, 3, 21)
    assert result.max_bytes == 20
    assert result.accounting_complete and not result.uncertain
    assert events[0][1]["frontier_budget"] == asdict(result)
    assert budget.wire_bytes == 21


def test_uncertain_or_unsettled_work_cannot_claim_complete_accounting():
    budget = FrontierFixture().budget()
    original = capture_frontier_budget_v2(budget, max_depth=2)
    budget.reserve_read(10)
    pending = capture_frontier_budget_v2(budget, max_depth=2)
    assert not pending.accounting_complete
    budget.stop_uncertain()
    failed = capture_frontier_budget_v2(budget, max_depth=2)
    assert failed.uncertain and not failed.accounting_complete
    assert failed.queries_used == 1
    assert (
        original.policy_signature == pending.policy_signature == failed.policy_signature
    )
