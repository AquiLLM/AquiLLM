"""Incompatible groups share the original ledger, frontier and final cap."""

import pytest

from apps.knowledge_graph.tests.ready_subset_support import ScopeFixture
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

pytestmark = pytest.mark.django_db


@pytest.mark.parametrize("consume_all", [False, True])
def test_groups_share_one_frontier_and_original_turn_budget(monkeypatch, consume_all):
    from apps.knowledge_graph.retrieval.ready_group_budget_v2 import (
        schedule_ready_groups_v2,
    )

    fixture = ScopeFixture(monkeypatch)
    selected = fixture.select_v2(ready=("A", "B"), pending=(), alternate=True)
    ledger = TurnBudget(TurnLimits())
    seen = []

    def execute(group, *, frontier, turn_budget, deadline):
        assert turn_budget is ledger
        seen.append(group.group_checksum)
        assert frontier.admit(identities=200 if consume_all else 1, edges=0)
        return ()

    candidates, coverage = schedule_ready_groups_v2(
        selected=selected,
        execute_group=execute,
        turn_budget=ledger,
        deadline=100.0,
        clock=lambda: 0.0,
    )
    assert len(seen) == (1 if consume_all else 2)
    assert len(selected.groups) == 2
    assert candidates == ()
    assert coverage.status == ("partial" if consume_all else "complete")
    if consume_all:
        assert coverage.reasons == ("frontier_budget", "incompatible_generation")


def test_candidate_maximum_is_global_across_groups(monkeypatch):
    from apps.knowledge_graph.retrieval.graph_handoff import GraphCandidateRefV2
    from apps.knowledge_graph.retrieval.ready_group_budget_v2 import (
        schedule_ready_groups_v2,
    )

    fixture = ScopeFixture(monkeypatch)
    selected = fixture.select_v2(ready=("A", "B"), pending=(), alternate=True)
    calls = []

    def execute(group, *, frontier, turn_budget, deadline):
        index = len(calls)
        calls.append(group.group_checksum)
        assert frontier.admit(identities=20, edges=0)
        return tuple(
            GraphCandidateRefV2(
                f"{index * 20 + number + 1:064x}",
                group.scope.ready.selected_generations[0].generation_key,
                None,
                group.group_checksum,
                group.scope.ready.bundle_checksum,
                ("a" * 64,),
                (),
                0.1 if index == 0 else 0.9,
            )
            for number in range(20)
        )

    refs, coverage = schedule_ready_groups_v2(
        selected=selected,
        execute_group=execute,
        turn_budget=TurnBudget(TurnLimits()),
        deadline=100.0,
        clock=lambda: 0.0,
    )
    assert len(calls) == 2 and len(refs) == 20
    assert all(ref.authority_group_checksum == calls[1] for ref in refs)
    assert coverage.status == "complete"
