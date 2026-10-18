"""Captured ancestor fields cannot be renewed without any source-bound context."""

import pytest

from apps.knowledge_graph.retrieval.ready_authority_reads_v2 import capture_authority_check
from lib.retrieval.operation import Operation, _OPERATION
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


@pytest.mark.parametrize("mutation", ["deadline", "clock", "budget", "parent"])
def test_original_ancestor_cannot_be_replaced_or_renewed(mutation):
    now = [0.0]
    clock = lambda: now[0]
    budget = TurnBudget(TurnLimits(), clock=clock)
    parent = Operation(budget, deadline=1.0, clock=clock)
    leaf = Operation(budget, deadline=5.0, clock=clock, parent=parent)
    token = _OPERATION.set(leaf)
    try:
        check = capture_authority_check()
        if mutation == "deadline":
            parent.deadline = 100.0
        elif mutation == "clock":
            parent.clock = lambda: 0.0
        elif mutation == "budget":
            parent.budget = TurnBudget(TurnLimits(), clock=clock)
        else:
            leaf.parent = Operation(budget, deadline=100.0, clock=clock)
        now[0] = 0.1 if mutation == "budget" else 2.0
        with pytest.raises(TimeoutError):
            check()
    finally:
        _OPERATION.reset(token)


@pytest.mark.parametrize("kind", ["cycle", "depth"])
def test_invalid_operation_ancestry_is_bounded_before_clock_calls(kind):
    budget = TurnBudget(TurnLimits())
    leaf = Operation(budget)
    if kind == "cycle":
        leaf.parent = leaf
    else:
        for _ in range(65):
            leaf = Operation(budget, parent=leaf)
    token = _OPERATION.set(leaf)
    try:
        with pytest.raises(TimeoutError):
            capture_authority_check()
    finally:
        _OPERATION.reset(token)


def test_original_parent_expiry_is_not_replaced_by_later_leaf_deadline():
    now = [0.0]
    clock = lambda: now[0]
    budget = TurnBudget(TurnLimits(), clock=clock)
    parent = Operation(budget, deadline=1.0, clock=clock)
    token = _OPERATION.set(Operation(budget, deadline=5.0, clock=clock, parent=parent))
    try:
        check = capture_authority_check()
        now[0] = 2.0
        with pytest.raises(TimeoutError):
            check()
    finally:
        _OPERATION.reset(token)


def test_last_clock_callback_cannot_replace_original_parent_budget():
    calls, armed = [0], [None]
    replacement = TurnBudget(TurnLimits())

    def clock():
        calls[0] += 1
        if calls[0] == armed[0]:
            parent.budget = replacement
        return 0.0

    budget = TurnBudget(TurnLimits(), clock=clock)
    parent = Operation(budget, deadline=10.0, clock=clock)
    token = _OPERATION.set(Operation(budget, deadline=5.0, clock=clock, parent=parent))
    try:
        check = capture_authority_check()
        before = calls[0]
        check()
        callbacks = calls[0] - before
        assert callbacks > 0
        armed[0] = calls[0] + callbacks
        with pytest.raises(TimeoutError):
            check()
        assert parent.budget is replacement
    finally:
        _OPERATION.reset(token)
