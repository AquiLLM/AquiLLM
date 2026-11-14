"""Only the unchanged captured child may witness a ready-authority deadline."""

from dataclasses import replace

import pytest

from apps.documents.services.source_deadline import _DEADLINE, source_preparation_scope
from apps.knowledge_graph.retrieval.graph_child_deadline import _ReadyAuthorityDeadline
from apps.knowledge_graph.retrieval.ready_authority_reads_v2 import capture_authority_check
from lib.retrieval.operation import operation_scope
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


@pytest.mark.parametrize("mutation", ["none", "child_context", "operation_parent"])
def test_deadline_conversion_rechecks_identity_after_clock_callback(mutation):
    budget = TurnBudget(TurnLimits(), clock=lambda: 0.0)
    now, armed, reads = [0.0], [False], []
    with operation_scope(budget, timeout_ms=10_000) as parent:
        with operation_scope(budget, timeout_ms=9000) as operation:

            def child_clock():
                if armed[0]:
                    reads.append(True)
                    # First read expires check_source_deadline. This second
                    # read is the typed conversion's callback-capable witness.
                    if len(reads) == 2:
                        if mutation == "child_context":
                            bound = _DEADLINE.get()
                            _DEADLINE.set((bound[0], bound[1], bound[2]))
                        elif mutation == "operation_parent":
                            operation.parent = replace(parent)
                return now[0]

            try:
                with source_preparation_scope(1.0, child_clock, budget):
                    captured_bound = _DEADLINE.get()
                    check = capture_authority_check()
                    now[0], armed[0] = 1.001, True
                    with pytest.raises(TimeoutError) as failed:
                        check()
                    assert len(reads) == 2
                    if mutation == "none":
                        assert type(failed.value) is _ReadyAuthorityDeadline
                        assert failed.value.bound is captured_bound
                        assert failed.value.operation is operation
                    else:
                        assert type(failed.value) is TimeoutError
                    # Restore lifetime for normal context-manager exit only;
                    # the rejection above occurred while the mutation existed.
                    now[0], armed[0] = 0.0, False
                    operation.parent = parent
                    _DEADLINE.set(captured_bound)
            finally:
                operation.parent = parent
            assert _DEADLINE.get() is None and operation.active()


def test_deadline_witness_uses_existing_millisecond_exhaustion_precision():
    budget = TurnBudget(TurnLimits(), clock=lambda: 0.0)
    now = [0.0]
    with operation_scope(budget, timeout_ms=10_000) as operation:
        with source_preparation_scope(1.0, lambda: now[0], budget):
            check = capture_authority_check()
            now[0] = .9995
            with pytest.raises(_ReadyAuthorityDeadline) as failed:
                check()
            assert failed.value.operation is operation and operation.active()
            now[0] = 0.0
