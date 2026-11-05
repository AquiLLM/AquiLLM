"""Accepted occurrences, rather than completed workers, own recovery priority."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from apps.chat.services.rag_recovery_origins import (
    accept_recovery_action, recovery_action, start_recovery_origins, take_recovery_order,
)
from apps.chat.services.rag_retrieval import fuse_ranked_tool_results
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from lib.llm.turn_context import TurnContext, bind_turn
from lib.retrieval import SourceEvidence, TurnBudget, TurnLimits
from lib.retrieval.evidence import fingerprint_source
from lib.retrieval.operation import operation_scope


def source(pk):
    text = f"independent source {pk}"
    return SourceEvidence(pk, "document", pk, fingerprint_source(text), text)


def payload(sources):
    return {"result": [dict(chunk_id=s.chunk_id, doc_id=s.document_id, chunk=s.chunk_number,
        rank=i, citation=f"[doc:{s.document_id} chunk:{s.chunk_id}]", text=s.text)
        for i,s in enumerate(sources, 1)]}


@contextmanager
def frame(monkeypatch):
    monkeypatch.setenv("RAG_SCOPED_BASELINE_RECOVERY_ENABLED", "1")
    now = [0.0]
    budget = TurnBudget(TurnLimits(), clock=lambda: now[0])
    runtime = SourceRuntime(budget, object())
    frozen = SimpleNamespace(representation=object(), dense=object())
    runtime._frozen_retrieval_scope = frozen
    turn = TurnContext(budget)
    with source_runtime_scope(runtime), bind_turn(turn):
        record = start_recovery_origins("question", budget)
        yield SimpleNamespace(now=now, budget=budget, runtime=runtime, turn=turn,
                              frozen=frozen, record=record)


def accept(case, sources, *, ordinary=None, result=None):
    """Unit fixture for the pending native identity record; full native path is
    covered by test_scoped_recovery_final_preparation.
    """
    result = payload(sources) if result is None else result
    with operation_scope(case.budget, timeout_ms=1000), recovery_action(case.record) as action:
        if ordinary is not None:
            identity = lambda s: (s.chunk_id, s.document_id, s.chunk_number, s.source_fingerprint)
            action.native.append((tuple(map(identity, sources)), tuple(map(identity, ordinary))))
        accept_recovery_action(action, result, sources)
    assert not action.operation.live  # Normal history closure must be accepted.
    return result


def test_accepted_ordinary_occurrences_override_recovery_and_ignore_replayed_payload(monkeypatch):
    with frame(monkeypatch) as case:
        a,b,c = source(1),source(2),source(3)
        first = accept(case, (a,b))
        second = accept(case, (c,b,a), ordinary=(a,))
        # b is recovery-only in the second search, but has a genuine ordinary
        # origin in the first. Replaying first is not a third accepted search.
        pool = fuse_ranked_tool_results([first, second, first])
        order = take_recovery_order(pool, "question", case.runtime.authorization)
        assert order.ordinary_ids == (1,2) and order.optional_ids == (3,)
        assert len(order.origins.occurrences) == 2
        assert take_recovery_order(pool, "question", case.runtime.authorization) is None


@pytest.mark.parametrize("failure", [TimeoutError, InterruptedError])
def test_finalized_but_unaccepted_action_adds_no_origin_or_weight(monkeypatch, failure):
    with frame(monkeypatch) as case:
        a,b = source(1),source(2)
        first = accept(case, (a,))
        with pytest.raises(failure):
            with operation_scope(case.budget, timeout_ms=1000), recovery_action(case.record) as action:
                action.native.append((((b.chunk_id,b.document_id,b.chunk_number,b.source_fingerprint),), ()))
                raise failure("caller did not accept the completed worker")
        assert len(case.record.occurrences) == 1
        assert take_recovery_order(fuse_ranked_tool_results([first]), "question", case.runtime.authorization) is None


@pytest.mark.parametrize("mutation", ["budget", "turn_budget", "authorization", "representation", "frozen", "query", "payload"])
def test_origin_receipt_cannot_be_consumed_under_changed_owner_or_input(monkeypatch, mutation):
    with frame(monkeypatch) as case:
        a,b = source(1),source(2)
        result = accept(case, (a,b), ordinary=(a,))
        pool = fuse_ranked_tool_results([result])
        query = "question"
        if mutation == "budget":
            case.runtime.budget = TurnBudget(TurnLimits())
        elif mutation == "turn_budget":
            case.turn.budget = TurnBudget(TurnLimits())
        elif mutation == "authorization":
            case.runtime.authorization = object()
        elif mutation == "representation":
            case.frozen.representation = object()
        elif mutation == "frozen":
            case.runtime._frozen_retrieval_scope = SimpleNamespace(representation=case.frozen.representation, dense=case.frozen.dense)
        elif mutation == "query":
            query = "different question"
        else:
            result["result"][0]["rank"] = 100
        with pytest.raises(SourceScopeUnavailable):
            take_recovery_order(pool, query, case.runtime.authorization)


def test_initial_accepted_obligations_are_deduplicated_without_claiming_native_history(monkeypatch):
    with frame(monkeypatch) as case:
        case.runtime._recovery_origins = None
        a,b,c = source(1),source(2),source(3)
        initial = payload((a,b))
        case.record = start_recovery_origins("question", case.budget, (initial,initial), lambda _: (a,b))
        result = accept(case, (a,c), ordinary=(a,))
        order = take_recovery_order(fuse_ranked_tool_results([initial,result]), "question", case.runtime.authorization)
        assert order.ordinary_ids == (1,2) and order.optional_ids == (3,)
        assert len(order.origins.occurrences) == 2


def test_reentrant_acquisition_discards_prior_origin_record(monkeypatch):
    with frame(monkeypatch) as case:
        assert start_recovery_origins("question", case.budget) is None
        with pytest.raises(SourceScopeUnavailable):
            accept(case, (source(1),source(2)), ordinary=(source(1),))


def test_clock_callback_cannot_replace_authority_during_origin_consumption(monkeypatch):
    with frame(monkeypatch) as case:
        a,b = source(1),source(2)
        result = accept(case, (a,b), ordinary=(a,))
        authority = case.runtime.authorization

        def clock():
            case.runtime.authorization = object()
            return case.now[0]

        case.budget._clock = clock
        with pytest.raises(SourceScopeUnavailable):
            take_recovery_order(fuse_ranked_tool_results([result]), "question", authority)


@pytest.mark.asyncio
async def test_default_off_does_not_repeat_source_view_materialization(monkeypatch):
    from apps.chat.services import rag_acquisition

    monkeypatch.delenv("RAG_SCOPED_BASELINE_RECOVERY_ENABLED", raising=False)
    monkeypatch.setattr(rag_acquisition, "active", lambda: False)
    budget = TurnBudget(TurnLimits())
    calls = []
    value = source(1)

    def views(results):
        calls.append(tuple(results))
        return tuple(s for result in results for s in result["sources"])

    async def execute(action):
        assert budget.reserve_action(action.signature)
        return {"sources": (value,)}

    result = await rag_acquisition.acquire_evidence("question", budget=budget, llm=None,
        execute=execute, evidence_views=views, iterative=False)
    assert result.results and budget.actions_used == 1
    assert len(calls) == 2, "off mode only needs pre-action and accepted-result bookkeeping"


@pytest.mark.parametrize("mutation", ["context", "deadline"])
def test_acceptance_rechecks_original_action_after_ownership_clock(monkeypatch, mutation):
    from lib.retrieval.operation import _OPERATION

    with frame(monkeypatch) as case:
        values = (source(1),source(2))
        with operation_scope(case.budget, timeout_ms=1000), recovery_action(case.record) as action:
            ticks = []

            def clock():
                ticks.append(True)
                if len(ticks) == 2:
                    if mutation == "context":
                        _OPERATION.set(None)
                    else:
                        case.now[0] = 1.001
                return case.now[0]

            case.budget._clock = clock
            with pytest.raises(SourceScopeUnavailable):
                accept_recovery_action(action, payload(values), values)
            assert case.record.occurrences == []


def test_origin_budget_check_rejects_parent_cycle_created_by_operation_clock(monkeypatch):
    import sys
    from lib.retrieval.operation import Operation

    with frame(monkeypatch) as case, operation_scope(case.budget, timeout_ms=1000) as owner:
        prior, calls = sys.gettrace(), []

        def clock():
            owner.parent = owner
            return case.now[0]

        def trace(frame, event, arg):
            if event == "call" and frame.f_code is Operation.active.__code__:
                calls.append(True)
                if len(calls) > 64:
                    raise AssertionError("receipt reached recursive mutable operation ancestry")
            return trace

        owner.clock = clock
        sys.settrace(trace)
        try:
            with pytest.raises(SourceScopeUnavailable, match="recovery_original_scope_unavailable"):
                case.record.check()
        finally:
            sys.settrace(prior)
            owner.parent = None
