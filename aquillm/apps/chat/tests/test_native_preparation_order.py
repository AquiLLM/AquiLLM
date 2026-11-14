"""Accepted native work obligations are separate from ordinary source provenance."""

from asyncio import CancelledError
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from apps.chat.services.rag_recovery_origins import (
    accept_recovery_action, recovery_action, start_recovery_origins, take_recovery_order,
)
from apps.chat.services.rag_retrieval import fuse_ranked_tool_results
from apps.chat.tests.test_recovery_origins import payload, source
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from lib.llm.turn_context import TurnContext, bind_turn
from lib.retrieval import TurnBudget, TurnLimits
from lib.retrieval.operation import operation_scope


@contextmanager
def native_frame(monkeypatch):
    monkeypatch.setenv("RAG_ACQUISITION_SCHEDULE", "native_closure")
    monkeypatch.setenv("RAG_SCOPED_BASELINE_RECOVERY_ENABLED", "0")
    budget = TurnBudget(TurnLimits(), clock=lambda: 0.0)
    runtime, turn = SourceRuntime(budget, object()), TurnContext(budget)
    runtime._frozen_retrieval_scope = SimpleNamespace(representation=object(), dense=object())
    with source_runtime_scope(runtime), bind_turn(turn):
        record = start_recovery_origins("question", budget)
        assert record is not None, "native work obligations need an accepted-action lifetime with scoped recovery off"
        yield SimpleNamespace(budget=budget, runtime=runtime, turn=turn, record=record)


def accept_native(case, sources, initial, *, failure=None):
    result = payload(sources)
    with operation_scope(case.budget, timeout_ms=1000), recovery_action(case.record) as action:
        # Like the existing origin unit fixture, model only the private pending
        # native identity record. The source/answer integration uses the real owner.
        identity = lambda s: (s.chunk_id, s.document_id, s.chunk_number, s.source_fingerprint)
        action.preparation.append((tuple(map(identity, sources)), tuple(map(identity, initial))))
        if failure is not None:
            raise failure("caller never accepted the native result")
        accept_recovery_action(action, result, sources)
    assert not action.operation.live
    return result


def take(case, results, query="question"):
    from apps.chat.services.rag_preparation_order import take_preparation_order

    pool = fuse_ranked_tool_results(results)
    return pool, take_preparation_order(pool, query, case.runtime.authorization)


def test_native_initial_obligations_preserve_pool_relevance_and_are_consumed_once(monkeypatch):
    with native_frame(monkeypatch) as case:
        a, b, c = source(1), source(2), source(3)
        first = accept_native(case, (b, a), (a,))
        second = accept_native(case, (c, b), (b,))
        pool, order = take(case, [first, second, first])
        ids = tuple(row["chunk_id"] for row in pool.rows)
        assert order.initial_ids == tuple(pk for pk in ids if pk in {1, 2})
        assert order.deferred_ids == (3,)
        assert order.fused_scores == pool.fused_scores  # Work priority cannot rewrite ordinary votes.
        assert len(order.origins.occurrences) == 2
        assert take(case, [first, second])[1] is None
        assert take_recovery_order(pool, "question", case.runtime.authorization) is None


@pytest.mark.parametrize("failure", [TimeoutError, CancelledError])
def test_unaccepted_native_completion_cannot_create_preparation_priority(monkeypatch, failure):
    with native_frame(monkeypatch) as case:
        a, b = source(1), source(2)
        accepted = accept_native(case, (a,), (a,))
        with pytest.raises(failure):
            accept_native(case, (b, a), (b,), failure=failure)
        assert len(case.record.occurrences) == 1
        assert take(case, [accepted])[1] is None


@pytest.mark.parametrize("mutation", ["payload", "query", "authorization", "representation", "turn"])
def test_native_preparation_order_rejects_changed_accepted_inputs(monkeypatch, mutation):
    with native_frame(monkeypatch) as case:
        a, b = source(1), source(2)
        result = accept_native(case, (a, b), (a,))
        query = "question"
        if mutation == "payload":
            result["result"][0]["text"] = "changed accepted body"
        elif mutation == "query":
            query = "another question"
        elif mutation == "authorization":
            case.runtime.authorization = object()
        elif mutation == "representation":
            case.runtime._frozen_retrieval_scope.representation = object()
        else:
            case.turn.budget = TurnBudget(TurnLimits())
        with pytest.raises(SourceScopeUnavailable):
            take(case, [result], query)
