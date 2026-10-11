"""Final scorer workers retain the invoking turn and private observation."""
from contextvars import ContextVar
from threading import Barrier
from time import monotonic
from types import SimpleNamespace
from uuid import UUID

import pytest

from apps.documents.services.chunk_rerank_scoring import score_missing_pairs
from lib.evidence_observation import active, observe, publish
from lib.llm.turn_context import TurnContext, bind_turn, current_turn
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


@pytest.mark.parametrize("kind", ["pointwise", "listwise"])
def test_final_workers_keep_original_context_without_leaking_it(kind):
    marker = ContextVar("final_score_test_marker", default=None)
    original = object()
    budget = TurnBudget(TurnLimits())
    turn = TurnContext(budget, max_func_calls=3)
    rows = tuple(SimpleNamespace(pk=i, doc_id=UUID(int=1), chunk_number=i,
                                 content=f"complete source {i}.") for i in (1, 2))
    calls, events = [], []
    barrier = Barrier(2, timeout=1)

    class Scorer:
        scoring_kind = kind
        scorer_fingerprint = "context-contract"

        def prepare_pair(self, query, chunk):
            return query, chunk.content

        def score_pair(self, pair, timeout_seconds):
            # Both workers enter concurrently: sharing one Context would fail.
            barrier.wait()
            assert current_turn() is turn and marker.get() is original
            assert active() and current_turn().budget is budget
            calls.append(pair)
            publish("score_context_probe", {"ordinal": len(calls)})
            return 0.5, pair

        def score_pool(self, pairs, timeout_seconds):
            assert current_turn() is turn and marker.get() is original
            assert active() and current_turn().budget is budget
            calls.extend(pairs)
            publish("score_context_probe", {"ordinal": len(calls)})
            return tuple((0.5, pair) for pair in pairs)

    token = marker.set(original)
    try:
        with bind_turn(turn), observe(lambda event, data: events.append(event)):
            result = score_missing_pairs(query="question", chunks=rows, scorer=Scorer(),
                deadline=monotonic() + 3, turn_budget=budget)
            assert result.status == "complete" and len(result.scores) == 2
            assert current_turn() is turn and marker.get() is original
            assert budget.can_publish()
    finally:
        marker.reset(token)
        budget.close("test_finished")
    assert calls and len(calls) == 2
    assert events.count("score_context_probe") == (2 if kind == "pointwise" else 1)
    assert current_turn() is None and not active() and marker.get() is None
