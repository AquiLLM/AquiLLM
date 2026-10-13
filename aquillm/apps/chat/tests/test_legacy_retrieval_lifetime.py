"""Legacy requests share retrieval accounting without activating preservation."""

from types import SimpleNamespace
import pytest
from apps.chat.services import rag_turn
from lib.llm.turn_context import current_turn
from apps.chat.tests.persisted_direct_test_support import (
    close_direct_worker as close_direct_worker, persisted_direct_case,
)
from apps.documents.services.source_loading import current_source_runtime
from apps.documents.services.representation_manual_scope import private_acquisition_runtime


@pytest.mark.asyncio
async def test_off_mode_has_one_budget_and_no_public_source_runtime(monkeypatch, close_direct_worker):
    monkeypatch.setattr(
        rag_turn, "rag_preservation_config", lambda: SimpleNamespace(active=False)
    )
    case = await persisted_direct_case([])
    consumer = case.consumer(None, (1,))
    async with rag_turn.preservation_turn(consumer, max_func_calls=3) as runtime:
        assert runtime is None
        turn = current_turn()
        assert turn is not None
        assert turn.handoff is not None  # Mandatory publication guard, not preservation.
        assert current_source_runtime() is None
        assert private_acquisition_runtime().budget is turn.budget
        assert turn.budget.reserve_action("first")
        async with rag_turn.preservation_turn(consumer, max_func_calls=3):
            assert current_turn().budget is turn.budget
            assert not current_turn().budget.reserve_action("first")
        assert turn.budget.can_publish()
    assert not turn.budget.can_publish()
    assert current_turn() is None


@pytest.mark.asyncio
async def test_legacy_cancellation_closes_the_parent_ledger(monkeypatch, close_direct_worker):
    import asyncio

    monkeypatch.setattr(
        rag_turn, "rag_preservation_config", lambda: SimpleNamespace(active=False)
    )
    case = await persisted_direct_case([])
    consumer = case.consumer(None, (1,))
    ready = asyncio.Event()
    budgets = []

    async def run():
        async with rag_turn.preservation_turn(consumer, max_func_calls=3):
            budgets.append(current_turn().budget)
            ready.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(run())
    try:
        await asyncio.wait_for(ready.wait(), 1)
    except TimeoutError:
        await task  # Surface missing legacy context rather than hang the red run.
    rag_turn.cancel_active_turn(consumer)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not budgets[0].publish(lambda: pytest.fail("late publish"))


@pytest.mark.asyncio
async def test_legacy_turn_lifetime_allows_sealed_synthesis_past_retrieval_deadline(
    monkeypatch, close_direct_worker,
):
    from lib.retrieval.turn_budget import TurnBudget, TurnLimits
    from lib.retrieval.synthesis_budget import seal_synthesis

    now = [0.0]
    budget = TurnBudget(
        TurnLimits(retrieval_ms=50, final_scoring_ms=0), clock=lambda: now[0]
    )
    monkeypatch.setattr(
        rag_turn, "rag_preservation_config", lambda: SimpleNamespace(active=False)
    )
    monkeypatch.setattr(rag_turn, "TurnBudget", lambda limits: budget)
    case = await persisted_direct_case([])
    consumer = case.consumer(None, (1,))
    async with rag_turn.preservation_turn(consumer, max_func_calls=3):
        assert current_turn().budget is budget
        lease = seal_synthesis(
            budget,
            "synthetic authorized evidence",
            calls=1,
            output_tokens=512,
            timeout_seconds=1,
        )
        now[0] = 0.1
        assert not budget.can_publish()
        assert current_turn().handoff is not None
        assert current_source_runtime() is None
        assert private_acquisition_runtime().budget is budget
        assert lease.can_publish()
        assert lease.start_dispatch(512) == 0
    import asyncio

    with pytest.raises(asyncio.CancelledError):
        lease.check_active()


@pytest.mark.asyncio
async def test_legacy_manual_retrieval_returns_on_deadline_and_cleans_actual_worker(
    monkeypatch,
):
    import asyncio
    from threading import Event
    from time import monotonic
    from apps.chat.services import manual_search_turn as manual
    from lib.llm.types.conversation import Conversation
    from lib.llm.types.messages import UserMessage
    from lib.retrieval.turn_budget import TurnBudget, TurnLimits

    monkeypatch.setattr(
        rag_turn, "rag_preservation_config", lambda: SimpleNamespace(active=False)
    )
    from apps.chat.services import rag_config

    monkeypatch.setattr(
        rag_config, "rag_preservation_config", lambda: SimpleNamespace(active=False)
    )
    started, release, finished = Event(), Event(), Event()
    writes, observed = [], []
    budget = TurnBudget(TurnLimits(retrieval_ms=50, final_scoring_ms=0))
    from lib.llm.turn_context import TurnContext, bind_turn

    def retrieve(*args):
        observed.append(current_turn().budget)
        started.set()
        try:
            release.wait(2)
            budget.publish(lambda: writes.append("late"))
            return "vector_search", "selected documents", {}, {"result": []}
        finally:
            finished.set()

    monkeypatch.setattr(manual, "_retrieve", retrieve)
    consumer = SimpleNamespace(col_ref=SimpleNamespace(collections=[1]))
    convo = Conversation(
        system="sys", messages=[UserMessage(content="/collection synthetic question")]
    )
    with bind_turn(TurnContext(budget)):
        start = monotonic()
        task = asyncio.create_task(manual.run_manual_search_turn(consumer, None, convo))
        await asyncio.to_thread(started.wait, 1)
        try:
            assert await asyncio.wait_for(task, 0.4) == "handled"
            assert monotonic() - start < 0.4
            assert not finished.is_set()
        finally:
            release.set()
            await asyncio.to_thread(finished.wait, 1)
    assert observed == [budget]
    assert writes == []
