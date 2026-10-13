"""Admission controls preserve each real turn mode and its original ledger."""
import asyncio

from asgiref.sync import async_to_sync
import pytest

from apps.chat.services.rag_action_tools import admit_tool_action
from apps.chat.services.rag_turn import preservation_turn
from apps.documents.services.source_loading import (
    SourceRuntime, current_source_runtime, source_runtime_scope,
)
from lib.llm.turn_context import current_turn
from lib.retrieval.turn_budget import TurnBudget, TurnLimits
from .cache_source_support import cache_corpus_factory, upload, transport
from .cache_turn_support import modes, useful

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("mode", ["default", "source"])
@pytest.mark.parametrize("control", ["cap", "duplicate", "cancel", "mismatch"])
def test_admission_uses_original_turn_budget(
    cache_corpus_factory, monkeypatch, control, mode,
):
    modes(monkeypatch, mode)
    monkeypatch.setenv("RAG_DIRECT_MAX_QUERIES", "3")
    corpus = cache_corpus_factory()
    consumer = corpus.consumer

    async def run():
        async with preservation_turn(consumer, max_func_calls=3) as public:
            turn = current_turn()
            runtime = turn._media_source_runtime
            snapshot = runtime._frozen_retrieval_scope
            budget = turn.budget
            expected_public = None if mode == "default" else runtime
            assert public is expected_public and current_source_runtime() is expected_public
            assert runtime.budget is budget and budget.limits.actions == 3
            assert budget.actions_used == 0
            useful(budget)
            if control == "cap":
                for number in range(3):
                    assert admit_tool_action("vector", f"calibration {number}")
                    assert budget.actions_used == number + 1
                assert not admit_tool_action("vector", "fourth")
                assert budget.actions_used == 3
            elif control == "duplicate":
                assert admit_tool_action("document", "calibration",
                    document_id=corpus.documents[0].id)
                assert not admit_tool_action("document", "calibration",
                    document_id=corpus.documents[0].id)
                assert budget.actions_used == 1
                assert admit_tool_action("document", "calibration",
                    document_id=corpus.documents[0].id, include_images=True)
                assert budget.actions_used == 2
            elif control == "cancel":
                budget.close("cancelled")
                with pytest.raises(asyncio.CancelledError, match="cancelled"):
                    admit_tool_action("vector", "late")
                assert budget.actions_used == 0
            else:
                foreign = TurnBudget(TurnLimits())
                try:
                    with source_runtime_scope(SourceRuntime(foreign, runtime.authorization)):
                        assert not admit_tool_action("vector", "foreign budget")
                        assert budget.actions_used == foreign.actions_used == 0
                finally:
                    foreign.close("fixture_finished")
            assert current_turn() is turn and current_source_runtime() is expected_public
            assert runtime.budget is budget and runtime._frozen_retrieval_scope is snapshot
    async_to_sync(run)()


def test_standalone_public_runtime_retains_its_finite_admission():
    # Admission alone grants no retrieval or publication authority.
    assert current_turn() is None and current_source_runtime() is None
    budget = TurnBudget(TurnLimits())
    try:
        with source_runtime_scope(SourceRuntime(budget, None)):
            assert admit_tool_action("vector", "one")
            assert not admit_tool_action("vector", "one")
            assert admit_tool_action("vector", "two")
            assert admit_tool_action("vector", "three")
            assert not admit_tool_action("vector", "four")
            assert budget.actions_used == 3
    finally:
        budget.close("fixture_finished")


def test_unbound_outside_turn_keeps_existing_admission_compatibility():
    assert current_turn() is None and current_source_runtime() is None
    assert admit_tool_action("vector", "standalone")
