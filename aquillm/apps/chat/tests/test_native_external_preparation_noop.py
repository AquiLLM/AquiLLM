"""Normal tool completion without a native capture creates no work priority."""

from types import SimpleNamespace

from asgiref.sync import async_to_sync
import pytest

from apps.chat.services.rag_turn import preservation_turn
from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from lib.llm.turn_context import call_tool_async
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


@pytest.mark.parametrize("name,payload", [
    ("document_ids", {"result": []}),
    ("whole_document", {"result": []}),
    ("more_context", {"result": []}),
    ("vector_search", {"exception": "tool unavailable"}),
    ("vector_search", {"result": [], "retrieval_status": "uncertain"}),
])
def test_unrelated_auxiliary_and_failed_tools_do_not_require_native_scope(monkeypatch, name, payload):
    monkeypatch.setenv("RAG_ACQUISITION_SCHEDULE", "native_closure")
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source")
    monkeypatch.setenv("RAG_EVIDENCE_SELECTION_MODE", "adaptive")
    monkeypatch.setenv("RAG_SCOPED_BASELINE_RECOVERY_ENABLED", "0")
    budget = TurnBudget(TurnLimits(), clock=lambda: 0.0)
    runtime = SourceRuntime(budget, object())
    conversation = Conversation(system="Answer using sources.", messages=[UserMessage(content="question")])
    consumer = SimpleNamespace(convo=conversation, user=object(), col_ref=SimpleNamespace(collections=[]))
    message = AssistantMessage(content="", stop_reason="tool_use", tool_call_id="noop", tool_call_name=name, tool_call_input={})
    result = ToolMessage(tool_name=name, for_whom="assistant", content="unchanged", result_dict=payload)
    calls = []

    def call(current):
        calls.append(current)
        return result

    async def run():
        async with preservation_turn(consumer, max_func_calls=3):
            return await call_tool_async(SimpleNamespace(call_tool=call), message)

    with source_runtime_scope(runtime):
        observed = async_to_sync(run)()
    assert observed is result and observed.result_dict == payload and calls == [message]
    assert getattr(runtime, "_frozen_retrieval_scope", None) is None
    origins = getattr(runtime, "_recovery_origins", None)
    assert origins is None or origins.occurrences == []
    assert budget.sources_used == budget.actions_used == budget.planner_calls == 0
