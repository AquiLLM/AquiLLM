"""Spent retrieval turns terminate honestly at the real tool/provider boundaries."""

from types import SimpleNamespace

import pytest
from django.contrib.auth import get_user_model

from apps.chat.models import ToolExecution, WSConversation
from apps.chat.services import rag_pipeline
from apps.chat.services.execution import execution_turn
from apps.chat.services.rag_answer_publication import legacy_tool_handoff
from apps.chat.tests.persisted_direct_test_support import (
    close_direct_worker as close_direct_worker,
)
from aquillm.llm import llm_tool
from lib.llm.answer_publication import LIMITED_ANSWER
from lib.llm.execution_context import uncertain_tool_message
from lib.llm.providers.openai import OpenAIInterface
from lib.llm.turn_context import TurnContext, bind_turn, call_tool_async
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, UserMessage
from lib.llm.types.tools import ToolChoice
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


pytestmark = pytest.mark.asyncio


@pytest.mark.django_db(transaction=True)
async def test_expired_bound_retrieval_never_dispatches_or_claims_uncertain_effects(
    close_direct_worker,
):
    user = await get_user_model().objects.acreate(username="expired-bound-retrieval")
    db = await WSConversation.objects.acreate(owner=user, system_prompt="sys")
    now = [0.0]
    budget = TurnBudget(TurnLimits(), clock=lambda: now[0])
    now[0] = 16.0
    dispatched = []

    @llm_tool(for_whom="assistant", required=[], param_descs={})
    def vector_search():
        """Search the selected documents."""
        dispatched.append(True)
        return {"result": []}

    message = AssistantMessage(
        content="", stop_reason="tool_use", tools=[vector_search],
        tool_choice=ToolChoice(type="auto"), tool_call_id="expired-search",
        tool_call_name="vector_search", tool_call_input={},
    )
    async with execution_turn(SimpleNamespace(db_convo=db)):
        with bind_turn(TurnContext(budget)):
            result = await call_tool_async(OpenAIInterface(None, "test"), message)
    assert dispatched == []
    assert not await ToolExecution.objects.filter(conversation=db).aexists()
    assert result.for_whom == "assistant"
    assert result.result_dict == {"result": [], "retrieval_status": "context_limited"}
    assert "may still be running" not in result.content


@pytest.mark.parametrize("expired", [True, False])
async def test_direct_retrieval_timeout_at_spent_deadline_finishes_handled(
    monkeypatch, expired,
):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    monkeypatch.setenv("RAG_EVIDENCE_SELECTION_MODE", "legacy")
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "legacy")
    monkeypatch.setenv("RAG_ITERATIVE_RETRIEVAL_ENABLED", "0")
    monkeypatch.setenv("RAG_QUERY_REWRITE_ENABLED", "0")
    monkeypatch.setenv("RAG_VOCABULARY_QUERY_ENABLED", "0")
    convo = Conversation(system="sys", messages=[
        UserMessage(content="Search selected documents for calibration"),
    ])
    consumer = SimpleNamespace(col_ref=SimpleNamespace(collections=[1]), convo=convo)
    now = [0.0]
    budget = TurnBudget(TurnLimits(), clock=lambda: now[0])

    async def timeout(*args, **kwargs):
        now[0] = 16.0 if expired else 1.0
        raise TimeoutError("retrieval deadline")

    monkeypatch.setattr(rag_pipeline, "run_parallel_searches", timeout)
    with bind_turn(TurnContext(budget)):
        outcome = await rag_pipeline.run_direct_rag_turn(consumer, object(), convo)
        if expired:
            assert outcome == "handled"
            assert consumer.convo[-1].content == LIMITED_ANSWER
            assert consumer.convo[-1].stop_reason == "end_turn"
            assert budget.remaining_ms() == 0
        else:
            assert outcome == "skipped"
            assert consumer.convo is convo


async def test_user_facing_uncertain_retrieval_terminates_real_provider_completion():
    message = AssistantMessage(
        content="", stop_reason="tool_use", tool_call_id="uncertain-search",
        tool_call_name="vector_search", tool_call_input={"search_string": "calibration"},
    )
    convo = Conversation(system="sys", messages=[
        UserMessage(content="Search selected documents for calibration"),
        message, uncertain_tool_message(message),
    ])
    consumer = SimpleNamespace(col_ref=SimpleNamespace(collections=[1]))

    async def handoff(llm, working, max_tokens, stream_func):
        return await legacy_tool_handoff(consumer, llm, working, max_tokens, stream_func)

    with bind_turn(TurnContext(TurnBudget(TurnLimits()), handoff=handoff)):
        result, status = await OpenAIInterface(None, "test").complete(convo, 512)
    assert status == "unchanged"
    assert result.model_dump() == convo.model_dump()
    assert result[-1].for_whom == "user"
    assert "not run again" in result[-1].content
