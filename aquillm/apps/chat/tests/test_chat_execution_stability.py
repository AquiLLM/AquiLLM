"""Regression coverage for cancellation, ownership and atomic append acceptance."""

import asyncio
from json import dumps
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from asgiref.testing import ApplicationCommunicator
from channels.db import database_sync_to_async
from django.contrib.auth import get_user_model

from apps.chat.consumers.chat import ChatConsumer
from apps.chat.models import WSConversation
from aquillm.llm import AssistantMessage, Conversation, UserMessage
from aquillm.message_adapters import load_conversation_from_db, save_conversation_to_db


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["legacy", "source"])
async def test_disconnect_cancels_work_independently_of_preservation_mode(
    monkeypatch, mode
):
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", mode)
    entered, released, cancelled = (asyncio.Event() for _ in range(3))

    class Consumer(ChatConsumer):
        async def connect(self):
            await self.accept()

        async def receive(self, text_data):
            entered.set()
            try:
                await released.wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

    app = ApplicationCommunicator(Consumer(), {"type": "websocket", "path": "/chat/1"})
    try:
        await app.send_input({"type": "websocket.connect"})
        await app.receive_output(2)
        await app.send_input({"type": "websocket.receive", "text": "run"})
        await asyncio.wait_for(entered.wait(), 2)
        await app.send_input({"type": "websocket.disconnect", "code": 1000})
        done, _ = await asyncio.wait({app.future}, timeout=0.5)
        assert app.future in done
        assert cancelled.is_set()
    finally:
        released.set()
        await asyncio.wait_for(asyncio.shield(app.future), 2)


@pytest.mark.django_db(transaction=True)
def test_execution_claim_is_exclusive_and_expired_writer_cannot_save():
    from apps.chat.services.execution import claim_execution, release_execution
    from aquillm.message_adapters import ConversationConflictError

    user = get_user_model().objects.create(username="exclusive-owner")
    db = WSConversation.objects.create(owner=user, system_prompt="sys", name="existing")
    first = claim_execution(db.pk)
    assert first is not None
    assert claim_execution(db.pk) is None
    db._execution_token = first
    release_execution(db.pk, first)
    assert claim_execution(db.pk) is not None
    with pytest.raises(ConversationConflictError):
        save_conversation_to_db(
            Conversation(system="sys", messages=[UserMessage(content="late")]), db
        )
    assert db.db_messages.count() == 0


@pytest.mark.django_db(transaction=True)
def test_tool_receipt_never_replays_uncertain_effect_and_reuses_completed_result():
    from apps.chat.services.execution import Execution, claim_execution
    from lib.llm.execution_context import (
        bind_execution,
        run_tool_once,
        UncertainToolExecution,
    )

    user = get_user_model().objects.create(username="tool-receipts")
    db = WSConversation.objects.create(owner=user, system_prompt="sys", name="existing")
    execution = Execution(db.pk, claim_execution(db.pk))
    message = AssistantMessage(
        content="",
        stop_reason="tool_use",
        tool_call_id="call-1",
        tool_call_name="operation",
        tool_call_input={},
    )
    effects = []

    def operation():
        effects.append("effect")
        return {"result": "done"}

    with bind_execution(execution):
        assert run_tool_once(message, operation) == {"result": "done"}
        assert run_tool_once(message, operation) == {"result": "done"}
    assert effects == ["effect"]
    from apps.chat.models import ToolExecution

    ToolExecution.objects.filter(conversation=db, call_id="call-1").update(result=None)
    with bind_execution(execution), pytest.raises(UncertainToolExecution):
        run_tool_once(message, operation)
    assert effects == ["effect"]


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_cancelled_sync_tool_is_not_replayed_by_reconnected_owner():
    from threading import Event
    from apps.chat.services.execution import execution_turn
    from apps.chat.tests.chat_message_test_support import _FakeLLMInterface
    from aquillm.llm import LLMTool
    from lib.llm.turn_context import call_tool_async

    user = await get_user_model().objects.acreate(username="cancelled-sync-tool")
    db = await WSConversation.objects.acreate(
        owner=user, system_prompt="sys", name="existing"
    )
    started, release = Event(), Event()
    effects = []

    def operation():
        effects.append("effect")
        started.set()
        assert release.wait(5)
        return {"result": "completed"}

    tool = LLMTool(
        llm_definition={
            "name": "operation",
            "description": "operation",
            "input_schema": {"type": "object", "properties": {}},
        },
        for_whom="assistant",
        _function=operation,
    )
    message = AssistantMessage(
        content="",
        stop_reason="tool_use",
        tool_call_id="saved-call",
        tool_call_name="operation",
        tool_call_input={},
    )
    message.tools = [tool]
    try:
        async with execution_turn(SimpleNamespace(db_convo=db)):
            running = asyncio.create_task(
                call_tool_async(_FakeLLMInterface([]), message)
            )
            assert await asyncio.to_thread(started.wait, 2)
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
        async with execution_turn(SimpleNamespace(db_convo=db)):
            result = await call_tool_async(_FakeLLMInterface([]), message)
            assert result.for_whom == "user"
            assert "not run again" in result.result_dict["exception"]
            assert effects == ["effect"]
    finally:
        release.set()


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_tool_exception_after_side_effect_terminates_automatic_loop():
    from apps.chat.services.execution import execution_turn
    from apps.chat.tests.chat_message_test_support import _FakeLLMInterface
    from aquillm.llm import LLMTool
    from lib.llm.turn_context import call_tool_async

    user = await get_user_model().objects.acreate(username="uncertain-exception")
    db = await WSConversation.objects.acreate(
        owner=user, system_prompt="sys", name="existing"
    )
    effects = []

    def operation():
        effects.append("effect")
        raise RuntimeError("transport failed after side effect")

    tool = LLMTool(
        llm_definition={
            "name": "operation",
            "description": "operation",
            "input_schema": {"type": "object", "properties": {}},
        },
        for_whom="assistant",
        _function=operation,
    )
    message = AssistantMessage(
        content="",
        stop_reason="tool_use",
        tool_call_id="failed-call",
        tool_call_name="operation",
        tool_call_input={},
    )
    message.tools = [tool]
    async with execution_turn(SimpleNamespace(db_convo=db)):
        result = await call_tool_async(_FakeLLMInterface([]), message)
    assert result.for_whom == "user"
    assert "not run again" in result.result_dict["exception"]
    assert effects == ["effect"]


@pytest.mark.django_db(transaction=True)
def test_simultaneous_database_claims_have_one_owner():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from django.db import connections
    from apps.chat.services.execution import claim_execution

    user = get_user_model().objects.create(username="claim-race")
    db = WSConversation.objects.create(owner=user, system_prompt="sys", name="existing")
    barrier = Barrier(2)

    def claim():
        try:
            barrier.wait(timeout=3)
            return claim_execution(db.pk)
        finally:
            connections.close_all()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: claim(), range(2)))
    assert sum(token is not None for token in results) == 1
