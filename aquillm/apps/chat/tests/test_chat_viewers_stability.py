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


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_rejected_stale_append_does_not_change_saved_collections():
    user = await get_user_model().objects.acreate(username="atomic-append")
    db = await WSConversation.objects.acreate(
        owner=user, name="existing", system_prompt="sys", selected_collection_ids=[7]
    )
    original = Conversation(
        system="sys",
        messages=[
            UserMessage(content="question"),
            AssistantMessage(content="answer", stop_reason="end_turn"),
        ],
    )
    await database_sync_to_async(save_conversation_to_db)(original, db)
    stale_db = await WSConversation.objects.aget(pk=db.pk)
    stale = await database_sync_to_async(load_conversation_from_db)(stale_db)
    current = original + UserMessage(content="newer question")
    await database_sync_to_async(save_conversation_to_db)(current, db)
    consumer = ChatConsumer()
    consumer.user, consumer.db_convo, consumer.convo = user, stale_db, stale
    consumer.tools = consumer.doc_tools = consumer.memory_tools = []
    consumer._chat_accepted = True
    consumer.send, consumer.close = AsyncMock(), AsyncMock()
    await consumer.receive(
        dumps(
            {
                "action": "append",
                "message": {"role": "user", "content": "stale prompt"},
                "collections": [99],
            }
        )
    )
    await db.arefresh_from_db()
    assert db.selected_collection_ids == [7]
    assert await db.db_messages.acount() == 3


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_second_viewer_waits_then_loads_completed_turn_without_replay(
    monkeypatch,
):
    from contextlib import asynccontextmanager
    from json import loads
    from apps.chat.consumers import chat

    user = await get_user_model().objects.acreate(username="pending-viewers")
    db = await WSConversation.objects.acreate(
        owner=user, system_prompt="sys", name="existing", selected_collection_ids=[7]
    )
    await database_sync_to_async(save_conversation_to_db)(
        Conversation(system="sys", messages=[UserMessage(content="question")]), db
    )
    entered, release = asyncio.Event(), asyncio.Event()
    invocations = []

    @asynccontextmanager
    async def turn(*args, **kwargs):
        yield

    async def spin(consumer, llm, convo, **kwargs):
        invocations.append(consumer)
        entered.set()
        await release.wait()
        consumer.convo = convo + AssistantMessage(
            content="Completed response", stop_reason="end_turn"
        )
        await kwargs["send_func"](consumer.convo)

    for name in ("build_document_tools", "build_memory_tools", "build_astronomy_tools"):
        monkeypatch.setattr(chat, name, lambda *args: [])
    monkeypatch.setattr(chat, "SKILLS_ENABLED", False)
    monkeypatch.setattr(chat, "preservation_turn", turn)
    monkeypatch.setattr(chat, "run_llm_spin", spin)
    monkeypatch.setattr(chat, "run_direct_rag_turn", AsyncMock(return_value="skipped"))
    monkeypatch.setattr(chat, "augment_conversation_with_memory_async", AsyncMock())
    monkeypatch.setattr(
        chat, "effective_base_system_for_memory_async", AsyncMock(return_value="sys")
    )
    monkeypatch.setattr(
        chat, "enqueue_conversation_memories_task", lambda **kwargs: None
    )
    monkeypatch.setattr(chat, "enqueue_index_conversation_task", lambda **kwargs: None)
    consumers = []
    snapshots_ready = []
    for _ in range(2):
        consumer = ChatConsumer()
        consumer.scope = {"user": user, "url_route": {"kwargs": {"convo_id": db.pk}}}
        snapshot_ready = asyncio.Event()
        snapshots_ready.append(snapshot_ready)
        consumer.accept, consumer.close = AsyncMock(), AsyncMock()
        consumer.send = AsyncMock(
            side_effect=lambda _ready=snapshot_ready, **kwargs: _ready.set()
        )
        consumers.append(consumer)
    first = asyncio.create_task(consumers[0].connect())
    await asyncio.wait_for(entered.wait(), 2)
    second = asyncio.create_task(consumers[1].connect())
    try:
        await asyncio.wait_for(snapshots_ready[1].wait(), 2)
        assert consumers[1].send.called and not consumers[1].close.called
        assert len(invocations) == 1
        await WSConversation.objects.filter(pk=db.pk).aupdate(
            selected_collection_ids=[9]
        )
        release.set()
        await asyncio.wait_for(asyncio.gather(first, second), 3)
        snapshots = [
            loads(call.kwargs["text_data"])["conversation"]
            for call in consumers[1].send.await_args_list
        ]
        assert len(snapshots[-1]["messages"]) == 2
        assert snapshots[-1]["selected_collections"] == [9]
        assert consumers[1].col_ref.collections == [9]
        assert len(invocations) == 1
    finally:
        release.set()
        await asyncio.gather(first, second)
