"""Real Channels dispatch must deliver disconnect during an owned turn."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from asgiref.testing import ApplicationCommunicator

from apps.chat.tests.test_direct_rag_websocket_smoke import _append_payload
from apps.chat.tests.test_rag_preservation_routes import configure, fake_search
from apps.chat.tests.test_rag_source_document_tools import docs as docs
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, UserMessage


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize("entry", ["connect", "append"])
async def test_asgi_disconnect_cancels_blocked_sdk_before_late_save(
    docs, monkeypatch, settings, entry
):
    from apps.chat.consumers import chat, chat_receive
    from lib.llm.providers.openai import OpenAIInterface

    configure(monkeypatch)
    fake_search(monkeypatch)
    settings.CHANNEL_LAYERS = {
        "default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}
    }
    user, doc, chunks, _ = docs
    started, released, cancelled, initial_saved = (asyncio.Event() for _ in range(4))

    async def create(**kwargs):
        started.set()
        try:
            await released.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=f"Only below 1 Pa [doc:{doc.id} chunk:{chunks[0].pk}].",
                        tool_calls=None,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )

    consumer = chat.ChatConsumer()
    consumer.llm_if = OpenAIInterface(
        SimpleNamespace(
            base_url="http://vllm:8000",
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        ),
        "test",
    )
    db = SimpleNamespace(
        id=1,
        selected_collection_ids=[doc.collection_id],
        system_prompt="sys",
        name="existing",
    )
    db.save = lambda **kwargs: None
    consumer._ChatConsumer__get_all_user_collections = AsyncMock()
    consumer._ChatConsumer__get_convo = AsyncMock(return_value=db)
    consumer._save_conversation = AsyncMock(
        side_effect=lambda **kw: initial_saved.set()
    )
    pending = Conversation(
        system="sys",
        messages=[
            UserMessage(content="Summarize selected documents")
            if entry == "connect"
            else AssistantMessage(content="Previous answer", stop_reason="end_turn")
        ],
    )
    monkeypatch.setattr(chat, "load_conversation_from_db", lambda _: pending)
    for name in ["build_document_tools", "build_astronomy_tools", "build_memory_tools"]:
        monkeypatch.setattr(chat, name, lambda *args: [])
    for module in [chat, chat_receive]:
        monkeypatch.setattr(
            module, "augment_conversation_with_memory_async", AsyncMock()
        )
        monkeypatch.setattr(
            module,
            "effective_base_system_for_memory_async",
            AsyncMock(return_value="sys"),
        )
    app = ApplicationCommunicator(
        consumer,
        {
            "type": "websocket",
            "path": "/chat/1",
            "user": user,
            "url_route": {"kwargs": {"convo_id": 1}},
        },
    )
    try:
        await app.send_input({"type": "websocket.connect"})
        assert (await app.receive_output(2))["type"] == "websocket.accept"
        if entry == "append":
            await asyncio.wait_for(initial_saved.wait(), 2)
            await app.send_input(
                {
                    "type": "websocket.receive",
                    "text": _append_payload(
                        "Summarize selected documents", [doc.collection_id]
                    ),
                }
            )
        await asyncio.wait_for(started.wait(), 5)
        saved_before = consumer._save_conversation.call_count
        await app.send_input({"type": "websocket.disconnect", "code": 1000})
        done, _ = await asyncio.wait({app.future}, timeout=0.75)
        assert app.future in done, (
            "serialized ASGI dispatch prevented disconnect cancellation"
        )
        assert cancelled.is_set()
        assert consumer._save_conversation.call_count == saved_before
        assert consumer._active_evidence_turn is None
    finally:
        released.set()
        if not app.future.done():
            await app.send_input({"type": "websocket.disconnect", "code": 1000})
        # Cleanup permits the old implementation to finish naturally, never using
        # communicator.wait(), which would forcibly cancel the app on timeout.
        await asyncio.wait_for(asyncio.shield(app.future), 5)


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize("active", [False, True])
async def test_actual_dispatch_serializes_turns_and_keeps_context_isolated(
    docs, monkeypatch, settings, active
):
    from apps.chat.consumers.chat import ChatConsumer
    from apps.chat.services.rag_turn import preservation_turn
    from apps.documents.services.source_loading import current_source_runtime

    configure(monkeypatch, source=active)
    settings.CHANNEL_LAYERS = {
        "default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}
    }
    user, doc, _, _ = docs
    entered, release = asyncio.Event(), asyncio.Event()
    events, ledgers = [], []

    class Consumer(ChatConsumer):
        async def connect(self):
            self.user = user
            self.col_ref.collections = [doc.collection_id]
            await self.accept()

        async def receive(self, text_data=None, bytes_data=None):
            assert current_source_runtime() is None
            self.convo = Conversation(
                system="sys", messages=[UserMessage(content=text_data)]
            )
            async with preservation_turn(self, max_func_calls=3) as runtime:
                ledgers.append(runtime.budget if runtime else None)
                events.append("start " + text_data)
                if text_data == "one":
                    entered.set()
                    await release.wait()
                events.append("end " + text_data)
            assert current_source_runtime() is None
            await self.send(text_data=text_data)

    consumer = Consumer()
    app = ApplicationCommunicator(consumer, {"type": "websocket", "path": "/chat/1"})
    try:
        await app.send_input({"type": "websocket.connect"})
        assert (await app.receive_output(2))["type"] == "websocket.accept"
        await app.send_input({"type": "websocket.receive", "text": "one"})
        await asyncio.wait_for(entered.wait(), 2)
        await app.send_input({"type": "websocket.receive", "text": "two"})
        assert events == ["start one"]
        release.set()
        assert (await app.receive_output(2))["text"] == "one"
        assert (await app.receive_output(2))["text"] == "two"
        assert events == ["start one", "end one", "start two", "end two"]
        if active:
            assert ledgers[0] is not ledgers[1]
            assert all(not ledger.can_publish() for ledger in ledgers)
        else:
            assert ledgers == [None, None]
            assert getattr(consumer, "_chat_event_owner", None) is None
    finally:
        release.set()
        await app.send_input({"type": "websocket.disconnect", "code": 1000})
        await asyncio.wait_for(asyncio.shield(app.future), 2)
