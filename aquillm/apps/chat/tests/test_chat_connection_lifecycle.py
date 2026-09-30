"""Regression tests for WebSocket startup and resumed chat turns."""

from __future__ import annotations

from contextlib import asynccontextmanager
from json import dumps, loads
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from anthropic._exceptions import OverloadedError
from channels.db import database_sync_to_async
from django.contrib.auth import get_user_model

from apps.chat.consumers.chat import ChatConsumer
from apps.chat.consumers.chat_delta import send_conversation_delta
from apps.chat.models import WSConversation
from apps.chat.services.feedback import FEEDBACK_TEXT_MAX_LEN
from apps.chat.tests.chat_message_test_support import (
    _FakeLLMInterface,
    _test_document_ids,
)
from aquillm.llm import AssistantMessage, Conversation, LLMResponse, UserMessage
from aquillm.message_adapters import (
    ConversationConflictError,
    load_conversation_from_db,
    save_conversation_to_db,
)


@asynccontextmanager
async def _turn(*args, **kwargs):
    yield


@pytest.fixture(autouse=True)
def stub_execution_ownership_for_transport_unit_tests(monkeypatch):
    # These tests stub database loading/persistence; ownership has real database
    # and concurrent-tool coverage in test_chat_execution_stability.
    monkeypatch.setattr("apps.chat.consumers.chat.execution_turn", _turn)
    monkeypatch.setattr("apps.chat.consumers.chat_receive.execution_turn", _turn)


def _consumer(*, user_authenticated=True, db_convo=None):
    consumer = ChatConsumer()
    consumer.scope = {
        "user": SimpleNamespace(id=17, is_authenticated=user_authenticated),
        "url_route": {"kwargs": {"convo_id": 42}},
    }
    consumer.accept = AsyncMock()
    consumer.send = AsyncMock()
    consumer.close = AsyncMock()
    consumer._save_conversation = AsyncMock()
    consumer._ChatConsumer__get_convo = AsyncMock(return_value=db_convo)
    return consumer


def _db_convo():
    return SimpleNamespace(
        id=42, system_prompt="system", selected_collection_ids=[7], name="Saved chat"
    )


def _payloads(consumer):
    return [
        loads(call.kwargs.get("text_data", call.args[0] if call.args else "{}"))
        for call in consumer.send.await_args_list
    ]


@pytest.mark.asyncio
async def test_invalid_conversation_fails_fatally_and_closes_permanently():
    consumer = _consumer()

    await consumer.connect()

    assert consumer.dead is True
    assert any(payload.get("fatal") is True for payload in _payloads(consumer))
    consumer.close.assert_awaited_once_with(code=4404)


@pytest.mark.asyncio
async def test_unauthenticated_connection_closes_permanently():
    consumer = _consumer(user_authenticated=False)

    await consumer.connect()

    assert consumer.dead is True
    assert any(payload.get("fatal") is True for payload in _payloads(consumer))
    consumer.close.assert_awaited_once_with(code=4401)
    consumer._ChatConsumer__get_convo.assert_not_awaited()


@pytest.mark.asyncio
async def test_tool_factory_failure_closes_accepted_connection_as_server_error():
    consumer = _consumer(db_convo=_db_convo())

    with patch(
        "apps.chat.consumers.chat.build_document_tools",
        side_effect=RuntimeError("broken"),
    ):
        await consumer.connect()

    assert consumer.dead is True
    assert any(payload.get("fatal") is True for payload in _payloads(consumer))
    consumer.close.assert_awaited_once_with(code=1011)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "messages",
    [
        [],
        [
            UserMessage(content="hello"),
            AssistantMessage(content="done", stop_reason="end_turn"),
        ],
    ],
)
async def test_idle_initial_snapshot_skips_memory_spin_and_save(messages):
    convo = Conversation(system="system", messages=messages)
    consumer = _consumer(db_convo=_db_convo())

    with (
        patch("apps.chat.consumers.chat.build_document_tools", return_value=[]),
        patch("apps.chat.consumers.chat.build_memory_tools", return_value=[]),
        patch("apps.chat.consumers.chat.build_astronomy_tools", return_value=[]),
        patch("apps.chat.consumers.chat.SKILLS_ENABLED", False),
        patch(
            "apps.chat.consumers.chat.database_sync_to_async",
            side_effect=lambda fn: AsyncMock(return_value=convo),
        ),
        patch(
            "apps.chat.consumers.chat.augment_conversation_with_memory_async",
            new_callable=AsyncMock,
        ) as augment,
        patch("apps.chat.consumers.chat.run_llm_spin", new_callable=AsyncMock) as spin,
    ):
        await consumer.connect()

    assert any("conversation" in payload for payload in _payloads(consumer))
    augment.assert_not_awaited()
    spin.assert_not_awaited()
    consumer._save_conversation.assert_not_awaited()
    consumer.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_delta_without_new_messages_does_not_save():
    convo = Conversation(system="system", messages=[UserMessage(content="hello")])
    consumer = SimpleNamespace(
        convo=convo,
        last_sent_sequence=0,
        _save_conversation=AsyncMock(),
        send=AsyncMock(),
        transport_connected=True,
    )

    await send_conversation_delta(consumer, convo)

    consumer._save_conversation.assert_not_awaited()
    consumer.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_pending_tool_resume_rebinds_and_bypasses_new_turn_routing():
    convo = Conversation(
        system="system",
        messages=[
            UserMessage(content="Search the document"),
            AssistantMessage(
                content="",
                stop_reason="tool_use",
                tool_call_id="call-1",
                tool_call_name="lookup",
                tool_call_input={},
            ),
        ],
    )
    consumer = _consumer(db_convo=_db_convo())
    tool = SimpleNamespace(name="lookup")

    with (
        patch("apps.chat.consumers.chat.build_document_tools", return_value=[tool]),
        patch("apps.chat.consumers.chat.build_memory_tools", return_value=[]),
        patch("apps.chat.consumers.chat.build_astronomy_tools", return_value=[]),
        patch("apps.chat.consumers.chat.SKILLS_ENABLED", False),
        patch(
            "apps.chat.consumers.chat.database_sync_to_async",
            side_effect=lambda fn: AsyncMock(return_value=convo),
        ),
        patch(
            "apps.chat.consumers.chat.augment_conversation_with_memory_async",
            new_callable=AsyncMock,
        ),
        patch(
            "apps.chat.consumers.chat.effective_base_system_for_memory_async",
            new_callable=AsyncMock,
        ),
        patch("apps.chat.consumers.chat.preservation_turn", _turn),
        patch(
            "apps.chat.consumers.chat.run_direct_rag_turn", new_callable=AsyncMock
        ) as direct_rag,
        patch("apps.chat.consumers.chat.run_llm_spin", new_callable=AsyncMock) as spin,
    ):
        await consumer.connect()

    assert convo[-1].tools == [tool]
    direct_rag.assert_not_awaited()
    spin.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("retry", [False, True])
async def test_reloaded_pending_user_restores_current_local_tool_for_provider(retry):
    messages = [UserMessage(content="Use the tool to inspect this FITS file")]
    if retry:
        messages.extend(
            [
                AssistantMessage(
                    content="Could not inspect it", stop_reason="end_turn"
                ),
                UserMessage(content="Try again"),
            ]
        )
    convo = Conversation(system="system", messages=messages)
    consumer = _consumer(db_convo=_db_convo())
    llm = _FakeLLMInterface(
        [
            LLMResponse(
                text="finished",
                tool_call={},
                stop_reason="end_turn",
                input_usage=1,
                output_usage=1,
            )
        ]
    )
    consumer.llm_if = llm

    with (
        patch("apps.chat.consumers.chat.build_document_tools", return_value=[]),
        patch("apps.chat.consumers.chat.build_memory_tools", return_value=[]),
        patch(
            "apps.chat.consumers.chat.build_astronomy_tools",
            return_value=[_test_document_ids],
        ),
        patch("apps.chat.consumers.chat.SKILLS_ENABLED", False),
        patch("apps.chat.consumers.chat.DEBUG", False),
        patch(
            "apps.chat.consumers.chat.database_sync_to_async",
            side_effect=lambda fn: AsyncMock(return_value=convo),
        ),
        patch(
            "apps.chat.consumers.chat.augment_conversation_with_memory_async",
            new_callable=AsyncMock,
        ),
        patch(
            "apps.chat.consumers.chat.effective_base_system_for_memory_async",
            new_callable=AsyncMock,
        ),
        patch("apps.chat.consumers.chat.preservation_turn", _turn),
        patch(
            "apps.chat.consumers.chat.run_direct_rag_turn",
            new_callable=AsyncMock,
            return_value="skipped",
        ),
    ):
        await consumer.connect()

    assert consumer.dead is False
    assert llm.calls[0]["tools"][0]["name"] == "_test_document_ids"
    assert llm.calls[0]["tool_choice"] == {"type": "auto"}


@pytest.mark.asyncio
async def test_pending_document_request_keeps_direct_rag_routing():
    convo = Conversation(
        system="system",
        messages=[UserMessage(content="Search these papers for calibration notes")],
    )
    consumer = _consumer(db_convo=_db_convo())

    with (
        patch(
            "apps.chat.consumers.chat.build_document_tools",
            return_value=[_test_document_ids],
        ),
        patch("apps.chat.consumers.chat.build_memory_tools", return_value=[]),
        patch("apps.chat.consumers.chat.build_astronomy_tools", return_value=[]),
        patch("apps.chat.consumers.chat.SKILLS_ENABLED", False),
        patch("apps.chat.consumers.chat.DEBUG", False),
        patch(
            "apps.chat.consumers.chat.database_sync_to_async",
            side_effect=lambda fn: AsyncMock(return_value=convo),
        ),
        patch(
            "apps.chat.consumers.chat.augment_conversation_with_memory_async",
            new_callable=AsyncMock,
        ),
        patch(
            "apps.chat.consumers.chat.effective_base_system_for_memory_async",
            new_callable=AsyncMock,
        ),
        patch("apps.chat.consumers.chat.preservation_turn", _turn),
        patch(
            "apps.chat.consumers.chat.run_direct_rag_turn",
            new_callable=AsyncMock,
            return_value="handled",
        ) as direct_rag,
        patch(
            "apps.chat.consumers.chat.send_conversation_delta", new_callable=AsyncMock
        ),
        patch("apps.chat.consumers.chat.run_llm_spin", new_callable=AsyncMock) as spin,
    ):
        await consumer.connect()

    assert convo[-1].tools == [_test_document_ids]
    assert convo[-1].tool_choice.type == "any"
    direct_rag.assert_awaited_once()
    spin.assert_not_awaited()


@pytest.mark.asyncio
async def test_startup_conflict_closes_with_refresh_required():
    convo = Conversation(system="system", messages=[UserMessage(content="hello")])
    consumer = _consumer(db_convo=_db_convo())
    consumer._save_conversation.side_effect = ConversationConflictError(
        "stale snapshot"
    )

    async def complete_spin(*args, **kwargs):
        convo.messages.append(AssistantMessage(content="reply", stop_reason="end_turn"))

    with (
        patch("apps.chat.consumers.chat.build_document_tools", return_value=[]),
        patch("apps.chat.consumers.chat.build_memory_tools", return_value=[]),
        patch("apps.chat.consumers.chat.build_astronomy_tools", return_value=[]),
        patch("apps.chat.consumers.chat.SKILLS_ENABLED", False),
        patch(
            "apps.chat.consumers.chat.database_sync_to_async",
            side_effect=lambda fn: AsyncMock(return_value=convo),
        ),
        patch(
            "apps.chat.consumers.chat.augment_conversation_with_memory_async",
            new_callable=AsyncMock,
        ),
        patch(
            "apps.chat.consumers.chat.effective_base_system_for_memory_async",
            new_callable=AsyncMock,
        ),
        patch("apps.chat.consumers.chat.preservation_turn", _turn),
        patch(
            "apps.chat.consumers.chat.run_direct_rag_turn",
            new_callable=AsyncMock,
            return_value="skipped",
        ),
        patch(
            "apps.chat.consumers.chat.run_llm_spin",
            new_callable=AsyncMock,
            side_effect=complete_spin,
        ),
    ):
        await consumer.connect()

    assert consumer.dead is True
    assert any(
        payload.get("fatal") is True and "refresh" in payload["exception"].lower()
        for payload in _payloads(consumer)
    )
    consumer.close.assert_awaited_once_with(code=4409)


@pytest.mark.asyncio
async def test_receive_conflict_closes_with_refresh_required():
    db_convo = _db_convo()
    db_convo.save = lambda **kwargs: None
    consumer = _consumer(db_convo=db_convo)
    consumer._chat_accepted = True
    consumer.dead = False
    consumer.user = consumer.scope["user"]
    consumer.db_convo = db_convo
    consumer.convo = Conversation(system="system", messages=[])
    consumer.doc_tools = []
    consumer.tools = []
    consumer._save_conversation.side_effect = ConversationConflictError(
        "stale snapshot"
    )

    await consumer.receive(
        dumps(
            {
                "action": "append",
                "message": {"role": "user", "content": "new message"},
                "collections": [],
            }
        )
    )

    assert consumer.dead is True
    assert any(
        payload.get("fatal") is True and "refresh" in payload["exception"].lower()
        for payload in _payloads(consumer)
    )
    consumer.close.assert_awaited_once_with(code=4409)


@pytest.mark.asyncio
async def test_overloaded_resume_closes_with_retryable_code():
    convo = Conversation(system="system", messages=[UserMessage(content="hello")])
    consumer = _consumer(db_convo=_db_convo())
    error = OverloadedError(
        "overloaded",
        response=httpx.Response(
            529, request=httpx.Request("POST", "https://example.invalid")
        ),
        body=None,
    )

    with (
        patch("apps.chat.consumers.chat.build_document_tools", return_value=[]),
        patch("apps.chat.consumers.chat.build_memory_tools", return_value=[]),
        patch("apps.chat.consumers.chat.build_astronomy_tools", return_value=[]),
        patch("apps.chat.consumers.chat.SKILLS_ENABLED", False),
        patch(
            "apps.chat.consumers.chat.database_sync_to_async",
            side_effect=lambda fn: AsyncMock(return_value=convo),
        ),
        patch(
            "apps.chat.consumers.chat.augment_conversation_with_memory_async",
            new_callable=AsyncMock,
        ),
        patch(
            "apps.chat.consumers.chat.effective_base_system_for_memory_async",
            new_callable=AsyncMock,
        ),
        patch("apps.chat.consumers.chat.preservation_turn", _turn),
        patch(
            "apps.chat.consumers.chat.run_direct_rag_turn",
            new_callable=AsyncMock,
            return_value="skipped",
        ),
        patch(
            "apps.chat.consumers.chat.run_llm_spin",
            new_callable=AsyncMock,
            side_effect=error,
        ),
    ):
        await consumer.connect()

    assert consumer.dead is True
    assert any(payload.get("fatal") is True for payload in _payloads(consumer))
    consumer.close.assert_awaited_once_with(code=1013)


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
async def test_long_feedback_then_append_preserves_canonical_feedback():
    user = await database_sync_to_async(get_user_model().objects.create_user)(
        username="long-feedback-append", password="pass"
    )
    db_convo = await WSConversation.objects.acreate(
        owner=user, system_prompt="system", name="Saved chat"
    )
    seeded = Conversation(
        system="system",
        messages=[
            UserMessage(content="original"),
            AssistantMessage(content="answer", stop_reason="end_turn"),
        ],
    )
    await database_sync_to_async(save_conversation_to_db)(seeded, db_convo)
    loaded = await database_sync_to_async(load_conversation_from_db)(db_convo)
    consumer = ChatConsumer()
    consumer.user = user
    consumer.db_convo = db_convo
    consumer.convo = loaded
    consumer.dead = False
    consumer._chat_accepted = True
    consumer.send = AsyncMock()
    consumer.close = AsyncMock()
    consumer.doc_tools = []
    consumer.memory_tools = []
    consumer.tools = []
    consumer.last_sent_sequence = len(loaded) - 1

    with (
        patch(
            "apps.chat.consumers.chat_receive.augment_conversation_with_memory_async",
            new_callable=AsyncMock,
        ),
        patch(
            "apps.chat.consumers.chat_receive.effective_base_system_for_memory_async",
            new_callable=AsyncMock,
        ),
        patch("apps.chat.consumers.chat_receive.preservation_turn", _turn),
        patch(
            "apps.chat.consumers.chat_receive.run_direct_rag_turn",
            new_callable=AsyncMock,
            return_value="skipped",
        ),
        patch("apps.chat.consumers.chat_receive.run_llm_spin", new_callable=AsyncMock),
        patch("apps.chat.consumers.chat.enqueue_conversation_memories_task"),
        patch("apps.chat.consumers.chat.enqueue_index_conversation_task"),
    ):
        await consumer.receive(
            dumps(
                {
                    "action": "feedback",
                    "uuid": str(loaded[-1].message_uuid),
                    "feedback_text": " x " * (FEEDBACK_TEXT_MAX_LEN + 50),
                }
            )
        )
        await consumer.receive(
            dumps(
                {
                    "action": "append",
                    "message": {"role": "user", "content": "follow up"},
                    "collections": [],
                }
            )
        )

    assert consumer.dead is False
    assert consumer.close.await_count == 0
    assert consumer.convo[-1].content == "follow up"
    rows = await database_sync_to_async(list)(
        db_convo.db_messages.order_by("sequence_number")
    )
    assert len(rows) == 3
    assert len(rows[1].feedback_text) == FEEDBACK_TEXT_MAX_LEN
    assert consumer.convo[1].feedback_text == rows[1].feedback_text
