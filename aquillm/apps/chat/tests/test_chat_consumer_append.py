"""Regression tests for ChatConsumer append handling."""
import json
from unittest.mock import AsyncMock, patch

import pytest

from django.contrib.auth import get_user_model

from aquillm.llm import Conversation
from aquillm.models import WSConversation
from apps.chat.consumers.chat import ChatConsumer, CollectionsRef
from apps.chat.tests.chat_message_test_support import _test_document_ids, _test_image_result_tool

User = get_user_model()


@pytest.mark.asyncio
@pytest.mark.django_db
@patch("apps.chat.consumers.chat.create_conversation_memories_task.delay")
@patch("apps.chat.consumers.chat.augment_conversation_with_memory_async", new_callable=AsyncMock)
async def test_append_without_files_does_not_raise(_augment, _mem_task):
    user = User.objects.create_user(username="appendtest", password="pass")
    db_convo = WSConversation.objects.create(owner=user, system_prompt="sys")

    consumer = ChatConsumer()
    consumer.base_send = AsyncMock()
    consumer.scope = {"user": user, "url_route": {"kwargs": {"convo_id": db_convo.id}}}
    consumer.user = user
    consumer.db_convo = db_convo
    consumer.convo = Conversation(system="sys", messages=[])
    consumer.dead = False
    consumer.col_ref = CollectionsRef([])
    consumer.doc_tools = []
    consumer.tools = []
    consumer.last_sent_sequence = -1
    consumer.llm_if = AsyncMock()
    consumer.llm_if.spin = AsyncMock()

    payload = json.dumps(
        {
            "action": "append",
            "message": {"role": "user", "content": "hello"},
            "collections": [],
        }
    )

    await consumer.receive(payload)

    assert consumer.convo is not None
    assert len(consumer.convo.messages) >= 1
    assert consumer.convo[-1].files == []


@pytest.mark.asyncio
@pytest.mark.django_db
@patch("apps.chat.consumers.chat.create_conversation_memories_task.delay")
@patch("apps.chat.consumers.chat_receive.augment_conversation_with_memory_async")
async def test_append_without_selected_collections_keeps_doc_tools(_augment, _mem_task):
    user = User.objects.create_user(username="appendtools", password="pass")
    db_convo = WSConversation.objects.create(owner=user, system_prompt="sys")

    consumer = ChatConsumer()
    consumer.base_send = AsyncMock()
    consumer.scope = {"user": user, "url_route": {"kwargs": {"convo_id": db_convo.id}}}
    consumer.user = user
    consumer.db_convo = db_convo
    consumer.convo = Conversation(system="sys", messages=[])
    consumer.dead = False
    consumer.col_ref = CollectionsRef([])
    consumer.doc_tools = [_test_document_ids]
    consumer.tools = [_test_document_ids, _test_image_result_tool]
    consumer.last_sent_sequence = -1
    consumer.llm_if = AsyncMock()
    consumer.llm_if.spin = AsyncMock()

    payload = json.dumps(
        {
            "action": "append",
            "message": {"role": "user", "content": "brand new chat"},
            "collections": [],
        }
    )

    await consumer.receive(payload)

    assert consumer.convo is not None
    assert len(consumer.convo.messages) >= 1
    tool_names = [tool.name for tool in (consumer.convo[-1].tools or [])]
    assert "_test_document_ids" in tool_names
