"""Regression tests for ChatConsumer append handling."""
import json
from unittest.mock import AsyncMock, patch

import pytest
from channels.db import database_sync_to_async
from django.contrib.auth import get_user_model
from django.test import override_settings

from apps.chat.consumers.chat import ChatConsumer, CollectionsRef
from apps.chat.consumers.chat_delta import send_conversation_snapshot
from apps.chat.tests.chat_message_test_support import (
    _test_document_ids,
    _test_image_result_tool,
)
from apps.collections.models import Collection, CollectionPermission
from apps.documents.models import RawTextDocument
from aquillm.llm import Conversation
from aquillm.models import WSConversation

User = get_user_model()


def test_collection_selection_is_isolated_per_websocket_consumer():
    first = ChatConsumer()
    second = ChatConsumer()

    first.col_ref.collections = [101, 202, 303]

    assert second.col_ref.collections == []


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
@patch("apps.chat.consumers.chat.enqueue_conversation_memories_task")
@patch("apps.chat.consumers.chat_receive.augment_conversation_with_memory_async", new_callable=AsyncMock)
async def test_append_without_files_does_not_raise(_augment, _mem_task):
    _augment.side_effect = lambda convo, *args, **kwargs: convo
    user = await database_sync_to_async(User.objects.create_user)(username="appendtest", password="pass")
    db_convo = await WSConversation.objects.acreate(owner=user, system_prompt="sys")

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

    consumer.llm_if.spin.assert_awaited_once()
    assert consumer.convo is not None
    assert len(consumer.convo.messages) >= 1
    assert consumer.convo[-1].files == []


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
@patch("apps.chat.consumers.chat.enqueue_conversation_memories_task")
@patch("apps.chat.consumers.chat_receive.augment_conversation_with_memory_async")
async def test_append_regular_chat_omits_tools(_augment, _mem_task):
    _augment.side_effect = lambda convo, *args, **kwargs: convo
    user = await database_sync_to_async(User.objects.create_user)(username="appendtools", password="pass")
    db_convo = await WSConversation.objects.acreate(owner=user, system_prompt="sys")

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
    assert consumer.convo[-1].tools == []
    assert consumer.convo[-1].tool_choice is None


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
@patch("apps.chat.consumers.chat.enqueue_conversation_memories_task")
@patch("apps.chat.consumers.chat_receive.augment_conversation_with_memory_async")
async def test_append_persists_selected_collections(_augment, _mem_task):
    _augment.side_effect = lambda convo, *args, **kwargs: convo
    user = await database_sync_to_async(User.objects.create_user)(username="appendcollections", password="pass")
    db_convo = await WSConversation.objects.acreate(owner=user, system_prompt="sys")

    consumer = ChatConsumer()
    consumer.base_send = AsyncMock()
    consumer.scope = {"user": user, "url_route": {"kwargs": {"convo_id": db_convo.id}}}
    consumer.user = user
    consumer.db_convo = db_convo
    consumer.convo = Conversation(system="sys", messages=[])
    consumer.dead = False
    consumer.col_ref = CollectionsRef([])
    consumer.doc_tools = [_test_document_ids]
    consumer.tools = [_test_document_ids]
    consumer.last_sent_sequence = -1
    consumer.llm_if = AsyncMock()
    consumer.llm_if.spin = AsyncMock()

    payload = json.dumps(
        {
            "action": "append",
            "message": {"role": "user", "content": "Search these papers"},
            "collections": [3, "7"],
        }
    )

    await consumer.receive(payload)
    await db_convo.arefresh_from_db()

    assert consumer.col_ref.collections == [3, "7"]
    assert db_convo.selected_collection_ids == [3, "7"]


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
async def test_selection_update_persists_without_appending_message():
    user = await database_sync_to_async(User.objects.create_user)(username="selectcollections", password="pass")
    db_convo = await WSConversation.objects.acreate(owner=user, system_prompt="sys")

    consumer = ChatConsumer()
    consumer.base_send = AsyncMock()
    consumer.scope = {"user": user, "url_route": {"kwargs": {"convo_id": db_convo.id}}}
    consumer.user = user
    consumer.db_convo = db_convo
    consumer.convo = Conversation(system="sys", messages=[])
    consumer.dead = False
    consumer.col_ref = CollectionsRef([])

    payload = json.dumps(
        {
            "action": "select_collections",
            "collections": [11, "13"],
        }
    )

    await consumer.receive(payload)
    await db_convo.arefresh_from_db()

    assert consumer.col_ref.collections == [11, "13"]
    assert db_convo.selected_collection_ids == [11, "13"]
    assert consumer.convo.messages == []


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
async def test_selection_acknowledges_atomic_skill_override_save():
    user = await database_sync_to_async(User.objects.create_user)(username="selectskills", password="pass")
    collection = await Collection.objects.acreate(name="Research")
    await CollectionPermission.objects.acreate(user=user, collection=collection, permission="VIEW")
    doc = RawTextDocument(
        title="research_skill.md", full_text="Use research conventions.",
        full_text_hash=RawTextDocument.hash_fn("Use research conventions."),
        collection=collection, ingested_by=user,
    )
    await database_sync_to_async(doc.save)(dont_rechunk=True)
    db_convo = await WSConversation.objects.acreate(owner=user, system_prompt="sys")
    consumer = ChatConsumer()
    consumer.base_send = AsyncMock()
    consumer.user = user
    consumer.db_convo = db_convo
    consumer.convo = Conversation(system="sys", messages=[])
    consumer.dead = False
    consumer.col_ref = CollectionsRef([])
    skill_id = f"{doc._meta.label_lower}:{doc.pk}"

    with patch("django.conf.settings.SKILLS_ENABLED", True), patch(
        "django.conf.settings.AQUILLM_COLLECTION_MARKDOWN_SKILLS_ENABLED", True, create=True
    ):
        await consumer.receive(json.dumps({
            "action": "select_collections", "collections": [collection.id],
            "skill_overrides": {skill_id: False},
            "request_id": "apply-1",
        }))
    await db_convo.arefresh_from_db()

    assert db_convo.selected_collection_ids == [collection.id]
    assert db_convo.skill_overrides == {skill_id: False}
    assert consumer.skill_overrides == {skill_id: False}
    sent = [json.loads(call.args[0]["text"]) for call in consumer.base_send.await_args_list]
    assert {"context_selection": {
        "selected_collections": [collection.id], "skill_overrides": {skill_id: False},
        "request_id": "apply-1",
    }} in sent


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize("invalid_overrides", [
    {"apps_documents.rawtextdocument:999999": "false"},
    {"apps_documents.rawtextdocument:999999": True},
])
async def test_invalid_skill_overrides_preserve_saved_and_consumer_selection(invalid_overrides):
    user = await database_sync_to_async(User.objects.create_user)(username="invalidskills", password="pass")
    db_convo = await WSConversation.objects.acreate(
        owner=user, system_prompt="sys", selected_collection_ids=[3], skill_overrides={},
    )
    consumer = ChatConsumer()
    consumer.base_send = AsyncMock()
    consumer.user = user
    consumer.db_convo = db_convo
    consumer.convo = Conversation(system="sys", messages=[])
    consumer.dead = False
    consumer.col_ref = CollectionsRef([3])
    consumer.skill_overrides = {}

    await consumer.receive(json.dumps({
        "action": "select_collections", "collections": [8],
        "skill_overrides": invalid_overrides, "request_id": "apply-invalid",
    }))
    await db_convo.arefresh_from_db()

    assert db_convo.selected_collection_ids == [3]
    assert db_convo.skill_overrides == {}
    assert consumer.col_ref.collections == [3]
    assert consumer.skill_overrides == {}
    sent = [json.loads(call.args[0]["text"]) for call in consumer.base_send.await_args_list]
    assert any(item.get("context_selection_error", {}).get("request_id") == "apply-invalid" for item in sent)
    assert not any("context_selection" in item for item in sent)


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
async def test_legacy_selection_preserves_overrides_and_snapshot_hydrates_them():
    user = await database_sync_to_async(User.objects.create_user)(username="legacyskills", password="pass")
    saved_overrides = {"apps_documents.rawtextdocument:321": False}
    db_convo = await WSConversation.objects.acreate(
        owner=user, system_prompt="sys", skill_overrides=saved_overrides,
    )
    consumer = ChatConsumer()
    consumer.base_send = AsyncMock()
    consumer.user = user
    consumer.db_convo = db_convo
    consumer.convo = Conversation(system="sys", messages=[])
    consumer.dead = False
    consumer.col_ref = CollectionsRef([])
    consumer._apply_saved_collection_selection()

    await consumer.receive(json.dumps({"action": "select_collections", "collections": [9]}))
    await send_conversation_snapshot(consumer)
    await db_convo.arefresh_from_db()

    assert db_convo.selected_collection_ids == [9]
    assert db_convo.skill_overrides == saved_overrides
    assert consumer.skill_overrides == saved_overrides
    sent = [json.loads(call.args[0]["text"]) for call in consumer.base_send.await_args_list]
    assert sent[-1]["conversation"]["skill_overrides"] == saved_overrides


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
@override_settings(
    SKILLS_ENABLED=True,
    AQUILLM_COLLECTION_MARKDOWN_SKILLS_ENABLED=True,
    AQUILLM_SKILLS_EXTRA_MODULES=[],
    AQUILLM_SKILLS_MARKDOWN_DIR="",
)
async def test_append_applies_independent_skill_before_prompt_construction():
    user = await database_sync_to_async(User.objects.create_user)(username="appendskill", password="pass")
    collection = await Collection.objects.acreate(name="Research")
    await CollectionPermission.objects.acreate(user=user, collection=collection, permission="VIEW")
    doc = RawTextDocument(
        title="research_skill.md", full_text="Independent append instruction",
        full_text_hash=RawTextDocument.hash_fn("Independent append instruction"),
        collection=collection, ingested_by=user,
    )
    await database_sync_to_async(doc.save)(dont_rechunk=True)
    db_convo = await WSConversation.objects.acreate(owner=user, system_prompt="sys")
    consumer = ChatConsumer()
    consumer.base_send = AsyncMock()
    consumer.user = user
    consumer.db_convo = db_convo
    consumer.convo = Conversation(system="sys", messages=[])
    consumer.dead = False
    consumer.col_ref = CollectionsRef([])
    consumer.doc_tools = []
    consumer.tools = []
    consumer.memory_tools = []
    consumer.last_sent_sequence = -1
    consumer._save_conversation = AsyncMock()
    observed_systems = []

    async def capture_system(convo, user_arg, system, convo_id, **kwargs):
        observed_systems.append(system)
        return convo

    with patch("apps.chat.consumers.chat_receive.augment_conversation_with_memory_async", side_effect=capture_system), patch(
        "apps.chat.consumers.chat_receive.run_direct_rag_turn", new=AsyncMock(return_value="skipped")
    ), patch("apps.chat.consumers.chat_receive.run_llm_spin", new=AsyncMock()):
        await consumer.receive(json.dumps({
            "action": "append", "collections": [],
            "skill_overrides": {f"{doc._meta.label_lower}:{doc.pk}": True},
            "message": {"role": "user", "content": "hello"},
        }))
    await db_convo.arefresh_from_db()

    assert db_convo.skill_overrides == {f"{doc._meta.label_lower}:{doc.pk}": True}
    assert len(observed_systems) == 1
    assert "Independent append instruction" in observed_systems[0]


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
@patch("apps.chat.consumers.chat.enqueue_conversation_memories_task")
@patch("apps.chat.consumers.chat_receive.augment_conversation_with_memory_async")
async def test_append_explicit_document_search_requires_doc_tool(_augment, _mem_task):
    _augment.side_effect = lambda convo, *args, **kwargs: convo
    user = await database_sync_to_async(User.objects.create_user)(username="appenddocsearch", password="pass")
    db_convo = await WSConversation.objects.acreate(owner=user, system_prompt="sys")

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
            "message": {
                "role": "user",
                "content": "Search the selected documents for calibration notes.",
            },
            "collections": [],
        }
    )

    await consumer.receive(payload)

    assert consumer.convo is not None
    assert consumer.convo[-1].tools == [_test_document_ids]
    assert consumer.convo[-1].tool_choice.type == "any"
