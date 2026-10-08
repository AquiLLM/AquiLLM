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
async def test_answer_save_finishes_without_waiting_for_title_provider(monkeypatch):
    user = await get_user_model().objects.acreate(username="nonblocking-title")
    db = await WSConversation.objects.acreate(owner=user, system_prompt="sys")
    consumer = ChatConsumer()
    consumer.db_convo = db
    consumer.convo = Conversation(
        system="sys",
        messages=[
            UserMessage(content="Explain the telescope calibration"),
            AssistantMessage(content="Use reference stars.", stop_reason="end_turn"),
        ],
    )
    from django.apps import apps

    entered, release = asyncio.Event(), asyncio.Event()

    async def provider(**kwargs):
        entered.set()
        await release.wait()
        return SimpleNamespace(text="Generated title")

    monkeypatch.setattr(
        apps.get_app_config("aquillm"),
        "llm_interface",
        SimpleNamespace(base_args={}, get_message=provider),
    )
    saving = asyncio.create_task(consumer._save_conversation())
    try:
        done, _ = await asyncio.wait({saving}, timeout=0.5)
        assert saving in done
        await db.arefresh_from_db()
        assert db.name == "Explain the telescope calibration"
        assert not entered.is_set()
    finally:
        release.set()
        await saving


@pytest.mark.django_db(transaction=True)
def test_title_timeout_keeps_fallback_and_metadata_identity(monkeypatch):
    from apps.chat.tasks.title import refine_conversation_title
    from django.apps import apps

    user = get_user_model().objects.create(username="title-timeout")
    db = WSConversation.objects.create(owner=user, system_prompt="sys", name="Fallback")
    save_conversation_to_db(
        Conversation(system="sys", messages=[UserMessage(content="question")]), db
    )
    before = db.updated_at

    async def provider(**kwargs):
        await asyncio.Event().wait()

    monkeypatch.setenv("CHAT_TITLE_TIMEOUT_SECONDS", "0.1")
    monkeypatch.setattr(
        apps.get_app_config("aquillm"),
        "llm_interface",
        SimpleNamespace(base_args={}, get_message=provider),
    )
    refine_conversation_title(db.pk, "Fallback")
    db.refresh_from_db()
    assert db.name == "Fallback" and db.updated_at == before


@pytest.mark.django_db(transaction=True)
def test_title_refinement_does_not_overwrite_manual_edit(monkeypatch):
    from apps.chat.tasks.title import refine_conversation_title
    from django.apps import apps

    user = get_user_model().objects.create(username="title-manual")
    db = WSConversation.objects.create(owner=user, system_prompt="sys", name="Fallback")
    save_conversation_to_db(
        Conversation(system="sys", messages=[UserMessage(content="question")]), db
    )

    async def provider(**kwargs):
        await WSConversation.objects.filter(pk=db.pk).aupdate(name="Manual title")
        return SimpleNamespace(text="Generated title")

    monkeypatch.setattr(
        apps.get_app_config("aquillm"),
        "llm_interface",
        SimpleNamespace(base_args={}, get_message=provider),
    )
    refine_conversation_title(db.pk, "Fallback")
    db.refresh_from_db()
    assert db.name == "Manual title"
