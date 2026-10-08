"""Renaming is owner-only metadata, independent of transcript publication."""

import json
from types import SimpleNamespace

import pytest
from django.apps import apps
from django.contrib.auth import get_user_model
from django.test import Client
from django.urls import reverse

from apps.chat.models import WSConversation
from apps.chat.tasks.title import refine_conversation_title
from aquillm.llm import AssistantMessage, Conversation, UserMessage
from aquillm.message_adapters import load_conversation_from_db, save_conversation_to_db


@pytest.fixture
def chat(db, client):
    owner = get_user_model().objects.create_user(username="rename-owner")
    client.force_login(owner)
    row = WSConversation.objects.create(owner=owner, name="Original title")
    save_conversation_to_db(
        Conversation(system="", messages=[UserMessage(content="My question")]), row
    )
    return row


def rename(client, chat, name):
    return client.post(
        reverse("rename_ws_convo", args=[chat.pk]),
        data=json.dumps({"name": name}),
        content_type="application/json",
    )


def test_rename_persists_trimmed_title_without_changing_activity(client, chat):
    before = chat.updated_at
    response = rename(client, chat, "  Project notes  ")
    assert response.status_code == 200
    assert response.json() == {"id": chat.pk, "name": "Project notes"}
    chat.refresh_from_db()
    assert chat.name == "Project notes"
    assert chat.name_is_manual is True
    assert chat.updated_at == before
    assert list(chat.db_messages.values_list("content", flat=True)) == ["My question"]
    page = client.get(reverse("ws_convo", args=[chat.pk]))
    assert page.status_code == 200
    assert b"Project notes" in page.content


@pytest.mark.parametrize("name", ["", " \n\t ", None, 42, [], "x" * 201])
def test_invalid_title_keeps_saved_name(client, chat, name):
    assert rename(client, chat, name).status_code == 400
    chat.refresh_from_db()
    assert chat.name == "Original title"


@pytest.mark.parametrize("body", ["{", "[]", "null", "{}"])
def test_malformed_payload_is_rejected(client, chat, body):
    response = client.post(
        reverse("rename_ws_convo", args=[chat.pk]), data=body, content_type="application/json"
    )
    assert response.status_code == 400
    chat.refresh_from_db()
    assert chat.name == "Original title"


def test_other_user_cannot_rename_chat(client, chat):
    other = get_user_model().objects.create_user(username="rename-other")
    client.force_login(other)
    assert rename(client, chat, "Stolen title").status_code == 404
    chat.refresh_from_db()
    assert chat.name == "Original title"


def test_rename_requires_login_and_post(client, chat):
    assert client.get(reverse("rename_ws_convo", args=[chat.pk])).status_code == 405
    client.logout()
    assert rename(client, chat, "Anonymous title").status_code == 302
    chat.refresh_from_db()
    assert chat.name == "Original title"


def test_rename_requires_csrf(chat):
    protected_client = Client(enforce_csrf_checks=True)
    protected_client.force_login(chat.owner)
    assert rename(protected_client, chat, "Missing token").status_code == 403


def test_stale_transcript_save_preserves_manual_title(client, chat):
    handle = WSConversation.objects.get(pk=chat.pk)
    transcript = load_conversation_from_db(handle)
    assert rename(client, chat, "My chosen title").status_code == 200
    transcript.messages.append(AssistantMessage(content="Answer", stop_reason="end_turn"))
    save_conversation_to_db(transcript, handle)
    chat.refresh_from_db()
    assert chat.name == "My chosen title"
    assert chat.name_is_manual is True
    assert chat.db_messages.count() == 2


@pytest.mark.django_db(transaction=True)
def test_pending_auto_title_cannot_overwrite_manual_title_matching_fallback(client, monkeypatch):
    owner = get_user_model().objects.create_user(username="rename-race")
    client.force_login(owner)
    row = WSConversation.objects.create(owner=owner, name="Original title")
    save_conversation_to_db(
        Conversation(system="", messages=[UserMessage(content="Question")]), row
    )

    async def provider(**kwargs):
        # The user chooses the same text while an older title task is running.
        from asgiref.sync import sync_to_async
        response = await sync_to_async(rename)(client, row, "Original title")
        assert response.status_code == 200
        return SimpleNamespace(text="Generated replacement")

    monkeypatch.setattr(
        apps.get_app_config("aquillm"), "llm_interface",
        SimpleNamespace(base_args={}, get_message=provider),
    )
    refine_conversation_title(row.pk, "Original title")
    row.refresh_from_db()
    assert row.name == "Original title"
    assert row.name_is_manual is True


def test_legacy_title_generation_preserves_manual_title(client, chat, monkeypatch):
    assert rename(client, chat, "My title").status_code == 200

    async def provider(**kwargs):
        return SimpleNamespace(text="Generated replacement")

    monkeypatch.setattr(
        apps.get_app_config("aquillm"), "llm_interface",
        SimpleNamespace(base_args={}, get_message=provider),
    )
    chat.set_name()  # Deliberately use a stale model instance.
    chat.refresh_from_db()
    assert chat.name == "My title"
