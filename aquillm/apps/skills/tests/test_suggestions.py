"""Tests for the feedback → suggestion flow (Phase 2).

Covers: pending-feedback query filtering, generation service (LLM mocked),
accept (with and without owner override), dismiss, status transitions,
and API permission gates.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from django.contrib.auth import get_user_model
from django.test import Client, override_settings

from apps.chat.models.conversation import WSConversation
from apps.chat.models.message import Message
from apps.collections.models import Collection, CollectionPermission
from apps.skills.models import CollectionSkill, FeedbackDismissal, SkillEditSuggestion
from apps.skills.services.suggestions import (
    _list_pending_feedback_sync,
    accept_suggestion_sync,
    dismiss_feedback_sync,
    dismiss_suggestion_sync,
)

User = get_user_model()


# ---- fixtures --------------------------------------------------------------


@pytest.fixture
def owner(db):
    return User.objects.create_user(username="owner", password="pw")


@pytest.fixture
def editor(db):
    return User.objects.create_user(username="editor", password="pw")


@pytest.fixture
def stranger(db):
    return User.objects.create_user(username="stranger", password="pw")


@pytest.fixture
def collection(owner, editor):
    c = Collection.objects.create(name="papers")
    CollectionPermission.objects.create(user=owner, collection=c, permission="MANAGE")
    CollectionPermission.objects.create(user=editor, collection=c, permission="EDIT")
    return c


@pytest.fixture
def convo_with_collection(owner, collection):
    return WSConversation.objects.create(
        owner=owner,
        name="test convo",
        system_prompt="sys",
        selected_collection_ids=[collection.id],
    )


def _make_message_pair(
    convo, *, user_text: str, assistant_text: str, rating: int | None, feedback_text: str | None,
):
    """Create a user message followed by an assistant message with optional feedback."""
    user_msg = Message.objects.create(
        conversation=convo,
        role="user",
        content=user_text,
        sequence_number=Message.objects.filter(conversation=convo).count(),
    )
    asst_msg = Message.objects.create(
        conversation=convo,
        role="assistant",
        content=assistant_text,
        model="gpt-4o",
        rating=rating,
        feedback_text=feedback_text,
        sequence_number=Message.objects.filter(conversation=convo).count(),
    )
    return user_msg, asst_msg


def _client(user):
    c = Client()
    c.force_login(user)
    return c


# ---- pending-feedback query ------------------------------------------------


@pytest.mark.django_db
@override_settings(SKILLS_ENABLED=True)
def test_pending_feedback_includes_low_rating_with_comment(collection, convo_with_collection):
    _, asst = _make_message_pair(
        convo_with_collection,
        user_text="who wrote this?",
        assistant_text="bad answer",
        rating=1,
        feedback_text="wrong",
    )
    result = _list_pending_feedback_sync(collection.id)
    assert [m.id for m in result] == [asst.id]


@pytest.mark.django_db
@override_settings(SKILLS_ENABLED=True)
def test_pending_feedback_excludes_high_rating(collection, convo_with_collection):
    _make_message_pair(
        convo_with_collection,
        user_text="q", assistant_text="a", rating=5, feedback_text="great!",
    )
    assert _list_pending_feedback_sync(collection.id) == []


@pytest.mark.django_db
@override_settings(SKILLS_ENABLED=True)
def test_pending_feedback_excludes_empty_comment(collection, convo_with_collection):
    _make_message_pair(
        convo_with_collection,
        user_text="q", assistant_text="a", rating=1, feedback_text="",
    )
    _make_message_pair(
        convo_with_collection,
        user_text="q", assistant_text="a", rating=2, feedback_text=None,
    )
    assert _list_pending_feedback_sync(collection.id) == []


@pytest.mark.django_db
@override_settings(SKILLS_ENABLED=True)
def test_pending_feedback_excludes_other_collections(owner):
    other = Collection.objects.create(name="other")
    CollectionPermission.objects.create(user=owner, collection=other, permission="MANAGE")
    convo = WSConversation.objects.create(
        owner=owner, name="x", system_prompt="sys", selected_collection_ids=[other.id],
    )
    _make_message_pair(convo, user_text="q", assistant_text="a", rating=1, feedback_text="bad")
    # Different collection — nothing should appear for it.
    target = Collection.objects.create(name="target")
    assert _list_pending_feedback_sync(target.id) == []


@pytest.mark.django_db
@override_settings(SKILLS_ENABLED=True)
def test_pending_feedback_excludes_already_pending_suggestion(
    collection, convo_with_collection, owner,
):
    _, asst = _make_message_pair(
        convo_with_collection, user_text="q", assistant_text="a", rating=1, feedback_text="bad",
    )
    SkillEditSuggestion.objects.create(
        collection=collection,
        source_message=asst,
        notes_body_at_generation="",
        proposed_body="## proposed",
        generated_by=owner,
    )
    assert _list_pending_feedback_sync(collection.id) == []


@pytest.mark.django_db
@override_settings(SKILLS_ENABLED=True)
def test_pending_feedback_includes_after_dismiss(collection, convo_with_collection, owner):
    """Dismissing a suggestion re-opens the feedback for re-drafting."""
    _, asst = _make_message_pair(
        convo_with_collection, user_text="q", assistant_text="a", rating=1, feedback_text="bad",
    )
    s = SkillEditSuggestion.objects.create(
        collection=collection,
        source_message=asst,
        notes_body_at_generation="",
        proposed_body="## proposed",
        generated_by=owner,
        status=SkillEditSuggestion.STATUS_DISMISSED,
    )
    assert s.status == SkillEditSuggestion.STATUS_DISMISSED
    assert [m.id for m in _list_pending_feedback_sync(collection.id)] == [asst.id]


# ---- accept / dismiss -------------------------------------------------------


@pytest.mark.django_db
@override_settings(SKILLS_ENABLED=True)
def test_accept_writes_proposed_body_to_notes(collection, convo_with_collection, owner):
    _, asst = _make_message_pair(
        convo_with_collection, user_text="q", assistant_text="a", rating=1, feedback_text="bad",
    )
    s = SkillEditSuggestion.objects.create(
        collection=collection,
        source_message=asst,
        notes_body_at_generation="",
        proposed_body="## new notes",
        generated_by=owner,
    )
    cs = accept_suggestion_sync(suggestion=s, override_body=None, user=owner)
    assert cs.body == "## new notes"
    assert cs.updated_by_id == owner.id
    s.refresh_from_db()
    assert s.status == SkillEditSuggestion.STATUS_ACCEPTED
    assert s.resolved_by_id == owner.id
    assert s.resolved_at is not None


@pytest.mark.django_db
@override_settings(SKILLS_ENABLED=True)
def test_accept_with_override_body_wins(collection, convo_with_collection, owner):
    _, asst = _make_message_pair(
        convo_with_collection, user_text="q", assistant_text="a", rating=1, feedback_text="bad",
    )
    s = SkillEditSuggestion.objects.create(
        collection=collection,
        source_message=asst,
        notes_body_at_generation="",
        proposed_body="## original proposal",
        generated_by=owner,
    )
    cs = accept_suggestion_sync(suggestion=s, override_body="## owner edited", user=owner)
    assert cs.body == "## owner edited"


@pytest.mark.django_db
@override_settings(SKILLS_ENABLED=True)
def test_accept_already_resolved_raises(collection, convo_with_collection, owner):
    _, asst = _make_message_pair(
        convo_with_collection, user_text="q", assistant_text="a", rating=1, feedback_text="bad",
    )
    s = SkillEditSuggestion.objects.create(
        collection=collection,
        source_message=asst,
        notes_body_at_generation="",
        proposed_body="## p",
        generated_by=owner,
        status=SkillEditSuggestion.STATUS_DISMISSED,
    )
    with pytest.raises(ValueError):
        accept_suggestion_sync(suggestion=s, override_body=None, user=owner)


@pytest.mark.django_db
@override_settings(SKILLS_ENABLED=True)
def test_dismiss_marks_status_and_resolver(collection, convo_with_collection, owner):
    _, asst = _make_message_pair(
        convo_with_collection, user_text="q", assistant_text="a", rating=1, feedback_text="bad",
    )
    s = SkillEditSuggestion.objects.create(
        collection=collection,
        source_message=asst,
        notes_body_at_generation="",
        proposed_body="## p",
        generated_by=owner,
    )
    dismiss_suggestion_sync(suggestion=s, user=owner)
    s.refresh_from_db()
    assert s.status == SkillEditSuggestion.STATUS_DISMISSED
    assert s.resolved_by_id == owner.id


# ---- generation service (LLM mocked) ---------------------------------------


@pytest.mark.django_db(transaction=True)
@override_settings(SKILLS_ENABLED=True)
async def test_generate_calls_llm_and_persists(collection, convo_with_collection, owner):
    from apps.skills.services import suggestions

    user_msg, asst = await Message.objects.acreate(
        conversation=convo_with_collection, role="user", content="who wrote this?", sequence_number=0,
    ), None
    asst = await Message.objects.acreate(
        conversation=convo_with_collection,
        role="assistant",
        content="written by FakeCo",
        model="gpt-4o",
        rating=1,
        feedback_text="Wrong — written by RealCo",
        sequence_number=1,
    )

    class FakeResponse:
        text = "## Authors\nThe authors are RealCo."

    class FakeLLM:
        async def get_message(self, *, system, messages, messages_pydantic, max_tokens):
            return FakeResponse()

    with patch.object(suggestions, "apps") as fake_django_apps:
        fake_django_apps.get_app_config.return_value.llm_interface = FakeLLM()
        s = await suggestions.generate_suggestion(
            collection_id=collection.id, message_id=asst.id, user=owner,
        )
    assert s.status == SkillEditSuggestion.STATUS_PENDING
    assert "RealCo" in s.proposed_body
    assert s.notes_body_at_generation == ""
    assert s.generated_by_id == owner.id


@pytest.mark.django_db(transaction=True)
@override_settings(SKILLS_ENABLED=True)
async def test_generate_rejects_message_without_collection(owner, collection):
    """Message in a conversation that didn't have this collection selected → ValueError."""
    from apps.skills.services import suggestions

    other_convo = await WSConversation.objects.acreate(
        owner=owner, name="x", system_prompt="sys", selected_collection_ids=[],
    )
    asst = await Message.objects.acreate(
        conversation=other_convo,
        role="assistant",
        content="a", model="gpt-4o", rating=1, feedback_text="bad",
        sequence_number=0,
    )
    with pytest.raises(ValueError):
        await suggestions.generate_suggestion(
            collection_id=collection.id, message_id=asst.id, user=owner,
        )


# ---- API endpoint permissions ----------------------------------------------


@pytest.mark.django_db
@override_settings(SKILLS_ENABLED=True)
def test_pending_feedback_api_requires_manage(collection, convo_with_collection, editor, stranger):
    _make_message_pair(
        convo_with_collection, user_text="q", assistant_text="a", rating=1, feedback_text="bad",
    )
    # EDIT-only user is forbidden.
    assert _client(editor).get(
        f"/api/collections/{collection.id}/pending-feedback/"
    ).status_code == 403
    # Stranger is forbidden.
    assert _client(stranger).get(
        f"/api/collections/{collection.id}/pending-feedback/"
    ).status_code == 403


@pytest.mark.django_db
@override_settings(SKILLS_ENABLED=True)
def test_pending_feedback_api_returns_items_for_manager(
    collection, convo_with_collection, owner,
):
    _, asst = _make_message_pair(
        convo_with_collection, user_text="q", assistant_text="a", rating=1, feedback_text="bad",
    )
    r = _client(owner).get(f"/api/collections/{collection.id}/pending-feedback/")
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 1
    assert items[0]["message_id"] == asst.id
    assert items[0]["feedback_text"] == "bad"


@pytest.mark.django_db
@override_settings(SKILLS_ENABLED=True)
def test_accept_api_requires_manage(collection, convo_with_collection, owner, editor):
    _, asst = _make_message_pair(
        convo_with_collection, user_text="q", assistant_text="a", rating=1, feedback_text="bad",
    )
    s = SkillEditSuggestion.objects.create(
        collection=collection,
        source_message=asst,
        notes_body_at_generation="",
        proposed_body="## p",
        generated_by=owner,
    )
    # Editor (no MANAGE) → 403
    r = _client(editor).post(
        f"/api/suggestions/{s.id}/accept/",
        data=json.dumps({}),
        content_type="application/json",
    )
    assert r.status_code == 403
    # Manager → 200
    r = _client(owner).post(
        f"/api/suggestions/{s.id}/accept/",
        data=json.dumps({}),
        content_type="application/json",
    )
    assert r.status_code == 200
    assert CollectionSkill.objects.get(collection=collection).body == "## p"


@pytest.mark.django_db
@override_settings(SKILLS_ENABLED=True)
def test_accept_api_rejects_oversized_override(collection, convo_with_collection, owner):
    _, asst = _make_message_pair(
        convo_with_collection, user_text="q", assistant_text="a", rating=1, feedback_text="bad",
    )
    s = SkillEditSuggestion.objects.create(
        collection=collection,
        source_message=asst,
        notes_body_at_generation="",
        proposed_body="## p",
        generated_by=owner,
    )
    r = _client(owner).post(
        f"/api/suggestions/{s.id}/accept/",
        data=json.dumps({"body": "x" * 17_000}),
        content_type="application/json",
    )
    assert r.status_code == 400


@pytest.mark.django_db
@override_settings(SKILLS_ENABLED=True)
def test_dismiss_api_requires_manage(collection, convo_with_collection, owner, editor):
    _, asst = _make_message_pair(
        convo_with_collection, user_text="q", assistant_text="a", rating=1, feedback_text="bad",
    )
    s = SkillEditSuggestion.objects.create(
        collection=collection,
        source_message=asst,
        notes_body_at_generation="",
        proposed_body="## p",
        generated_by=owner,
    )
    r = _client(editor).post(f"/api/suggestions/{s.id}/dismiss/")
    assert r.status_code == 403
    r = _client(owner).post(f"/api/suggestions/{s.id}/dismiss/")
    assert r.status_code == 200
    s.refresh_from_db()
    assert s.status == SkillEditSuggestion.STATUS_DISMISSED


@pytest.mark.django_db
@override_settings(SKILLS_ENABLED=True)
def test_suggestions_list_api(collection, convo_with_collection, owner):
    _, asst = _make_message_pair(
        convo_with_collection, user_text="q", assistant_text="a", rating=1, feedback_text="bad",
    )
    SkillEditSuggestion.objects.create(
        collection=collection,
        source_message=asst,
        notes_body_at_generation="",
        proposed_body="## p",
        generated_by=owner,
    )
    r = _client(owner).get(f"/api/collections/{collection.id}/suggestions/")
    assert r.status_code == 200
    assert len(r.json()["items"]) == 1


# ---- feedback-level dismissal ----------------------------------------------


@pytest.mark.django_db
@override_settings(SKILLS_ENABLED=True)
def test_dismiss_feedback_hides_from_pending(collection, convo_with_collection, owner):
    _, asst = _make_message_pair(
        convo_with_collection, user_text="q", assistant_text="a", rating=1, feedback_text="bad",
    )
    assert [m.id for m in _list_pending_feedback_sync(collection.id)] == [asst.id]
    dismiss_feedback_sync(collection=collection, source_message=asst, user=owner)
    assert _list_pending_feedback_sync(collection.id) == []


@pytest.mark.django_db
@override_settings(SKILLS_ENABLED=True)
def test_dismiss_feedback_is_idempotent(collection, convo_with_collection, owner):
    _, asst = _make_message_pair(
        convo_with_collection, user_text="q", assistant_text="a", rating=1, feedback_text="bad",
    )
    dismiss_feedback_sync(collection=collection, source_message=asst, user=owner)
    dismiss_feedback_sync(collection=collection, source_message=asst, user=owner)
    assert FeedbackDismissal.objects.filter(
        collection=collection, source_message=asst,
    ).count() == 1


@pytest.mark.django_db
@override_settings(SKILLS_ENABLED=True)
def test_dismiss_feedback_api_requires_manage(collection, convo_with_collection, owner, editor):
    _, asst = _make_message_pair(
        convo_with_collection, user_text="q", assistant_text="a", rating=1, feedback_text="bad",
    )
    # Editor (no MANAGE) → 403
    r = _client(editor).post(
        f"/api/collections/{collection.id}/pending-feedback/{asst.id}/dismiss/",
    )
    assert r.status_code == 403
    assert not FeedbackDismissal.objects.filter(
        collection=collection, source_message=asst,
    ).exists()

    # Manager → 200
    r = _client(owner).post(
        f"/api/collections/{collection.id}/pending-feedback/{asst.id}/dismiss/",
    )
    assert r.status_code == 200
    assert FeedbackDismissal.objects.filter(
        collection=collection, source_message=asst,
    ).exists()


@pytest.mark.django_db
@override_settings(SKILLS_ENABLED=True)
def test_dismiss_feedback_api_rejects_cross_collection(
    collection, convo_with_collection, owner,
):
    """Dismissing requires the message's convo to have this collection selected."""
    other = Collection.objects.create(name="other")
    CollectionPermission.objects.create(user=owner, collection=other, permission="MANAGE")
    _, asst = _make_message_pair(
        convo_with_collection, user_text="q", assistant_text="a", rating=1, feedback_text="bad",
    )
    r = _client(owner).post(
        f"/api/collections/{other.id}/pending-feedback/{asst.id}/dismiss/",
    )
    assert r.status_code == 400


@pytest.mark.django_db
@override_settings(SKILLS_ENABLED=True)
def test_dismissed_feedback_excluded_even_after_suggestion_dismissed(
    collection, convo_with_collection, owner,
):
    """Once feedback is dismissed at the row level, dismissing or creating
    suggestions doesn't bring it back."""
    _, asst = _make_message_pair(
        convo_with_collection, user_text="q", assistant_text="a", rating=1, feedback_text="bad",
    )
    SkillEditSuggestion.objects.create(
        collection=collection,
        source_message=asst,
        notes_body_at_generation="",
        proposed_body="## p",
        generated_by=owner,
        status=SkillEditSuggestion.STATUS_DISMISSED,
    )
    # Without FeedbackDismissal, the row would re-appear (existing test asserts this).
    # Add the row-level dismissal and confirm it stays hidden.
    dismiss_feedback_sync(collection=collection, source_message=asst, user=owner)
    assert _list_pending_feedback_sync(collection.id) == []
