"""Tests for feedback/rating persistence and validation."""
from uuid import uuid4

import pytest
from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.utils import timezone

from apps.chat.models import Message, WSConversation
from apps.chat.services.feedback import (
    FEEDBACK_TEXT_MAX_LEN,
    apply_message_feedback_text,
    apply_message_rating,
)

User = get_user_model()


@pytest.fixture
def user(db):
    return User.objects.create_user(username="fbuser", password="x")


@pytest.fixture
def conversation(db, user):
    return WSConversation.objects.create(owner=user)


@pytest.mark.django_db
def test_apply_rating_sets_timestamp(conversation):
    uid = uuid4()
    Message.objects.create(
        conversation=conversation,
        message_uuid=uid,
        role="assistant",
        content="hi",
        sequence_number=1,
        stop_reason="end_turn",
    )
    apply_message_rating(conversation.id, str(uid), 4)
    m = Message.objects.get(message_uuid=uid)
    assert m.rating == 4
    assert m.feedback_submitted_at is not None
    assert timezone.now() - m.feedback_submitted_at < timezone.timedelta(seconds=30)


@pytest.mark.django_db
def test_apply_feedback_sets_timestamp(conversation):
    uid = uuid4()
    Message.objects.create(
        conversation=conversation,
        message_uuid=uid,
        role="assistant",
        content="hi",
        sequence_number=1,
        stop_reason="end_turn",
    )
    apply_message_feedback_text(conversation.id, str(uid), "  great  ")
    m = Message.objects.get(message_uuid=uid)
    assert m.feedback_text == "great"
    assert m.feedback_submitted_at is not None


@pytest.mark.django_db
def test_rating_out_of_range_raises(conversation):
    uid = uuid4()
    Message.objects.create(
        conversation=conversation,
        message_uuid=uid,
        role="assistant",
        content="hi",
        sequence_number=1,
        stop_reason="end_turn",
    )
    with pytest.raises(ValidationError):
        apply_message_rating(conversation.id, str(uid), 6)


@pytest.mark.django_db
def test_feedback_truncated(conversation):
    uid = uuid4()
    Message.objects.create(
        conversation=conversation,
        message_uuid=uid,
        role="assistant",
        content="hi",
        sequence_number=1,
        stop_reason="end_turn",
    )
    long_text = "x" * (FEEDBACK_TEXT_MAX_LEN + 50)
    apply_message_feedback_text(conversation.id, str(uid), long_text)
    m = Message.objects.get(message_uuid=uid)
    assert len(m.feedback_text) == FEEDBACK_TEXT_MAX_LEN


@pytest.mark.django_db
def test_rating_only_updates_assistant(conversation):
    uid = uuid4()
    Message.objects.create(
        conversation=conversation,
        message_uuid=uid,
        role="user",
        content="q",
        sequence_number=0,
    )
    apply_message_rating(conversation.id, str(uid), 5)
    m = Message.objects.get(message_uuid=uid)
    assert m.rating is None
    assert m.feedback_submitted_at is None
