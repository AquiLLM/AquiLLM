"""Tests for feedback export queryset and question_number logic."""
from datetime import timedelta
from uuid import uuid4

import pytest
from django.contrib.auth import get_user_model
from django.utils import timezone

from apps.chat.models import Message, WSConversation
from apps.platform_admin.services.feedback_export import (
    feedback_export_queryset,
    parse_query_bounds,
)

User = get_user_model()


@pytest.fixture
def owner(db):
    return User.objects.create_user(username="owner1", password="x")


@pytest.fixture
def other_user(db):
    return User.objects.create_user(username="owner2", password="x")


@pytest.mark.django_db
def test_question_number_counts_prior_user_messages(owner):
    conv = WSConversation.objects.create(owner=owner)
    Message.objects.create(
        conversation=conv,
        message_uuid=uuid4(),
        role="user",
        content="q1",
        sequence_number=0,
    )
    a1 = Message.objects.create(
        conversation=conv,
        message_uuid=uuid4(),
        role="assistant",
        content="a1",
        sequence_number=1,
        stop_reason="end_turn",
        rating=5,
    )
    Message.objects.create(
        conversation=conv,
        message_uuid=uuid4(),
        role="user",
        content="q2",
        sequence_number=2,
    )
    a2 = Message.objects.create(
        conversation=conv,
        message_uuid=uuid4(),
        role="assistant",
        content="a2",
        sequence_number=3,
        stop_reason="end_turn",
        feedback_text="ok",
    )

    rows = {m.id: m.question_number for m in feedback_export_queryset()}
    assert rows[a1.id] == 1
    assert rows[a2.id] == 2


@pytest.mark.django_db
def test_filters_by_user_number_and_min_rating(owner, other_user):
    c1 = WSConversation.objects.create(owner=owner)
    c2 = WSConversation.objects.create(owner=other_user)
    m1 = Message.objects.create(
        conversation=c1,
        message_uuid=uuid4(),
        role="assistant",
        content="x",
        sequence_number=1,
        stop_reason="end_turn",
        rating=3,
    )
    Message.objects.create(
        conversation=c2,
        message_uuid=uuid4(),
        role="assistant",
        content="y",
        sequence_number=1,
        stop_reason="end_turn",
        rating=5,
    )

    qs = feedback_export_queryset(user_number=owner.id)
    assert list(qs.values_list("id", flat=True)) == [m1.id]

    qs2 = feedback_export_queryset(min_rating=4)
    assert qs2.count() == 1


@pytest.mark.django_db
def test_date_filters_use_effective_date(owner):
    conv = WSConversation.objects.create(owner=owner)
    old = timezone.now() - timedelta(days=10)
    new = timezone.now() - timedelta(days=1)
    m = Message.objects.create(
        conversation=conv,
        message_uuid=uuid4(),
        role="assistant",
        content="x",
        sequence_number=1,
        stop_reason="end_turn",
        rating=2,
    )
    Message.objects.filter(pk=m.pk).update(created_at=old, feedback_submitted_at=new)

    m.refresh_from_db()
    start, end = parse_query_bounds(
        (timezone.now() - timedelta(days=5)).date().isoformat(),
        timezone.now().date().isoformat(),
    )
    assert feedback_export_queryset(start_date=start, end_date=end).filter(pk=m.pk).exists()

    start2, end2 = parse_query_bounds(
        (timezone.now() - timedelta(days=20)).date().isoformat(),
        (timezone.now() - timedelta(days=9)).date().isoformat(),
    )
    assert not feedback_export_queryset(start_date=start2, end_date=end2).filter(pk=m.pk).exists()
