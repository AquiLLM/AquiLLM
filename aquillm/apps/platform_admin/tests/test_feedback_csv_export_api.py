"""HTTP tests for feedback ratings CSV export."""
import csv
import io
from uuid import uuid4

import pytest
from django.contrib.auth import get_user_model
from django.test import Client

from apps.chat.models import Message, WSConversation

User = get_user_model()


@pytest.fixture
def superuser(db):
    return User.objects.create_superuser(username="su", email="su@example.com", password="pw")


@pytest.fixture
def staff_user(db):
    u = User.objects.create_user(username="st", password="pw")
    u.is_staff = True
    u.save()
    return u


@pytest.fixture
def conversation(db, superuser):
    return WSConversation.objects.create(owner=superuser)


@pytest.mark.django_db
def test_csv_requires_superuser(staff_user):
    client = Client()
    client.force_login(staff_user)
    r = client.get("/api/feedback/ratings.csv")
    assert r.status_code == 403


@pytest.mark.django_db
def test_csv_streams_rows_and_escaping(superuser, conversation):
    Message.objects.create(
        conversation=conversation,
        message_uuid=uuid4(),
        role="user",
        content="q",
        sequence_number=0,
    )
    Message.objects.create(
        conversation=conversation,
        message_uuid=uuid4(),
        role="assistant",
        content="a",
        sequence_number=1,
        stop_reason="end_turn",
        rating=4,
        feedback_text='He said "hi", then left\nsecond line',
    )

    client = Client()
    client.force_login(superuser)
    r = client.get("/api/feedback/ratings.csv")
    assert r.status_code == 200
    assert r["Content-Type"].startswith("text/csv")
    assert "attachment" in r["Content-Disposition"]
    assert "feedback_ratings_" in r["Content-Disposition"]

    rows = list(csv.reader(io.StringIO(r.content.decode("utf-8"))))
    assert rows[0] == ["date", "user_number", "rating", "question_number", "comments"]
    assert len(rows) == 2
    assert rows[1][1] == str(superuser.id)
    assert rows[1][2] == "4"
    assert rows[1][3] == "1"
    assert "second line" in rows[1][4]
