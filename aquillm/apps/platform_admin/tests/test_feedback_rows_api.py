"""Tests for the feedback dashboard rows API."""

import json

from django.contrib.auth import get_user_model
from django.test import Client, TestCase
from django.urls import reverse

from apps.chat.models import Message, WSConversation

User = get_user_model()


def make_superuser(username: str = "rowsadmin") -> User:
    return User.objects.create_superuser(
        username=username,
        email=f"{username}@test.com",
        password="testpass123",
    )


def make_user(username: str) -> User:
    return User.objects.create_user(
        username=username,
        email=f"{username}@test.com",
        password="testpass123",
    )


def make_conversation(owner: User, name: str = "test convo") -> WSConversation:
    return WSConversation.objects.create(owner=owner, name=name)


def make_message(conversation: WSConversation, seq: int = 1, **kwargs) -> Message:
    defaults = dict(
        role="assistant",
        content="test content",
        sequence_number=seq,
        rating=4,
        feedback_text="good response",
    )
    defaults.update(kwargs)
    return Message.objects.create(conversation=conversation, **defaults)


class FeedbackRowsEndpointTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.superuser = make_superuser()
        self.owner = make_user("rowsowner")
        self.convo = make_conversation(self.owner, name="rows test convo")

        for i in range(5):
            make_message(
                self.convo,
                seq=i + 1,
                rating=i + 1,
                feedback_text=f"feedback {i}",
            )

        self.url = reverse("api_feedback_dashboard_rows")

    def test_returns_200_for_superuser(self):
        self.client.login(username="rowsadmin", password="testpass123")
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)

    def test_response_has_required_keys(self):
        self.client.login(username="rowsadmin", password="testpass123")
        data = json.loads(self.client.get(self.url).content)

        for key in (
            "rows",
            "page",
            "page_size",
            "total_count",
            "total_pages",
            "prql",
        ):
            self.assertIn(key, data, msg=f"missing key: {key}")

    def test_total_count_matches_created_feedback_rows(self):
        self.client.login(username="rowsadmin", password="testpass123")
        data = json.loads(self.client.get(self.url).content)
        self.assertEqual(data["total_count"], 5)

    def test_rows_contain_expected_fields(self):
        self.client.login(username="rowsadmin", password="testpass123")
        data = json.loads(self.client.get(self.url).content)
        self.assertGreater(len(data["rows"]), 0)

        row = data["rows"][0]
        for field in (
            "id",
            "message_uuid",
            "conversation_id",
            "conversation_name",
            "user_id",
            "username",
            "rating",
            "feedback_text",
            "feedback_submitted_at",
            "created_at",
            "effective_date",
            "role",
            "content_snippet",
            "model",
            "tool_call_name",
            "usage",
            "has_feedback_text",
        ):
            self.assertIn(field, row, msg=f"missing field in row: {field}")

    def test_username_in_row_matches_owner(self):
        self.client.login(username="rowsadmin", password="testpass123")
        data = json.loads(self.client.get(self.url).content)

        for row in data["rows"]:
            self.assertEqual(row["username"], "rowsowner")

    def test_conversation_name_in_row_matches_conversation(self):
        self.client.login(username="rowsadmin", password="testpass123")
        data = json.loads(self.client.get(self.url).content)

        for row in data["rows"]:
            self.assertEqual(row["conversation_name"], "rows test convo")

    def test_pagination_page_1_default(self):
        self.client.login(username="rowsadmin", password="testpass123")
        data = json.loads(self.client.get(self.url).content)
        self.assertEqual(data["page"], 1)

    def test_pagination_page_size_respected(self):
        self.client.login(username="rowsadmin", password="testpass123")
        data = json.loads(self.client.get(self.url, {"page_size": 2}).content)

        self.assertEqual(len(data["rows"]), 2)
        self.assertEqual(data["page_size"], 2)
        self.assertEqual(data["total_pages"], 3)

    def test_page_size_is_capped_at_200(self):
        self.client.login(username="rowsadmin", password="testpass123")
        data = json.loads(self.client.get(self.url, {"page_size": 999}).content)
        self.assertEqual(data["page_size"], 200)

    def test_filter_by_exact_rating_narrows_rows(self):
        self.client.login(username="rowsadmin", password="testpass123")
        data = json.loads(self.client.get(self.url, {"exact_rating": 5}).content)

        self.assertEqual(data["total_count"], 1)
        self.assertEqual(data["rows"][0]["rating"], 5)

    def test_filter_by_feedback_text_search_narrows_rows(self):
        self.client.login(username="rowsadmin", password="testpass123")
        data = json.loads(
            self.client.get(self.url, {"feedback_text_search": "feedback 0"}).content
        )

        self.assertEqual(data["total_count"], 1)
        self.assertEqual(data["rows"][0]["feedback_text"], "feedback 0")

    def test_prql_is_display_only_string(self):
        self.client.login(username="rowsadmin", password="testpass123")
        data = json.loads(
            self.client.get(self.url, {"exact_rating": 5}).content
        )

        self.assertIsInstance(data["prql"], str)
        self.assertIn("from feedback", data["prql"])
        self.assertIn("filter rating == 5", data["prql"])

    def test_regular_user_is_forbidden(self):
        regular = make_user("regularrows")
        self.client.login(username=regular.username, password="testpass123")

        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 403)
