"""Tests for the feedback dashboard export API."""

from django.contrib.auth import get_user_model
from django.test import Client, TestCase
from django.urls import reverse

from apps.chat.models import Message, WSConversation

User = get_user_model()


def make_superuser(username: str = "exportadmin") -> User:
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


def make_conversation(owner: User, name: str = "export convo") -> WSConversation:
    return WSConversation.objects.create(owner=owner, name=name)


def make_message(conversation: WSConversation, seq: int = 1, **kwargs) -> Message:
    defaults = dict(
        role="assistant",
        content="test content",
        sequence_number=seq,
        rating=4,
        feedback_text="good response",
        model="claude-3-5-sonnet",
        tool_call_name=None,
    )
    defaults.update(kwargs)
    return Message.objects.create(conversation=conversation, **defaults)


def read_streaming_response(response) -> str:
    return b"".join(response.streaming_content).decode("utf-8")


class FeedbackExportEndpointTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.superuser = make_superuser()
        self.owner = make_user("exportowner")
        self.convo = make_conversation(self.owner, name="export test convo")

        make_message(
            self.convo,
            seq=1,
            rating=5,
            feedback_text="export test feedback",
            model="claude-3-5-sonnet",
        )

        self.url = reverse("api_feedback_dashboard_export")

    def test_returns_csv_for_superuser(self):
        self.client.login(username="exportadmin", password="testpass123")
        resp = self.client.get(self.url)

        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/csv", resp.get("Content-Type", ""))

    def test_has_content_disposition_attachment(self):
        self.client.login(username="exportadmin", password="testpass123")
        resp = self.client.get(self.url)

        self.assertIn("attachment", resp.get("Content-Disposition", ""))
        self.assertIn(".csv", resp.get("Content-Disposition", ""))

    def test_csv_has_header_row(self):
        self.client.login(username="exportadmin", password="testpass123")
        resp = self.client.get(self.url)
        content = read_streaming_response(resp)
        first_line = content.split("\n")[0]

        self.assertIn("id", first_line)
        self.assertIn("username", first_line)
        self.assertIn("rating", first_line)
        self.assertIn("feedback_text", first_line)
        self.assertIn("effective_date", first_line)

    def test_csv_contains_data_row(self):
        self.client.login(username="exportadmin", password="testpass123")
        resp = self.client.get(self.url)
        content = read_streaming_response(resp)

        self.assertIn("export test feedback", content)
        self.assertIn("exportowner", content)
        self.assertIn("export test convo", content)

    def test_filter_applied_to_export(self):
        make_message(
            self.convo,
            seq=2,
            rating=1,
            feedback_text="low rating feedback",
        )

        self.client.login(username="exportadmin", password="testpass123")
        resp = self.client.get(self.url, {"exact_rating": 5})
        content = read_streaming_response(resp)

        self.assertIn("export test feedback", content)
        self.assertNotIn("low rating feedback", content)

    def test_feedback_text_search_filter_applied_to_export(self):
        make_message(
            self.convo,
            seq=2,
            rating=4,
            feedback_text="different export text",
        )

        self.client.login(username="exportadmin", password="testpass123")
        resp = self.client.get(
            self.url,
            {"feedback_text_search": "export test feedback"},
        )
        content = read_streaming_response(resp)

        self.assertIn("export test feedback", content)
        self.assertNotIn("different export text", content)

    def test_regular_user_is_forbidden(self):
        regular = make_user("regularexport")
        self.client.login(username=regular.username, password="testpass123")

        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 403)
