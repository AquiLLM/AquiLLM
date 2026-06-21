"""
tests for the dashboard api endpoint response shapes and data correctness

covers:
    summary endpoint keys and values
    filters endpoint keys and values
    rows endpoint shape and pagination
    export endpoint content type and csv structure
"""
import json

from django.contrib.auth import get_user_model
from django.test import Client, TestCase
from django.urls import reverse

from apps.chat.models import Message, WSConversation

User = get_user_model()


def make_superuser(username: str = "apiadmin") -> User:
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


class SummaryEndpointTests(TestCase):

    def setUp(self):
        self.client = Client()
        self.superuser = make_superuser()
        self.owner = make_user("sumowner")
        self.convo = make_conversation(self.owner)
        make_message(self.convo, seq=1, rating=5, feedback_text="excellent")
        make_message(self.convo, seq=2, rating=3, feedback_text=None)
        make_message(self.convo, seq=3, rating=None, feedback_text="comment only")
        self.url = reverse("api_feedback_dashboard_summary")

    def test_returns_200(self):
        self.client.login(username="apiadmin", password="testpass123")
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)

    def test_content_type_is_json(self):
        self.client.login(username="apiadmin", password="testpass123")
        resp = self.client.get(self.url)
        self.assertIn("application/json", resp["Content-Type"])

    def test_has_all_required_keys(self):
        self.client.login(username="apiadmin", password="testpass123")
        data = json.loads(self.client.get(self.url).content)
        for key in (
            "total_count", "rated_count", "avg_rating",
            "rating_distribution", "has_text_count", "date_min", "date_max"
        ):
            self.assertIn(key, data, msg=f"missing key: {key}")

    def test_total_count_correct(self):
        self.client.login(username="apiadmin", password="testpass123")
        data = json.loads(self.client.get(self.url).content)
        self.assertEqual(data["total_count"], 3)

    def test_rated_count_correct(self):
        self.client.login(username="apiadmin", password="testpass123")
        data = json.loads(self.client.get(self.url).content)
        self.assertEqual(data["rated_count"], 2)

    def test_rating_distribution_keys_are_strings(self):
        self.client.login(username="apiadmin", password="testpass123")
        data = json.loads(self.client.get(self.url).content)
        for k in data["rating_distribution"]:
            self.assertIsInstance(k, str)

    def test_filter_by_user_narrows_summary(self):
        self.client.login(username="apiadmin", password="testpass123")
        other = make_user("othersumuser")
        other_convo = make_conversation(other)
        make_message(other_convo, seq=1, rating=1, feedback_text="bad")
        data = json.loads(
            self.client.get(self.url, {"user_id": self.owner.id}).content
        )
        self.assertEqual(data["total_count"], 3)


class FiltersEndpointTests(TestCase):

    def setUp(self):
        self.client = Client()
        self.superuser = make_superuser("filtadmin")
        self.owner = make_user("filtowner")
        self.convo = make_conversation(self.owner)
        make_message(
            self.convo, seq=1, rating=4, role="assistant",
            model="claude-3-5-sonnet", tool_call_name=None,
        )
        make_message(
            self.convo, seq=2, rating=2, role="user",
            model="gpt-4", tool_call_name="search_tool",
        )
        self.url = reverse("api_feedback_dashboard_filters")

    def test_returns_200(self):
        self.client.login(username="filtadmin", password="testpass123")
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)

    def test_has_all_required_keys(self):
        self.client.login(username="filtadmin", password="testpass123")
        data = json.loads(self.client.get(self.url).content)
        for key in ("users", "roles", "models", "tool_names", "ratings"):
            self.assertIn(key, data, msg=f"missing key: {key}")

    def test_users_includes_owner(self):
        self.client.login(username="filtadmin", password="testpass123")
        data = json.loads(self.client.get(self.url).content)
        usernames = {u["username"] for u in data["users"]}
        self.assertIn("filtowner", usernames)

    def test_models_includes_both(self):
        self.client.login(username="filtadmin", password="testpass123")
        data = json.loads(self.client.get(self.url).content)
        self.assertIn("claude-3-5-sonnet", data["models"])
        self.assertIn("gpt-4", data["models"])

    def test_tool_names_includes_search_tool(self):
        self.client.login(username="filtadmin", password="testpass123")
        data = json.loads(self.client.get(self.url).content)
        self.assertIn("search_tool", data["tool_names"])

    def test_roles_includes_assistant_and_user(self):
        self.client.login(username="filtadmin", password="testpass123")
        data = json.loads(self.client.get(self.url).content)
        self.assertIn("assistant", data["roles"])
        self.assertIn("user", data["roles"])


class RowsEndpointTests(TestCase):

    def setUp(self):
        self.client = Client()
        self.superuser = make_superuser("rowsadmin")
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

    def test_returns_200(self):
        self.client.login(username="rowsadmin", password="testpass123")
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)

    def test_response_has_required_keys(self):
        self.client.login(username="rowsadmin", password="testpass123")
        data = json.loads(self.client.get(self.url).content)
        for key in ("rows", "page", "page_size", "total_count", "total_pages"):
            self.assertIn(key, data, msg=f"missing key: {key}")

    def test_total_count_matches_created(self):
        self.client.login(username="rowsadmin", password="testpass123")
        data = json.loads(self.client.get(self.url).content)
        self.assertEqual(data["total_count"], 5)

    def test_rows_contain_expected_fields(self):
        self.client.login(username="rowsadmin", password="testpass123")
        data = json.loads(self.client.get(self.url).content)
        self.assertGreater(len(data["rows"]), 0)
        row = data["rows"][0]
        for field in (
            "id", "message_uuid", "conversation_id", "conversation_name",
            "user_id", "username", "rating", "feedback_text",
            "effective_date", "role", "content_snippet",
        ):
            self.assertIn(field, row, msg=f"missing field in row: {field}")

    def test_username_in_row_matches_owner(self):
        self.client.login(username="rowsadmin", password="testpass123")
        data = json.loads(self.client.get(self.url).content)
        for row in data["rows"]:
            self.assertEqual(row["username"], "rowsowner")

    def test_conversation_name_in_row(self):
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
        data = json.loads(
            self.client.get(self.url, {"page_size": 2}).content
        )
        self.assertEqual(len(data["rows"]), 2)
        self.assertEqual(data["total_pages"], 3)

    def test_filter_by_rating_narrows_rows(self):
        self.client.login(username="rowsadmin", password="testpass123")
        data = json.loads(
            self.client.get(self.url, {"exact_rating": 5}).content
        )
        self.assertEqual(data["total_count"], 1)
        self.assertEqual(data["rows"][0]["rating"], 5)

    def test_filter_by_feedback_text_search(self):
        self.client.login(username="rowsadmin", password="testpass123")
        data = json.loads(
            self.client.get(self.url, {"feedback_text_search": "feedback 0"}).content
        )
        self.assertEqual(data["total_count"], 1)


class ExportEndpointTests(TestCase):

    def setUp(self):
        self.client = Client()
        self.superuser = make_superuser("exportadmin")
        self.owner = make_user("exportowner")
        self.convo = make_conversation(self.owner, name="export convo")
        make_message(
            self.convo, seq=1, rating=5, feedback_text="export test feedback"
        )
        self.url = reverse("api_feedback_dashboard_export")

    def test_returns_csv_content_type(self):
        self.client.login(username="exportadmin", password="testpass123")
        resp = self.client.get(self.url)
        self.assertNotEqual(resp.status_code, 403)
        self.assertIn("text/csv", resp.get("Content-Type", ""))

    def test_has_content_disposition_attachment(self):
        self.client.login(username="exportadmin", password="testpass123")
        resp = self.client.get(self.url)
        self.assertIn("attachment", resp.get("Content-Disposition", ""))
        self.assertIn(".csv", resp.get("Content-Disposition", ""))

    def test_csv_has_header_row(self):
        self.client.login(username="exportadmin", password="testpass123")
        resp = self.client.get(self.url)
        content = b"".join(resp.streaming_content).decode("utf-8")
        first_line = content.split("\n")[0]
        self.assertIn("username", first_line)
        self.assertIn("rating", first_line)
        self.assertIn("feedback_text", first_line)

    def test_csv_contains_data_row(self):
        self.client.login(username="exportadmin", password="testpass123")
        resp = self.client.get(self.url)
        content = b"".join(resp.streaming_content).decode("utf-8")
        self.assertIn("export test feedback", content)
        self.assertIn("exportowner", content)

    def test_filter_applied_to_export(self):
        # add a second message that should not appear when filtering by rating 5
        make_message(
            self.convo, seq=2, rating=1, feedback_text="low rating feedback"
        )
        self.client.login(username="exportadmin", password="testpass123")
        resp = self.client.get(self.url, {"exact_rating": 5})
        content = b"".join(resp.streaming_content).decode("utf-8")
        self.assertIn("export test feedback", content)
        self.assertNotIn("low rating feedback", content)