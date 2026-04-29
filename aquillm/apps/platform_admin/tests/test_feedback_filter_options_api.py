"""Tests for the feedback dashboard filter-options API."""

import json

from django.contrib.auth import get_user_model
from django.test import Client, TestCase
from django.urls import reverse

from apps.chat.models import Message, WSConversation

User = get_user_model()


def make_superuser(username: str = "filtersadmin") -> User:
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


def make_conversation(owner: User, name: str = "filters convo") -> WSConversation:
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


class FeedbackFilterOptionsEndpointTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.superuser = make_superuser()
        self.owner = make_user("filtersowner")
        self.convo = make_conversation(self.owner)

        make_message(
            self.convo,
            seq=1,
            rating=4,
            role="assistant",
            model="claude-3-5-sonnet",
            tool_call_name=None,
        )
        make_message(
            self.convo,
            seq=2,
            rating=2,
            role="user",
            model="gpt-4",
            tool_call_name="search_tool",
        )

        self.url = reverse("api_feedback_dashboard_filters")

    def test_returns_200_for_superuser(self):
        self.client.login(username="filtersadmin", password="testpass123")
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)

    def test_content_type_is_json(self):
        self.client.login(username="filtersadmin", password="testpass123")
        resp = self.client.get(self.url)
        self.assertIn("application/json", resp["Content-Type"])

    def test_has_all_required_keys(self):
        self.client.login(username="filtersadmin", password="testpass123")
        data = json.loads(self.client.get(self.url).content)

        for key in ("users", "roles", "models", "tool_names", "ratings"):
            self.assertIn(key, data, msg=f"missing key: {key}")

    def test_users_include_feedback_owner(self):
        self.client.login(username="filtersadmin", password="testpass123")
        data = json.loads(self.client.get(self.url).content)

        usernames = {user["username"] for user in data["users"]}
        self.assertIn("filtersowner", usernames)

    def test_user_options_have_id_and_username(self):
        self.client.login(username="filtersadmin", password="testpass123")
        data = json.loads(self.client.get(self.url).content)

        self.assertGreater(len(data["users"]), 0)
        for user in data["users"]:
            self.assertIn("id", user)
            self.assertIn("username", user)

    def test_roles_include_assistant_and_user(self):
        self.client.login(username="filtersadmin", password="testpass123")
        data = json.loads(self.client.get(self.url).content)

        self.assertIn("assistant", data["roles"])
        self.assertIn("user", data["roles"])

    def test_models_include_non_null_models(self):
        self.client.login(username="filtersadmin", password="testpass123")
        data = json.loads(self.client.get(self.url).content)

        self.assertIn("claude-3-5-sonnet", data["models"])
        self.assertIn("gpt-4", data["models"])

    def test_models_exclude_null_and_empty_values(self):
        make_message(
            self.convo,
            seq=3,
            rating=5,
            feedback_text="no model",
            model=None,
        )

        self.client.login(username="filtersadmin", password="testpass123")
        data = json.loads(self.client.get(self.url).content)

        self.assertNotIn(None, data["models"])
        self.assertNotIn("", data["models"])

    def test_tool_names_include_non_null_tool_names(self):
        self.client.login(username="filtersadmin", password="testpass123")
        data = json.loads(self.client.get(self.url).content)

        self.assertIn("search_tool", data["tool_names"])

    def test_tool_names_exclude_null_values(self):
        self.client.login(username="filtersadmin", password="testpass123")
        data = json.loads(self.client.get(self.url).content)

        self.assertNotIn(None, data["tool_names"])
        self.assertNotIn("", data["tool_names"])

    def test_ratings_include_present_ratings(self):
        self.client.login(username="filtersadmin", password="testpass123")
        data = json.loads(self.client.get(self.url).content)

        self.assertIn(4, data["ratings"])
        self.assertIn(2, data["ratings"])

    def test_ratings_are_sorted(self):
        self.client.login(username="filtersadmin", password="testpass123")
        data = json.loads(self.client.get(self.url).content)

        self.assertEqual(data["ratings"], sorted(data["ratings"]))

    def test_regular_user_is_forbidden(self):
        regular = make_user("regularfilters")
        self.client.login(username=regular.username, password="testpass123")

        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 403)
