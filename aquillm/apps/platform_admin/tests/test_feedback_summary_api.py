"""Tests for the feedback dashboard summary API."""

import json

from django.contrib.auth import get_user_model
from django.test import Client, TestCase
from django.urls import reverse

from apps.chat.models import Message, WSConversation

User = get_user_model()


def make_superuser(username: str = "summaryadmin") -> User:
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


def make_conversation(owner: User, name: str = "summary convo") -> WSConversation:
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


class FeedbackSummaryEndpointTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.superuser = make_superuser()
        self.owner = make_user("summaryowner")
        self.convo = make_conversation(self.owner)

        make_message(self.convo, seq=1, rating=5, feedback_text="excellent")
        make_message(self.convo, seq=2, rating=3, feedback_text=None)
        make_message(self.convo, seq=3, rating=None, feedback_text="comment only")

        self.url = reverse("api_feedback_dashboard_summary")

    def test_returns_200_for_superuser(self):
        self.client.login(username="summaryadmin", password="testpass123")
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)

    def test_content_type_is_json(self):
        self.client.login(username="summaryadmin", password="testpass123")
        resp = self.client.get(self.url)
        self.assertIn("application/json", resp["Content-Type"])

    def test_has_all_required_keys(self):
        self.client.login(username="summaryadmin", password="testpass123")
        data = json.loads(self.client.get(self.url).content)

        for key in (
            "total_count",
            "rated_count",
            "avg_rating",
            "rating_distribution",
            "has_text_count",
            "date_min",
            "date_max",
        ):
            self.assertIn(key, data, msg=f"missing key: {key}")

    def test_total_count_correct(self):
        self.client.login(username="summaryadmin", password="testpass123")
        data = json.loads(self.client.get(self.url).content)
        self.assertEqual(data["total_count"], 3)

    def test_rated_count_correct(self):
        self.client.login(username="summaryadmin", password="testpass123")
        data = json.loads(self.client.get(self.url).content)
        self.assertEqual(data["rated_count"], 2)

    def test_average_rating_correct(self):
        self.client.login(username="summaryadmin", password="testpass123")
        data = json.loads(self.client.get(self.url).content)
        self.assertAlmostEqual(data["avg_rating"], 4.0, places=1)

    def test_has_text_count_correct(self):
        self.client.login(username="summaryadmin", password="testpass123")
        data = json.loads(self.client.get(self.url).content)
        self.assertEqual(data["has_text_count"], 2)

    def test_rating_distribution_keys_are_strings(self):
        self.client.login(username="summaryadmin", password="testpass123")
        data = json.loads(self.client.get(self.url).content)

        for key in data["rating_distribution"]:
            self.assertIsInstance(key, str)

    def test_rating_distribution_counts_are_correct(self):
        self.client.login(username="summaryadmin", password="testpass123")
        data = json.loads(self.client.get(self.url).content)

        self.assertEqual(data["rating_distribution"]["5"], 1)
        self.assertEqual(data["rating_distribution"]["3"], 1)
        self.assertEqual(data["rating_distribution"]["1"], 0)
        self.assertEqual(data["rating_distribution"]["2"], 0)
        self.assertEqual(data["rating_distribution"]["4"], 0)

    def test_filter_by_user_narrows_summary(self):
        other = make_user("othersummary")
        other_convo = make_conversation(other, name="other summary convo")
        make_message(other_convo, seq=1, rating=1, feedback_text="bad")

        self.client.login(username="summaryadmin", password="testpass123")
        data = json.loads(
            self.client.get(self.url, {"user_id": self.owner.id}).content
        )

        self.assertEqual(data["total_count"], 3)
        self.assertEqual(data["rated_count"], 2)

    def test_filter_by_exact_rating_narrows_summary(self):
        self.client.login(username="summaryadmin", password="testpass123")
        data = json.loads(
            self.client.get(self.url, {"exact_rating": 5}).content
        )

        self.assertEqual(data["total_count"], 1)
        self.assertEqual(data["rated_count"], 1)
        self.assertEqual(data["avg_rating"], 5.0)

    def test_no_matching_rows_returns_zero_summary(self):
        self.client.login(username="summaryadmin", password="testpass123")
        data = json.loads(
            self.client.get(self.url, {"exact_rating": 2}).content
        )

        self.assertEqual(data["total_count"], 0)
        self.assertEqual(data["rated_count"], 0)
        self.assertIsNone(data["avg_rating"])

    def test_regular_user_is_forbidden(self):
        regular = make_user("regularsummary")
        self.client.login(username=regular.username, password="testpass123")

        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 403)
