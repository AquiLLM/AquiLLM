"""
tests for the aggregate metrics layer

covers:
    get_summary_metrics counts, avg, distribution, date range
    get_summary_metrics with filters applied
    get_filter_options user list, roles, models, tool names, ratings
"""
from django.contrib.auth import get_user_model
from django.test import TestCase

from apps.chat.models import Message, WSConversation
from apps.platform_admin.services.feedback_aggregates import (
    get_filter_options,
    get_summary_metrics,
)
from apps.platform_admin.services.feedback_dataset import FeedbackFilters

User = get_user_model()


def make_user(username: str) -> User:
    return User.objects.create_user(
        username=username,
        email=f"{username}@test.com",
        password="testpass",
    )


def make_conversation(owner: User, name: str = "convo") -> WSConversation:
    return WSConversation.objects.create(owner=owner, name=name)


def make_message(conversation: WSConversation, seq: int = 1, **kwargs) -> Message:
    defaults = dict(
        role="assistant",
        content="content",
        sequence_number=seq,
        rating=None,
        feedback_text=None,
    )
    defaults.update(kwargs)
    return Message.objects.create(conversation=conversation, **defaults)


class SummaryMetricsTests(TestCase):

    def setUp(self):
        self.user = make_user("sumuser")
        self.convo = make_conversation(self.user)

        # rated 5, with text
        make_message(self.convo, seq=1, rating=5, feedback_text="excellent")
        # rated 3, no text
        make_message(self.convo, seq=2, rating=3, feedback_text=None)
        # no rating, text only
        make_message(self.convo, seq=3, rating=None, feedback_text="a comment")

    def test_total_count_is_three(self):
        s = get_summary_metrics(FeedbackFilters())
        self.assertEqual(s["total_count"], 3)

    def test_rated_count_excludes_no_rating_row(self):
        s = get_summary_metrics(FeedbackFilters())
        self.assertEqual(s["rated_count"], 2)

    def test_avg_rating_is_correct(self):
        # (5 + 3) / 2 = 4.0
        s = get_summary_metrics(FeedbackFilters())
        self.assertIsNotNone(s["avg_rating"])
        self.assertAlmostEqual(s["avg_rating"], 4.0, places=1)

    def test_avg_rating_is_none_when_no_ratings(self):
        # create a separate user and convo with only text feedback
        u = make_user("norating")
        c = make_conversation(u)
        Message.objects.all().delete()
        make_message(c, seq=1, rating=None, feedback_text="text only")
        s = get_summary_metrics(FeedbackFilters())
        self.assertIsNone(s["avg_rating"])

    def test_has_text_count(self):
        # msg1 has text, msg2 does not, msg3 has text = 2
        s = get_summary_metrics(FeedbackFilters())
        self.assertEqual(s["has_text_count"], 2)

    def test_rating_distribution_has_all_five_keys(self):
        s = get_summary_metrics(FeedbackFilters())
        dist = s["rating_distribution"]
        for k in (1, 2, 3, 4, 5):
            self.assertIn(k, dist, msg=f"missing key {k} in distribution")

    def test_rating_distribution_correct_counts(self):
        s = get_summary_metrics(FeedbackFilters())
        dist = s["rating_distribution"]
        self.assertEqual(dist[5], 1)
        self.assertEqual(dist[3], 1)
        self.assertEqual(dist[1], 0)
        self.assertEqual(dist[2], 0)
        self.assertEqual(dist[4], 0)

    def test_date_min_is_string_or_none(self):
        s = get_summary_metrics(FeedbackFilters())
        if s["date_min"] is not None:
            self.assertIsInstance(s["date_min"], str)
            # should end with Z for utc
            self.assertTrue(s["date_min"].endswith("Z"))

    def test_date_max_is_string_or_none(self):
        s = get_summary_metrics(FeedbackFilters())
        if s["date_max"] is not None:
            self.assertIsInstance(s["date_max"], str)
            self.assertTrue(s["date_max"].endswith("Z"))

    def test_filtered_summary_respects_filters(self):
        s = get_summary_metrics(FeedbackFilters(min_rating=5))
        self.assertEqual(s["total_count"], 1)
        self.assertEqual(s["avg_rating"], 5.0)

    def test_total_count_zero_on_no_match(self):
        s = get_summary_metrics(FeedbackFilters(exact_rating=2))
        self.assertEqual(s["total_count"], 0)
        self.assertIsNone(s["avg_rating"])

    def test_user_filter_narrows_summary(self):
        other_user = make_user("other")
        other_convo = make_conversation(other_user)
        make_message(other_convo, seq=1, rating=1, feedback_text="bad")
        s = get_summary_metrics(FeedbackFilters(user_id=self.user.id))
        # should only count the original three rows, not other_user's row
        self.assertEqual(s["total_count"], 3)


class FilterOptionsTests(TestCase):

    def setUp(self):
        self.user1 = make_user("optuser1")
        self.user2 = make_user("optuser2")
        self.convo1 = make_conversation(self.user1)
        self.convo2 = make_conversation(self.user2)

        make_message(
            self.convo1,
            seq=1,
            rating=4,
            role="assistant",
            model="claude-3-5-sonnet",
            tool_call_name=None,
        )
        make_message(
            self.convo2,
            seq=1,
            rating=2,
            role="user",
            model="gpt-4",
            tool_call_name="search_tool",
        )

    def test_users_list_not_empty(self):
        opts = get_filter_options()
        self.assertGreater(len(opts["users"]), 0)

    def test_users_have_id_and_username_keys(self):
        opts = get_filter_options()
        for u in opts["users"]:
            self.assertIn("id", u)
            self.assertIn("username", u)

    def test_users_includes_both_owners(self):
        opts = get_filter_options()
        usernames = {u["username"] for u in opts["users"]}
        self.assertIn("optuser1", usernames)
        self.assertIn("optuser2", usernames)

    def test_roles_contains_assistant_and_user(self):
        opts = get_filter_options()
        self.assertIn("assistant", opts["roles"])
        self.assertIn("user", opts["roles"])

    def test_models_contains_both_models(self):
        opts = get_filter_options()
        self.assertIn("claude-3-5-sonnet", opts["models"])
        self.assertIn("gpt-4", opts["models"])

    def test_models_excludes_null(self):
        # null models should never appear in the list
        opts = get_filter_options()
        self.assertNotIn(None, opts["models"])
        self.assertNotIn("", opts["models"])

    def test_tool_names_contains_search_tool(self):
        opts = get_filter_options()
        self.assertIn("search_tool", opts["tool_names"])

    def test_tool_names_excludes_null(self):
        opts = get_filter_options()
        self.assertNotIn(None, opts["tool_names"])

    def test_ratings_contains_present_values(self):
        opts = get_filter_options()
        self.assertIn(4, opts["ratings"])
        self.assertIn(2, opts["ratings"])

    def test_ratings_excludes_none(self):
        opts = get_filter_options()
        self.assertNotIn(None, opts["ratings"])

    def test_ratings_is_sorted(self):
        opts = get_filter_options()
        self.assertEqual(opts["ratings"], sorted(opts["ratings"]))