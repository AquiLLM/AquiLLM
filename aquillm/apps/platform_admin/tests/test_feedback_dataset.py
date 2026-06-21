"""
tests for the feedback dataset layer

covers:
    feedback_dataset_queryset base filtering
    annotation presence and correctness
    apply_filters for each filter type
    FeedbackFilters.from_request_params parsing
    get_filtered_queryset convenience function
"""
from datetime import timedelta

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.utils import timezone

from apps.chat.models import Message, WSConversation
from apps.platform_admin.services.feedback_dataset import (
    FeedbackFilters,
    apply_filters,
    feedback_dataset_queryset,
    get_filtered_queryset,
)

User = get_user_model()


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def make_user(username: str) -> User:
    """create a plain user for test ownership of conversations"""
    return User.objects.create_user(
        username=username,
        email=f"{username}@test.com",
        password="testpass",
    )


def make_conversation(owner: User, name: str = "test convo") -> WSConversation:
    """create a conversation owned by the given user"""
    return WSConversation.objects.create(owner=owner, name=name)


def make_message(conversation: WSConversation, seq: int = 1, **kwargs) -> Message:
    """
    create a message with sensible defaults,
    caller can override any field via kwargs
    """
    defaults = dict(
        role="assistant",
        content="test content",
        sequence_number=seq,
        rating=None,
        feedback_text=None,
    )
    defaults.update(kwargs)
    return Message.objects.create(conversation=conversation, **defaults)


# ---------------------------------------------------------------------------
# base queryset tests
# ---------------------------------------------------------------------------

class FeedbackDatasetQuerysetTests(TestCase):

    def setUp(self):
        self.user = make_user("qduser")
        self.convo = make_conversation(self.user)

    def _ids(self) -> set:
        return set(feedback_dataset_queryset().values_list("id", flat=True))

    def test_includes_row_with_rating(self):
        msg = make_message(self.convo, rating=4)
        self.assertIn(msg.id, self._ids())

    def test_includes_row_with_feedback_text(self):
        msg = make_message(self.convo, feedback_text="good")
        self.assertIn(msg.id, self._ids())

    def test_includes_row_with_both_rating_and_text(self):
        msg = make_message(self.convo, rating=5, feedback_text="excellent")
        self.assertIn(msg.id, self._ids())

    def test_excludes_row_with_no_rating_and_no_text(self):
        msg = make_message(self.convo, rating=None, feedback_text=None)
        self.assertNotIn(msg.id, self._ids())

    def test_excludes_row_with_empty_string_text_and_no_rating(self):
        # empty string feedback_text with no rating is not feedback-bearing
        msg = make_message(self.convo, rating=None, feedback_text="")
        self.assertNotIn(msg.id, self._ids())

    def test_includes_row_with_rating_and_empty_text(self):
        # rating alone is sufficient even if text is empty
        msg = make_message(self.convo, rating=3, feedback_text="")
        self.assertIn(msg.id, self._ids())

    def test_includes_all_three_roles(self):
        # all roles can be feedback-bearing, none are excluded at the base level
        make_message(self.convo, role="assistant", rating=3, seq=1)
        make_message(self.convo, role="user", rating=4, seq=2)
        make_message(self.convo, role="tool", rating=5, seq=3)
        roles = set(feedback_dataset_queryset().values_list("role", flat=True))
        self.assertIn("assistant", roles)
        self.assertIn("user", roles)
        self.assertIn("tool", roles)

    def test_effective_date_annotation_present(self):
        make_message(self.convo, rating=3)
        row = feedback_dataset_queryset().first()
        self.assertTrue(hasattr(row, "effective_date"))
        self.assertIsNotNone(row.effective_date)

    def test_content_snippet_annotation_present(self):
        make_message(self.convo, rating=3, content="hello world")
        row = feedback_dataset_queryset().first()
        self.assertTrue(hasattr(row, "content_snippet"))
        self.assertEqual(row.content_snippet, "hello world")

    def test_content_snippet_truncated_to_300(self):
        long_content = "x" * 500
        make_message(self.convo, rating=3, content=long_content)
        row = feedback_dataset_queryset().first()
        self.assertEqual(len(row.content_snippet), 300)

    def test_has_feedback_text_true_when_text_present(self):
        make_message(self.convo, feedback_text="some text")
        row = feedback_dataset_queryset().first()
        self.assertTrue(row.has_feedback_text)

    def test_has_feedback_text_false_when_only_rating(self):
        make_message(self.convo, rating=4, feedback_text=None)
        row = feedback_dataset_queryset().first()
        self.assertFalse(row.has_feedback_text)

    def test_username_annotation_matches_owner(self):
        make_message(self.convo, rating=3)
        row = feedback_dataset_queryset().first()
        self.assertEqual(row.username, "qduser")

    def test_user_id_annotation_matches_owner_pk(self):
        make_message(self.convo, rating=3)
        row = feedback_dataset_queryset().first()
        self.assertEqual(row.user_id, self.user.id)

    def test_conversation_name_annotation_matches_convo(self):
        make_message(self.convo, rating=3)
        row = feedback_dataset_queryset().first()
        self.assertEqual(row.conversation_name, "test convo")

    def test_effective_date_prefers_feedback_submitted_at(self):
        # when feedback_submitted_at is set it should be used over created_at
        future = timezone.now() + timedelta(days=10)
        msg = make_message(self.convo, rating=3)
        Message.objects.filter(pk=msg.pk).update(feedback_submitted_at=future)
        row = feedback_dataset_queryset().get(pk=msg.pk)
        self.assertEqual(
            row.effective_date.date(),
            future.date(),
        )

    def test_effective_date_falls_back_to_created_at(self):
        # when feedback_submitted_at is null, created_at should be used
        msg = make_message(self.convo, rating=3)
        row = feedback_dataset_queryset().get(pk=msg.pk)
        self.assertEqual(
            row.effective_date.date(),
            msg.created_at.date(),
        )


# ---------------------------------------------------------------------------
# apply_filters tests
# ---------------------------------------------------------------------------

class ApplyFiltersTests(TestCase):

    def setUp(self):
        self.user1 = make_user("alice")
        self.user2 = make_user("bob")
        self.convo1 = make_conversation(self.user1, name="alice convo")
        self.convo2 = make_conversation(self.user2, name="bob convo")

        # msg1: alice, rating 5, text, assistant
        self.msg1 = make_message(
            self.convo1,
            seq=1,
            rating=5,
            feedback_text="perfect",
            role="assistant",
            model="claude-3",
        )
        # msg2: bob, rating 2, text, user
        self.msg2 = make_message(
            self.convo2,
            seq=1,
            rating=2,
            feedback_text="needs work",
            role="user",
            model="gpt-4",
        )
        # msg3: alice, no rating, text only, assistant
        self.msg3 = make_message(
            self.convo1,
            seq=2,
            rating=None,
            feedback_text="just a comment",
            role="assistant",
            model="claude-3",
        )

    def _ids(self, filters: FeedbackFilters) -> set:
        return set(get_filtered_queryset(filters).values_list("id", flat=True))

    def test_no_filters_returns_all_feedback_rows(self):
        ids = self._ids(FeedbackFilters())
        self.assertIn(self.msg1.id, ids)
        self.assertIn(self.msg2.id, ids)
        self.assertIn(self.msg3.id, ids)

    def test_filter_user_id_includes_only_that_user(self):
        ids = self._ids(FeedbackFilters(user_id=self.user1.id))
        self.assertIn(self.msg1.id, ids)
        self.assertIn(self.msg3.id, ids)
        self.assertNotIn(self.msg2.id, ids)

    def test_filter_min_rating_excludes_lower(self):
        ids = self._ids(FeedbackFilters(min_rating=5))
        self.assertIn(self.msg1.id, ids)
        self.assertNotIn(self.msg2.id, ids)

    def test_filter_max_rating_excludes_higher(self):
        ids = self._ids(FeedbackFilters(max_rating=2))
        self.assertIn(self.msg2.id, ids)
        self.assertNotIn(self.msg1.id, ids)

    def test_filter_exact_rating(self):
        ids = self._ids(FeedbackFilters(exact_rating=5))
        self.assertIn(self.msg1.id, ids)
        self.assertNotIn(self.msg2.id, ids)

    def test_exact_rating_ignores_min_and_max(self):
        # exact_rating=5 should not be affected by min_rating=1, max_rating=3
        ids = self._ids(FeedbackFilters(exact_rating=5, min_rating=1, max_rating=3))
        self.assertIn(self.msg1.id, ids)
        self.assertNotIn(self.msg2.id, ids)

    def test_filter_feedback_text_search_case_insensitive(self):
        ids = self._ids(FeedbackFilters(feedback_text_search="PERFECT"))
        self.assertIn(self.msg1.id, ids)
        self.assertNotIn(self.msg2.id, ids)

    def test_filter_feedback_text_search_partial_match(self):
        ids = self._ids(FeedbackFilters(feedback_text_search="needs"))
        self.assertIn(self.msg2.id, ids)
        self.assertNotIn(self.msg1.id, ids)

    def test_filter_conversation_name_search(self):
        ids = self._ids(FeedbackFilters(conversation_name_search="alice"))
        self.assertIn(self.msg1.id, ids)
        self.assertNotIn(self.msg2.id, ids)

    def test_filter_role_assistant(self):
        ids = self._ids(FeedbackFilters(role="assistant"))
        self.assertIn(self.msg1.id, ids)
        self.assertIn(self.msg3.id, ids)
        self.assertNotIn(self.msg2.id, ids)

    def test_filter_role_user(self):
        ids = self._ids(FeedbackFilters(role="user"))
        self.assertIn(self.msg2.id, ids)
        self.assertNotIn(self.msg1.id, ids)

    def test_filter_model(self):
        ids = self._ids(FeedbackFilters(model="gpt-4"))
        self.assertIn(self.msg2.id, ids)
        self.assertNotIn(self.msg1.id, ids)

    def test_filter_has_feedback_text_true(self):
        # add a rating-only message with no text
        msg4 = make_message(self.convo1, seq=3, rating=3, feedback_text=None)
        ids = self._ids(FeedbackFilters(has_feedback_text=True))
        self.assertIn(self.msg1.id, ids)
        self.assertIn(self.msg2.id, ids)
        self.assertIn(self.msg3.id, ids)
        self.assertNotIn(msg4.id, ids)

    def test_filter_has_feedback_text_false(self):
        # add a rating-only message with no text
        msg4 = make_message(self.convo1, seq=3, rating=3, feedback_text=None)
        ids = self._ids(FeedbackFilters(has_feedback_text=False))
        self.assertIn(msg4.id, ids)
        self.assertNotIn(self.msg1.id, ids)
        self.assertNotIn(self.msg2.id, ids)
        self.assertNotIn(self.msg3.id, ids)

    def test_filter_has_feedback_text_none_returns_all(self):
        ids = self._ids(FeedbackFilters(has_feedback_text=None))
        self.assertIn(self.msg1.id, ids)
        self.assertIn(self.msg2.id, ids)
        self.assertIn(self.msg3.id, ids)

    def test_combined_filters_narrow_results(self):
        # user1 + min_rating 5 should return only msg1
        ids = self._ids(FeedbackFilters(user_id=self.user1.id, min_rating=5))
        self.assertIn(self.msg1.id, ids)
        self.assertNotIn(self.msg2.id, ids)
        self.assertNotIn(self.msg3.id, ids)

    def test_date_filter_start(self):
        # set feedback_submitted_at in the past for msg1 so we can filter it out
        past = timezone.now() - timedelta(days=30)
        Message.objects.filter(pk=self.msg1.pk).update(feedback_submitted_at=past)
        tomorrow = timezone.now() + timedelta(days=1)
        # start_date after msg1 should exclude it
        ids = self._ids(FeedbackFilters(start_date=tomorrow))
        self.assertNotIn(self.msg1.id, ids)

    def test_date_filter_end(self):
        future = timezone.now() + timedelta(days=30)
        Message.objects.filter(pk=self.msg1.pk).update(feedback_submitted_at=future)
        yesterday = timezone.now() - timedelta(days=1)
        # end_date before msg1 should exclude it
        ids = self._ids(FeedbackFilters(end_date=yesterday))
        self.assertNotIn(self.msg1.id, ids)


# ---------------------------------------------------------------------------
# FeedbackFilters.from_request_params tests
# ---------------------------------------------------------------------------

class FromRequestParamsTests(TestCase):

    def test_parses_user_id_int(self):
        f = FeedbackFilters.from_request_params({"user_id": "42"})
        self.assertEqual(f.user_id, 42)

    def test_invalid_user_id_gives_none(self):
        f = FeedbackFilters.from_request_params({"user_id": "notanint"})
        self.assertIsNone(f.user_id)

    def test_empty_user_id_gives_none(self):
        f = FeedbackFilters.from_request_params({"user_id": ""})
        self.assertIsNone(f.user_id)

    def test_parses_min_rating(self):
        f = FeedbackFilters.from_request_params({"min_rating": "3"})
        self.assertEqual(f.min_rating, 3)

    def test_parses_exact_rating(self):
        f = FeedbackFilters.from_request_params({"exact_rating": "4"})
        self.assertEqual(f.exact_rating, 4)

    def test_parses_feedback_text_search(self):
        f = FeedbackFilters.from_request_params({"feedback_text_search": " good "})
        self.assertEqual(f.feedback_text_search, "good")

    def test_whitespace_only_string_gives_none(self):
        f = FeedbackFilters.from_request_params({"feedback_text_search": "   "})
        self.assertIsNone(f.feedback_text_search)

    def test_parses_has_feedback_text_true_variants(self):
        for val in ("true", "True", "1", "yes"):
            f = FeedbackFilters.from_request_params({"has_feedback_text": val})
            self.assertTrue(f.has_feedback_text, msg=f"expected True for {val!r}")

    def test_parses_has_feedback_text_false_variants(self):
        for val in ("false", "False", "0", "no"):
            f = FeedbackFilters.from_request_params({"has_feedback_text": val})
            self.assertFalse(f.has_feedback_text, msg=f"expected False for {val!r}")

    def test_parses_date_range(self):
        f = FeedbackFilters.from_request_params({
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
        })
        self.assertIsNotNone(f.start_date)
        self.assertIsNotNone(f.end_date)

    def test_empty_params_give_all_none(self):
        f = FeedbackFilters.from_request_params({})
        self.assertIsNone(f.user_id)
        self.assertIsNone(f.min_rating)
        self.assertIsNone(f.start_date)
        self.assertIsNone(f.end_date)
        self.assertIsNone(f.feedback_text_search)
        self.assertIsNone(f.conversation_name_search)
        self.assertIsNone(f.role)
        self.assertIsNone(f.model)
        self.assertIsNone(f.tool_call_name)
        self.assertIsNone(f.has_feedback_text)

    def test_to_dict_round_trips(self):
        f = FeedbackFilters(user_id=7, min_rating=3, role="assistant")
        d = f.to_dict()
        self.assertEqual(d["user_id"], 7)
        self.assertEqual(d["min_rating"], 3)
        self.assertEqual(d["role"], "assistant")
        self.assertIsNone(d["exact_rating"])