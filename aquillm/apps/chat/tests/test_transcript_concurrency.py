"""Real database regressions for stale WebSocket transcript publication."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.db import close_old_connections, transaction
from django.test import TestCase, TransactionTestCase, skipUnlessDBFeature

from apps.chat.services.feedback import apply_message_rating
from apps.chat.tests.chat_message_test_support import _FakeTitleLLM
from aquillm import message_adapters
from aquillm.llm import AssistantMessage, Conversation, UserMessage
from aquillm.models import WSConversation


class TranscriptConcurrencyTests(TestCase):
    def setUp(self):
        owner = get_user_model().objects.create_user(username="transcript-owner")
        self.db = WSConversation.objects.create(owner=owner, system_prompt="Base")
        message_adapters.save_conversation_to_db(
            Conversation(system="Runtime", messages=[UserMessage(content="Question")]),
            self.db,
        )

    def load_writer(self):
        handle = WSConversation.objects.get(pk=self.db.pk)
        return handle, message_adapters.load_conversation_from_db(handle)

    def contents(self):
        return list(
            self.db.db_messages.order_by("sequence_number").values_list(
                "content", flat=True
            )
        )

    def test_stale_snapshot_cannot_delete_a_newer_append(self):
        current_db, current = self.load_writer()
        stale_db, stale = self.load_writer()
        current.messages.append(
            AssistantMessage(content="Saved answer", stop_reason="end_turn")
        )
        message_adapters.save_conversation_to_db(current, current_db)

        # The missing check used to silently delete Saved answer here.
        with self.assertRaises(RuntimeError):
            message_adapters.save_conversation_to_db(stale, stale_db)

        self.assertEqual(self.contents(), ["Question", "Saved answer"])
        self.db.refresh_from_db()
        self.assertEqual(self.db.updated_at, current_db.updated_at)

    def test_stale_overlapping_edit_raises_a_recoverable_conflict(self):
        current_db, current = self.load_writer()
        stale_db, stale = self.load_writer()
        current.messages[0].content = "Accepted edit"
        stale.messages[0].content = "Stale edit"
        message_adapters.save_conversation_to_db(current, current_db)

        with self.assertRaises(RuntimeError) as raised:
            message_adapters.save_conversation_to_db(stale, stale_db)

        self.assertEqual(type(raised.exception).__name__, "ConversationConflictError")
        self.assertEqual(self.contents(), ["Accepted edit"])

    def test_sequential_saves_allow_runtime_replacement_and_shortening(self):
        handle, convo = self.load_writer()
        original = self.db.db_messages.get()
        convo = convo + AssistantMessage(content="Answer", stop_reason="end_turn")
        message_adapters.save_conversation_to_db(convo, handle)
        convo = convo + UserMessage(content="Follow-up")
        message_adapters.save_conversation_to_db(convo, handle)
        message_adapters.save_conversation_to_db(
            Conversation(system="Runtime memory", messages=[convo.messages[0]]), handle
        )

        self.assertEqual(self.contents(), ["Question"])
        preserved = self.db.db_messages.get()
        self.assertEqual(
            (preserved.pk, preserved.created_at), (original.pk, original.created_at)
        )
        self.db.refresh_from_db()
        self.assertEqual(self.db.system_prompt, "Base")

    def test_stale_save_does_not_erase_external_feedback(self):
        handle, convo = self.load_writer()
        convo.messages.append(
            AssistantMessage(content="Answer", stop_reason="end_turn")
        )
        message_adapters.save_conversation_to_db(convo, handle)
        apply_message_rating(self.db.pk, convo.messages[-1].message_uuid, 5)

        with self.assertRaises(RuntimeError):
            message_adapters.save_conversation_to_db(convo, handle)

        answer = self.db.db_messages.get(role="assistant")
        self.assertEqual(answer.rating, 5)
        self.assertIsNotNone(answer.feedback_submitted_at)

    def test_own_feedback_followed_by_append_is_not_a_conflict(self):
        handle, convo = self.load_writer()
        convo.messages.append(
            AssistantMessage(content="Answer", stop_reason="end_turn")
        )
        message_adapters.save_conversation_to_db(convo, handle)
        apply_message_rating(self.db.pk, convo.messages[-1].message_uuid, 5)
        # The rating consumer updates its runtime message after the DB write.
        convo.messages[-1].rating = 5
        submitted_at = self.db.db_messages.get(role="assistant").feedback_submitted_at
        convo.messages.append(UserMessage(content="Next question"))
        message_adapters.save_conversation_to_db(convo, handle)

        answer = self.db.db_messages.get(role="assistant")
        self.assertEqual(
            (answer.rating, answer.feedback_submitted_at), (5, submitted_at)
        )
        self.assertEqual(self.contents(), ["Question", "Answer", "Next question"])

    def test_collection_and_title_updates_do_not_invalidate_transcript(self):
        handle, convo = self.load_writer()
        self.db.selected_collection_ids = [42]
        self.db.name = "Renamed elsewhere"
        self.db.save(update_fields=["selected_collection_ids", "name", "updated_at"])
        convo.messages.append(
            AssistantMessage(content="Answer", stop_reason="end_turn")
        )
        message_adapters.save_conversation_to_db(convo, handle)

        self.db.refresh_from_db()
        self.assertEqual(self.db.selected_collection_ids, [42])
        self.assertEqual(self.db.name, "Renamed elsewhere")
        self.assertEqual(self.contents(), ["Question", "Answer"])

    def test_load_refreshes_selection_and_overrides_on_a_stale_handle(self):
        stale_handle = WSConversation.objects.get(pk=self.db.pk)
        other_connection = WSConversation.objects.get(pk=self.db.pk)
        other_connection.selected_collection_ids = [42]
        other_connection.skill_overrides = {
            "apps_documents.rawtextdocument:123": False
        }
        other_connection.save(
            update_fields=["selected_collection_ids", "skill_overrides", "updated_at"]
        )

        loaded = message_adapters.load_conversation_from_db(stale_handle)

        self.assertEqual([message.content for message in loaded], ["Question"])
        self.assertEqual(stale_handle.selected_collection_ids, [42])
        self.assertEqual(
            stale_handle.skill_overrides,
            {"apps_documents.rawtextdocument:123": False},
        )

    def test_auto_title_does_not_overwrite_newer_collection_selection(self):
        handle, _ = self.load_writer()
        self.db.selected_collection_ids = [42]
        self.db.save(update_fields=["selected_collection_ids", "updated_at"])
        config = SimpleNamespace(llm_interface=_FakeTitleLLM("A useful title"))
        with patch(
            "apps.chat.models.conversation.apps.get_app_config", return_value=config
        ):
            handle.set_name()
        self.db.refresh_from_db()
        self.assertEqual(self.db.name, "A useful title")
        self.assertEqual(self.db.selected_collection_ids, [42])

    def test_unloaded_existing_transcript_requires_an_explicit_load(self):
        handle = WSConversation.objects.get(pk=self.db.pk)
        replacement = Conversation(system="Runtime", messages=[])
        with self.assertRaises(RuntimeError):
            message_adapters.save_conversation_to_db(replacement, handle)
        self.assertEqual(self.contents(), ["Question"])

    def test_reload_after_conflict_allows_explicit_retry(self):
        current_db, current = self.load_writer()
        stale_db, stale = self.load_writer()
        current.messages[0].content = "Accepted edit"
        message_adapters.save_conversation_to_db(current, current_db)
        with self.assertRaises(RuntimeError):
            message_adapters.save_conversation_to_db(stale, stale_db)

        reloaded = message_adapters.load_conversation_from_db(stale_db)
        reloaded.messages.append(
            AssistantMessage(content="Retry answer", stop_reason="end_turn")
        )
        message_adapters.save_conversation_to_db(reloaded, stale_db)
        self.assertEqual(self.contents(), ["Accepted edit", "Retry answer"])

    def test_outer_publication_rollback_requires_reload_and_preserves_rows(self):
        handle, convo = self.load_writer()
        original_updated_at = handle.updated_at
        convo.messages.append(
            AssistantMessage(content="Cancelled answer", stop_reason="end_turn")
        )
        with self.assertRaisesRegex(ValueError, "cancel publication"):
            with transaction.atomic():
                message_adapters.save_conversation_to_db(convo, handle)
                raise ValueError("cancel publication")

        self.db.refresh_from_db()
        self.assertEqual(self.contents(), ["Question"])
        self.assertEqual(self.db.updated_at, original_updated_at)
        with self.assertRaises(message_adapters.ConversationConflictError):
            message_adapters.save_conversation_to_db(convo, handle)
        refreshed = message_adapters.load_conversation_from_db(handle)
        refreshed.messages.append(
            AssistantMessage(content="Fresh answer", stop_reason="end_turn")
        )
        message_adapters.save_conversation_to_db(refreshed, handle)
        self.assertEqual(self.contents(), ["Question", "Fresh answer"])

    def test_two_empty_writers_cannot_both_publish_first_message(self):
        empty = WSConversation.objects.create(owner=self.db.owner, system_prompt="Base")
        other = WSConversation.objects.get(pk=empty.pk)
        first = message_adapters.load_conversation_from_db(empty)
        second = message_adapters.load_conversation_from_db(other)
        first.messages.append(UserMessage(content="First writer"))
        second.messages.append(UserMessage(content="Second writer"))
        message_adapters.save_conversation_to_db(first, empty)
        with self.assertRaises(RuntimeError):
            message_adapters.save_conversation_to_db(second, other)
        self.assertEqual(
            list(empty.db_messages.values_list("content", flat=True)), ["First writer"]
        )


class ParallelTranscriptPublicationTests(TransactionTestCase):
    @skipUnlessDBFeature("has_select_for_update")
    def test_parallel_first_writes_have_one_winner_and_one_conflict(self):
        owner = get_user_model().objects.create_user(username="parallel-owner")
        stored = WSConversation.objects.create(owner=owner, system_prompt="Base")
        ready = Barrier(2)

        def publish(content):
            close_old_connections()
            try:
                handle = WSConversation.objects.get(pk=stored.pk)
                convo = message_adapters.load_conversation_from_db(handle)
                convo.messages.append(UserMessage(content=content))
                ready.wait(timeout=5)
                try:
                    message_adapters.save_conversation_to_db(convo, handle)
                except RuntimeError as exc:
                    return type(exc).__name__, content
                return "saved", content
            finally:
                close_old_connections()

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(publish, ("Writer one", "Writer two")))

        self.assertEqual(
            sorted(status for status, _ in outcomes),
            ["ConversationConflictError", "saved"],
        )
        winner = next(content for status, content in outcomes if status == "saved")
        self.assertEqual(
            list(stored.db_messages.values_list("content", flat=True)), [winner]
        )
