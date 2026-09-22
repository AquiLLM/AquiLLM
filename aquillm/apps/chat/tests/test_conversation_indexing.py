"""DB-backed index -> search round-trip for past-chat search (mocked embeddings)."""
from __future__ import annotations

from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase

from apps.chat.models import ConversationChunk, Message, WSConversation
from apps.chat.services.conversation_indexing import index_conversation
from apps.chat.services.conversation_search import search_conversation_chunks
from apps.chat.tasks.conversation_indexing import (
    enqueue_index_conversation_task,
    index_conversation_task,
)

User = get_user_model()

_ZERO_VEC = [0.0] * 1024


def _fake_get_embeddings(texts, input_type="search_document"):
    return [list(_ZERO_VEC) for _ in texts]


def _fake_get_embedding(query, input_type="search_query"):
    return list(_ZERO_VEC)


def _add_messages(convo, pairs):
    seq = 0
    for role, content in pairs:
        Message.objects.create(
            conversation=convo, role=role, content=content, sequence_number=seq
        )
        seq += 1


class ConversationIndexingTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="alice", password="x")
        self.convo = WSConversation.objects.create(owner=self.user, name="Telescopes")
        _add_messages(
            self.convo,
            [
                ("user", "How do I align the quasar spectrograph?"),
                ("assistant", "Use the calibration lamp and the alignment jig first."),
                ("user", "And the exposure time?"),
                ("assistant", "Start near 300 seconds for the quasar field."),
            ],
        )

    @patch(
        "apps.chat.services.conversation_indexing.get_embeddings",
        side_effect=_fake_get_embeddings,
    )
    def test_reopening_chat_does_not_discard_pending_index(self, _mock):
        from aquillm.message_adapters import (
            load_conversation_from_db,
            save_conversation_to_db,
        )

        with patch.object(index_conversation_task, "apply_async") as queue:
            enqueue_index_conversation_task(
                self.convo.id, self.convo.updated_at.isoformat()
            )
        pending = queue.call_args.kwargs["kwargs"]
        save_conversation_to_db(load_conversation_from_db(self.convo), self.convo)

        index_conversation_task(**pending)

        self.assertTrue(ConversationChunk.objects.filter(conversation=self.convo).exists())

    @patch(
        "apps.chat.services.conversation_indexing.get_embeddings",
        side_effect=_fake_get_embeddings,
    )
    def test_collection_selection_does_not_discard_pending_index(self, _mock):
        with patch.object(index_conversation_task, "apply_async") as queue:
            enqueue_index_conversation_task(
                self.convo.id, self.convo.updated_at.isoformat()
            )
        pending = queue.call_args.kwargs["kwargs"]
        self.convo.selected_collection_ids = [123]
        self.convo.save(update_fields=["selected_collection_ids", "updated_at"])

        index_conversation_task(**pending)

        self.assertTrue(ConversationChunk.objects.filter(conversation=self.convo).exists())

    @patch(
        "apps.chat.services.conversation_indexing.get_embeddings",
        side_effect=_fake_get_embeddings,
    )
    def test_changed_transcript_is_requeued_then_indexed(self, _mock):
        with patch.object(index_conversation_task, "apply_async") as queue:
            enqueue_index_conversation_task(
                self.convo.id, self.convo.updated_at.isoformat()
            )
        pending = queue.call_args.kwargs["kwargs"]
        self.convo.db_messages.filter(sequence_number=3).update(
            content="The exposure time is now 450 seconds."
        )
        self.convo.save(update_fields=["updated_at"])

        with patch.object(index_conversation_task, "apply_async") as queue:
            index_conversation_task(**pending)
        self.assertFalse(ConversationChunk.objects.filter(conversation=self.convo).exists())
        self.assertEqual(queue.call_count, 1)

        index_conversation_task(**queue.call_args.kwargs["kwargs"])

        contents = " ".join(self.convo.chunks.values_list("content", flat=True))
        self.assertIn("450 seconds", contents)

    def test_slower_old_index_cannot_replace_newer_transcript_chunks(self):
        def embed_after_newer_index_finishes(texts, input_type="search_document"):
            self.convo.db_messages.filter(sequence_number=3).update(
                content="The exposure time is now 450 seconds."
            )
            with patch(
                "apps.chat.services.conversation_indexing.get_embeddings",
                side_effect=_fake_get_embeddings,
            ):
                index_conversation(self.convo.id)
            return _fake_get_embeddings(texts, input_type=input_type)

        with patch(
            "apps.chat.services.conversation_indexing.get_embeddings",
            side_effect=embed_after_newer_index_finishes,
        ):
            index_conversation_task(self.convo.id)

        contents = " ".join(self.convo.chunks.values_list("content", flat=True))
        self.assertIn("450 seconds", contents)
        self.assertNotIn("300 seconds", contents)

    @patch(
        "apps.chat.services.conversation_indexing.get_embeddings",
        side_effect=_fake_get_embeddings,
    )
    def test_index_creates_chunks_and_is_idempotent(self, _mock):
        n = index_conversation(self.convo.id)
        self.assertGreaterEqual(n, 1)
        chunks = ConversationChunk.objects.filter(conversation=self.convo)
        self.assertEqual(chunks.count(), n)
        self.assertTrue(all(c.embedding is not None for c in chunks))
        self.assertEqual(chunks.order_by("chunk_number").first().start_sequence, 0)

        self.convo.refresh_from_db()
        self.assertTrue(self.convo.index_complete)
        self.assertTrue(self.convo.indexed_transcript_hash)

        # Re-running with an unchanged transcript should not rebuild.
        with patch(
            "apps.chat.services.conversation_indexing.get_embeddings",
            side_effect=_fake_get_embeddings,
        ) as mock2:
            again = index_conversation(self.convo.id)
            self.assertEqual(again, n)
            mock2.assert_not_called()

    @patch(
        "apps.chat.services.conversation_indexing.get_embeddings",
        side_effect=_fake_get_embeddings,
    )
    @patch("aquillm.utils.get_embedding", side_effect=_fake_get_embedding)
    def test_search_is_owner_scoped_and_excludes_current(self, _e, _es):
        index_conversation(self.convo.id)

        # A second user's conversation must never surface for alice.
        other = User.objects.create_user(username="bob", password="x")
        other_convo = WSConversation.objects.create(owner=other, name="Bob chat")
        _add_messages(
            other_convo, [("user", "quasar quasar quasar"), ("assistant", "ok")]
        )
        index_conversation(other_convo.id)

        results = search_conversation_chunks(self.user, "quasar", top_k=5)
        self.assertTrue(results)
        self.assertTrue(all(c.conversation.owner_id == self.user.id for c in results))

        # Excluding the only matching conversation yields nothing for alice.
        excluded = search_conversation_chunks(
            self.user, "quasar", top_k=5, exclude_conversation_id=self.convo.id
        )
        self.assertEqual(excluded, [])

    @patch(
        "apps.chat.services.conversation_indexing.get_embeddings",
        side_effect=_fake_get_embeddings,
    )
    def test_reindex_replaces_chunks_when_transcript_changes(self, _mock):
        index_conversation(self.convo.id)
        before = self.convo.indexed_transcript_hash

        Message.objects.create(
            conversation=self.convo,
            role="user",
            content="One more thing about the dome flats?",
            sequence_number=99,
        )
        index_conversation(self.convo.id)
        self.convo.refresh_from_db()
        self.assertNotEqual(self.convo.indexed_transcript_hash, before)
        contents = " ".join(
            ConversationChunk.objects.filter(conversation=self.convo).values_list(
                "content", flat=True
            )
        )
        self.assertIn("dome flats", contents)
