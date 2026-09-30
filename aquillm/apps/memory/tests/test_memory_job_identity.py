"""Memory work follows transcript changes, independently of collection/title edits."""
from datetime import timedelta
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.utils import timezone

from apps.chat.models import Message, WSConversation
from apps.memory.models import ConversationMemoryJob
from aquillm.tasks import create_conversation_memories_task, enqueue_conversation_memories_task


class MemoryJobIdentityTests(TestCase):
    def setUp(self):
        self.conversation = WSConversation.objects.create(owner=get_user_model().objects.create(username='memory-identity'))
        self.message = Message.objects.create(conversation=self.conversation, role='assistant', content='Saved answer', sequence_number=0)

    def queued(self):
        with patch.object(create_conversation_memories_task, 'apply_async') as publish:
            with self.captureOnCommitCallbacks(execute=True):
                enqueue_conversation_memories_task(self.conversation.pk, self.conversation.updated_at.isoformat())
        return publish.call_args.kwargs['kwargs']

    def test_selection_edit_does_not_drop_pending_memory(self):
        payload = self.queued()
        self.conversation.selected_collection_ids = [7]
        self.conversation.save(update_fields=['selected_collection_ids', 'updated_at'])
        with patch('aquillm.tasks._user_is_globally_idle', return_value=True), patch(
            'aquillm.tasks._run_conversation_memory_creation'
        ) as create:
            create_conversation_memories_task.run(**payload)
        create.assert_called_once()

    def test_stale_transcript_keeps_one_pending_job_with_current_identity(self):
        payload = self.queued()
        self.message.content = 'Changed answer'
        self.message.save(update_fields=['content'])
        with patch(
            'aquillm.tasks._run_conversation_memory_creation'
        ) as create:
            create_conversation_memories_task.run(**payload)
        create.assert_not_called()
        from aquillm.tasks import _memory_transcript_hash
        job = ConversationMemoryJob.objects.get(conversation=self.conversation)
        self.assertEqual(job.state, 'pending')
        self.assertEqual(job.desired_transcript_hash, _memory_transcript_hash(self.conversation))

    def test_legacy_timestamp_job_survives_metadata_edits(self):
        old_snapshot = (timezone.now() - timedelta(minutes=10)).isoformat()
        with patch('aquillm.tasks._user_is_globally_idle', return_value=True), patch(
            'aquillm.tasks._run_conversation_memory_creation'
        ) as create:
            create_conversation_memories_task.run(self.conversation.pk, old_snapshot)
        create.assert_called_once()

    def test_long_activity_defers_existing_job_without_exhausting_it(self):
        payload = self.queued()
        with patch('aquillm.tasks._user_is_globally_idle', return_value=False), patch.object(
            create_conversation_memories_task, 'retry'
        ) as retry:
            create_conversation_memories_task.run(**payload)
            for index in range(20):
                create_conversation_memories_task.run(self.conversation.pk, queued_transcript_hash=f'old-{index}')
        retry.assert_not_called()
        self.assertEqual(ConversationMemoryJob.objects.count(), 1)
        self.assertEqual(ConversationMemoryJob.objects.get().state, 'pending')
