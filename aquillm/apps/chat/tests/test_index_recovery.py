"""Embedding outages retain useful text without declaring semantic indexing complete."""
from io import StringIO
from unittest.mock import patch

from celery.exceptions import Retry
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase

from apps.chat.models import Message, WSConversation
from apps.chat.services.conversation_indexing import index_conversation
from apps.chat.tasks.conversation_indexing import index_conversation_task


class IndexRecoveryTests(TestCase):
    def setUp(self):
        self.conversation = WSConversation.objects.create(
            owner=get_user_model().objects.create(username='index-recovery'), name='Known title'
        )
        for index, (role, content) in enumerate([
            ('user', 'Explain stellar calibration.'), ('assistant', 'Use reference stars.')
        ]):
            Message.objects.create(conversation=self.conversation, role=role, content=content, sequence_number=index)

    def test_failed_embeddings_remain_incomplete_and_recover_on_normal_retry(self):
        with patch('apps.chat.services.conversation_indexing.get_embeddings', side_effect=OSError('offline')), patch(
            'apps.chat.services.conversation_indexing.get_embedding', side_effect=OSError('offline')
        ):
            index_conversation(self.conversation.pk)
        self.conversation.refresh_from_db()
        self.assertFalse(self.conversation.index_complete)
        self.assertTrue(self.conversation.chunks.filter(embedding__isnull=True).exists())
        with patch('apps.chat.services.conversation_indexing.get_embeddings', return_value=[[0.0] * 1024]) as provider:
            index_conversation(self.conversation.pk)
        self.conversation.refresh_from_db()
        self.assertTrue(self.conversation.index_complete)
        self.assertFalse(self.conversation.chunks.filter(embedding__isnull=True).exists())
        provider.assert_called_once()

    def test_task_retries_incomplete_embeddings_with_a_delay(self):
        with patch('apps.chat.services.conversation_indexing.get_embeddings', side_effect=OSError('offline')), patch(
            'apps.chat.services.conversation_indexing.get_embedding', side_effect=OSError('offline')
        ), patch.object(index_conversation_task, 'retry', side_effect=Retry()) as retry:
            with self.assertRaises(Retry):
                index_conversation_task.run(self.conversation.pk)
        self.assertGreaterEqual(retry.call_args.kwargs['countdown'], 60)

    def test_async_force_reaches_the_worker(self):
        with patch.object(index_conversation_task, 'apply_async') as queue:
            call_command('index_conversations', '--async', '--force', stdout=StringIO())
        self.assertTrue(queue.call_args.kwargs['kwargs']['force'])
        with patch('apps.chat.services.conversation_indexing.index_conversation') as indexer:
            index_conversation_task.run(**queue.call_args.kwargs['kwargs'])
        indexer.assert_called_once_with(self.conversation.pk, force=True)

    def test_legacy_complete_flag_does_not_skip_null_vectors(self):
        with patch('apps.chat.services.conversation_indexing.get_embeddings', side_effect=OSError('offline')), patch(
            'apps.chat.services.conversation_indexing.get_embedding', side_effect=OSError('offline')
        ):
            index_conversation(self.conversation.pk)
        WSConversation.objects.filter(pk=self.conversation.pk).update(index_complete=True)
        with patch('apps.chat.services.conversation_indexing.get_embeddings', side_effect=OSError('still offline')), patch(
            'apps.chat.services.conversation_indexing.get_embedding', side_effect=OSError('still offline')
        ):
            index_conversation(self.conversation.pk)
        self.conversation.refresh_from_db()
        self.assertFalse(self.conversation.index_complete)
        with patch('apps.chat.services.conversation_indexing.get_embeddings', return_value=[[0.0] * 1024]):
            index_conversation(self.conversation.pk)
        self.assertFalse(self.conversation.chunks.filter(embedding__isnull=True).exists())
