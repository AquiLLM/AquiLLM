"""Exercise durable completion through the real Mem0 write boundary."""
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from threading import Event
from unittest.mock import Mock, patch

import pytest
from django.contrib.auth import get_user_model
from django.db import close_old_connections
from django.utils import timezone

from apps.chat.models import Message, WSConversation
from apps.memory.models import ConversationMemoryJob, EpisodicMemory
from aquillm import memory
from aquillm.tasks import create_conversation_memories_task, enqueue_conversation_memories_task, recover_conversation_memory_jobs
from lib.memory.mem0 import operations

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.fixture
def source(monkeypatch):
    convo = WSConversation.objects.create(owner=get_user_model().objects.create(username='mem0-completion'))
    Message.objects.create(conversation=convo, role='user', content='Remember our telescope setup.', sequence_number=0)
    Message.objects.create(conversation=convo, role='assistant', content='I will remember it.', sequence_number=1)
    monkeypatch.setattr(memory, 'use_mem0', lambda: True)
    monkeypatch.setattr(memory, '_enqueue_profile_fact_promotion', lambda **kwargs: None)
    monkeypatch.setattr('aquillm.tasks._user_is_globally_idle', lambda *args, **kwargs: True)
    monkeypatch.setattr('aquillm.utils.get_embedding', lambda *args, **kwargs: [0.0] * 1024)
    with patch.object(create_conversation_memories_task, 'apply_async') as publish:
        enqueue_conversation_memories_task(convo.pk)
    return convo, publish.call_args.kwargs['kwargs']


@pytest.mark.parametrize('dual_write', [False, True])
@pytest.mark.parametrize('failure', ['provider', 'missing_client', 'graph_fallback'])
def test_actual_mem0_failure_remains_pending_and_successful_no_add_finishes(source, monkeypatch, dual_write, failure):
    convo, payload = source
    monkeypatch.setattr(memory, 'MEM0_DUAL_WRITE_LOCAL', dual_write)
    monkeypatch.setenv('MEM0_GRAPH_ENABLED', '1' if failure == 'graph_fallback' else '0')
    monkeypatch.setenv('MEM0_GRAPH_FAIL_OPEN', '1')

    class Client:
        failed = True
        calls = 0

        def add(self, *args, **kwargs):
            self.calls += 1
            if self.failed:
                raise OSError('provider unavailable')
            return {'results': []}  # Valid inference with nothing to add.

    client = Client()
    promotion = Mock()
    monkeypatch.setattr(memory, '_enqueue_profile_fact_promotion', promotion)
    monkeypatch.setattr(operations, 'get_mem0_oss', lambda: None if failure == 'missing_client' and client.failed else client)
    with pytest.raises((OSError, RuntimeError)):
        create_conversation_memories_task.run(**payload)
    job = ConversationMemoryJob.objects.get(conversation=convo)
    assert job.state == 'pending'
    assert job.completed_transcript_hash == ''
    # Local dedupe must never hide this unfinished remote operation on retry.
    assert not EpisodicMemory.objects.filter(conversation=convo).exists()
    promotion.assert_not_called()
    client.failed = False
    ConversationMemoryJob.objects.filter(pk=convo.pk).update(next_attempt_at=timezone.now() - timedelta(seconds=1))
    with patch.object(create_conversation_memories_task, 'apply_async') as publish:
        recover_conversation_memory_jobs.run()
    recovered = publish.call_args.kwargs['kwargs']
    create_conversation_memories_task.run(**recovered)
    calls = client.calls
    create_conversation_memories_task.run(**recovered)
    job.refresh_from_db()
    assert job.state == 'idle'
    assert job.completed_transcript_hash == job.desired_transcript_hash
    assert EpisodicMemory.objects.filter(conversation=convo).count() == int(dual_write)
    assert client.calls == calls
    promotion.assert_called_once()


def test_slow_actual_mem0_call_keeps_execution_owner_past_wrapper_deadline(source, monkeypatch):
    convo, payload = source
    monkeypatch.setattr(memory, 'MEM0_DUAL_WRITE_LOCAL', False)
    monkeypatch.setattr(operations, 'MEM0_TIMEOUT_SECONDS', 0.01)
    monkeypatch.setenv('MEM0_GRAPH_ENABLED', '0')
    started, release = Event(), Event()

    class Client:
        calls = 0

        def add(self, *args, **kwargs):
            self.calls += 1
            started.set()
            assert release.wait(5)
            return {'results': []}

    client = Client()
    monkeypatch.setattr(operations, 'get_mem0_oss', lambda: client)

    def run():
        close_old_connections()
        try:
            create_conversation_memories_task.run(**payload)
        finally:
            close_old_connections()

    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(run)
        try:
            assert started.wait(5)
            # Waiting on the actual worker distinguishes retained ownership
            # from a timeout helper that leaves an unowned provider thread.
            from concurrent.futures import TimeoutError
            with pytest.raises(TimeoutError):
                first.result(timeout=0.1)
            ConversationMemoryJob.objects.filter(pk=convo.pk).update(
                next_attempt_at=timezone.now() - timedelta(seconds=1),
                lease_expires_at=timezone.now() - timedelta(seconds=1),
            )
            with patch.object(create_conversation_memories_task, 'apply_async') as publish:
                recover_conversation_memory_jobs.run()
            publish.assert_not_called()
            assert client.calls == 1
        finally:
            release.set()
        first.result(timeout=5)
    assert ConversationMemoryJob.objects.get(conversation=convo).state == 'idle'


def test_partial_dual_write_retries_unfinished_turn_without_replaying_completed_turn(source, monkeypatch):
    convo, payload = source
    monkeypatch.setattr(memory, 'MEM0_DUAL_WRITE_LOCAL', True)
    monkeypatch.setenv('MEM0_GRAPH_ENABLED', '0')
    first = convo.db_messages.get(role='assistant')
    Message.objects.create(conversation=convo, role='user', content='Also remember the camera.', sequence_number=2)
    second = Message.objects.create(conversation=convo, role='assistant', content='Camera noted.', sequence_number=3)

    class Client:
        failed = True
        seen = []

        def add(self, *args, **kwargs):
            identity = kwargs['metadata']['assistant_message_uuid']
            self.seen.append(identity)
            if self.failed and identity == str(second.message_uuid):
                raise OSError('second turn failed')
            return {'results': [{'event': 'ADD'}]}

    client = Client()
    monkeypatch.setattr(operations, 'get_mem0_oss', lambda: client)
    with pytest.raises(OSError):
        create_conversation_memories_task.run(**payload)
    assert list(EpisodicMemory.objects.values_list('assistant_message_uuid', flat=True)) == [first.message_uuid]
    client.failed = False
    ConversationMemoryJob.objects.filter(pk=convo.pk).update(next_attempt_at=timezone.now() - timedelta(seconds=1))
    with patch.object(create_conversation_memories_task, 'apply_async') as publish:
        recover_conversation_memory_jobs.run()
    create_conversation_memories_task.run(**publish.call_args.kwargs['kwargs'])
    assert client.seen == [str(first.message_uuid), str(second.message_uuid), str(second.message_uuid)]
    assert EpisodicMemory.objects.filter(conversation=convo).count() == 2
    assert ConversationMemoryJob.objects.get(conversation=convo).state == 'idle'
