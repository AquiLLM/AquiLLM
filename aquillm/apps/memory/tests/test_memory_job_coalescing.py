from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from threading import Event
from unittest.mock import patch

import pytest
from django.apps import apps
from django.contrib.auth import get_user_model
from django.db import close_old_connections
from django.utils import timezone

from apps.chat.models import Message, WSConversation
from aquillm.tasks import create_conversation_memories_task, enqueue_conversation_memories_task

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.fixture
def source():
    convo = WSConversation.objects.create(owner=get_user_model().objects.create(username='memory-coalescing'))
    message = Message.objects.create(conversation=convo, role='assistant', content='Turn 0', sequence_number=0)
    return convo, message


def jobs():
    return apps.get_model('apps_memory', 'ConversationMemoryJob')


def test_twenty_turns_share_one_delivery_and_old_jobs_do_not_replay_inference(source):
    convo, message = source
    with patch.object(create_conversation_memories_task, 'apply_async') as publish:
        for index in range(20):
            message.content = f'Turn {index}'
            message.save(update_fields=['content'])
            enqueue_conversation_memories_task(convo.pk)
    assert publish.call_count == 1
    payload = publish.call_args.kwargs['kwargs']
    with patch('aquillm.tasks._user_is_globally_idle', return_value=True), patch(
        'aquillm.tasks._run_conversation_memory_creation'
    ) as infer:
        create_conversation_memories_task.run(**payload)
        for index in range(20):
            create_conversation_memories_task.run(convo.pk, queued_transcript_hash=f'old-{index}')
        create_conversation_memories_task.run(**payload)
    infer.assert_called_once()


def test_broker_failure_retains_latest_work_for_periodic_recovery(source):
    convo, message = source
    with patch.object(create_conversation_memories_task, 'apply_async', side_effect=ConnectionError('offline')):
        enqueue_conversation_memories_task(convo.pk)
    message.content = 'Newer answer'
    message.save(update_fields=['content'])
    enqueue_conversation_memories_task(convo.pk)
    jobs().objects.update(next_attempt_at=timezone.now() - timedelta(seconds=1))
    from aquillm.tasks import recover_conversation_memory_jobs
    with patch.object(create_conversation_memories_task, 'apply_async') as publish:
        recover_conversation_memory_jobs.run()
    with patch('aquillm.tasks._user_is_globally_idle', return_value=True), patch(
        'aquillm.tasks._run_conversation_memory_creation'
    ) as infer:
        create_conversation_memories_task.run(**publish.call_args.kwargs['kwargs'])
    infer.assert_called_once()
    job = jobs().objects.get(conversation=convo)
    assert job.completed_transcript_hash == job.desired_transcript_hash


def test_live_inference_excludes_second_claim_even_after_lease_expiry_and_retains_new_turn(source):
    convo, message = source
    with patch.object(create_conversation_memories_task, 'apply_async') as publish:
        enqueue_conversation_memories_task(convo.pk)
    payload = publish.call_args.kwargs['kwargs']
    started, release = Event(), Event()

    def infer(db_convo):
        started.set()
        assert release.wait(10)

    def execute():
        close_old_connections()
        try:
            create_conversation_memories_task.run(**payload)
        finally:
            close_old_connections()

    from aquillm.tasks import recover_conversation_memory_jobs
    with patch('aquillm.tasks._user_is_globally_idle', return_value=True), patch(
        'aquillm.tasks._run_conversation_memory_creation', side_effect=infer
    ) as inference, patch.object(create_conversation_memories_task, 'apply_async') as publish:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(execute)
            try:
                assert started.wait(10)
                jobs().objects.update(lease_expires_at=timezone.now() - timedelta(seconds=1), next_attempt_at=timezone.now() - timedelta(seconds=1))
                recover_conversation_memory_jobs.run()
                pool.submit(execute).result(timeout=5)
                message.content = 'New turn while inference runs'
                message.save(update_fields=['content'])
                enqueue_conversation_memories_task(convo.pk)
                publish.assert_not_called()
            finally:
                release.set()
            first.result(timeout=10)
    inference.assert_called_once()
    job = jobs().objects.get(conversation=convo)
    assert job.desired_transcript_hash != job.completed_transcript_hash
    assert job.state == 'pending'


def test_provider_failure_releases_owner_and_lost_worker_lease_recovers(source):
    convo, _ = source
    with patch.object(create_conversation_memories_task, 'apply_async') as publish:
        enqueue_conversation_memories_task(convo.pk)
    original = publish.call_args.kwargs['kwargs']
    with patch('aquillm.tasks._user_is_globally_idle', return_value=True), patch(
        'aquillm.tasks._run_conversation_memory_creation', side_effect=OSError('provider unavailable')
    ):
        with pytest.raises(OSError):
            create_conversation_memories_task.run(**original)
    job = jobs().objects.get(conversation=convo)
    assert job.state == 'pending' and job.last_error == 'OSError'
    # Simulate a later worker dying after claiming, with no live advisory owner.
    jobs().objects.update(state='running', next_attempt_at=timezone.now() - timedelta(seconds=1), lease_expires_at=timezone.now() - timedelta(seconds=1))
    from aquillm.tasks import recover_conversation_memory_jobs
    with patch.object(create_conversation_memories_task, 'apply_async') as publish:
        recover_conversation_memory_jobs.run()
    with patch('aquillm.tasks._user_is_globally_idle', return_value=True), patch(
        'aquillm.tasks._run_conversation_memory_creation'
    ) as infer:
        create_conversation_memories_task.run(**original)
        infer.assert_not_called()
        create_conversation_memories_task.run(**publish.call_args.kwargs['kwargs'])
    infer.assert_called_once()


def test_memory_recovery_is_registered_on_main_scheduler():
    from aquillm.celery_schedules import application_maintenance_schedule
    entry = application_maintenance_schedule(enabled=True)['conversation-memory-recovery']
    assert entry == {'task': 'aquillm.tasks.recover_conversation_memory_jobs', 'schedule': 60,
                     'kwargs': {'limit': 25}, 'options': {'queue': 'celery'}}
