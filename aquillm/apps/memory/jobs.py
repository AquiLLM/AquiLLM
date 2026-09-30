"""Durable coalescing and session ownership for conversation memory inference."""
from contextlib import contextmanager
from datetime import timedelta
import hashlib
import uuid

from django.db import connections, transaction
from django.db.models import F
from django.utils import timezone

from apps.chat.models import WSConversation
from apps.memory.models.jobs import ConversationMemoryJob

LEASE_SECONDS = 900


@contextmanager
def _execution_lock(conversation_id):
    # Separate session: no database transaction stays open during provider work.
    # A live holder excludes recovery even if its row lease has expired.
    key = int.from_bytes(hashlib.sha256(f'memory-job:{conversation_id}'.encode()).digest()[:8], 'big', signed=True)
    # Keep the configured alias for Django's PostgreSQL type-registration signal;
    # this wrapper still owns a separate physical connection.
    connection = connections['default'].copy(alias='default')
    acquired = False
    try:
        with connection.cursor() as cursor:
            cursor.execute('SELECT pg_try_advisory_lock(%s)', [key])
            acquired = cursor.fetchone()[0]
        yield acquired
    finally:
        try:
            if acquired:
                with connection.cursor() as cursor:
                    cursor.execute('SELECT pg_advisory_unlock(%s)', [key])
        finally:
            connection.close()


def request_memory_job(conversation_id, *, publish=True):
    from aquillm.tasks import _memory_transcript_hash, _mem0_infer_idle_seconds

    with transaction.atomic():
        convo = WSConversation.objects.filter(pk=conversation_id).first()
        if convo is None:
            return
        ConversationMemoryJob.objects.get_or_create(conversation=convo)
        job = ConversationMemoryJob.objects.select_for_update().get(conversation=convo)
        job.desired_transcript_hash = _memory_transcript_hash(convo)
        if job.state == 'idle' and job.completed_transcript_hash != job.desired_transcript_hash:
            job.state = 'pending'
            job.next_attempt_at = timezone.now()
        job.save(update_fields=['desired_transcript_hash', 'state', 'next_attempt_at'])
        if publish:
            transaction.on_commit(
                lambda: dispatch_memory_job(conversation_id, countdown=_mem0_infer_idle_seconds()), robust=True,
            )


def _failed_attempt(conversation_id, token, state, exc):
    job = ConversationMemoryJob.objects.filter(conversation_id=conversation_id, token=token, state=state).first()
    if job is not None:
        ConversationMemoryJob.objects.filter(pk=job.pk, token=token, state=state).update(
            state='pending', token=None, lease_expires_at=None,
            next_attempt_at=timezone.now() + timedelta(seconds=min(900, 30 * 2 ** min(job.attempts, 5))),
            last_error=type(exc).__name__[:128],
        )


def dispatch_memory_job(conversation_id, *, countdown=0):
    from aquillm.tasks import create_conversation_memories_task

    with _execution_lock(conversation_id) as acquired:
        if not acquired:
            return False
        with transaction.atomic():
            job = ConversationMemoryJob.objects.select_for_update().filter(pk=conversation_id).first()
            now = timezone.now()
            if job is None or job.desired_transcript_hash == job.completed_transcript_hash:
                return False
            if job.next_attempt_at > now:
                return False
            token = uuid.uuid4()
            job.state, job.token = 'queued', token
            job.attempts += 1
            job.lease_expires_at = now + timedelta(seconds=LEASE_SECONDS + countdown)
            job.next_attempt_at, job.last_error = job.lease_expires_at, ''
            job.save()
    try:
        create_conversation_memories_task.apply_async(
            kwargs={'conversation_id': conversation_id, 'queued_job_token': str(token)}, countdown=countdown,
        )
        return True
    except Exception as exc:
        _failed_attempt(conversation_id, token, 'queued', exc)
        return False


def execute_memory_job(conversation_id, queued_job_token=None):
    from aquillm.tasks import (
        _memory_transcript_hash, _mem0_infer_idle_seconds, _run_conversation_memory_creation,
        _user_is_globally_idle, _utcnow,
    )

    if queued_job_token is None:
        # In-flight legacy timestamp/hash jobs join the durable intent. Their
        # obsolete payload never authorizes another inference after completion.
        request_memory_job(conversation_id, publish=False)
    with _execution_lock(conversation_id) as acquired:
        if not acquired:
            return
        with transaction.atomic():
            job = ConversationMemoryJob.objects.select_for_update().filter(pk=conversation_id).first()
            convo = WSConversation.objects.filter(pk=conversation_id).first()
            if job is None or convo is None:
                return
            if queued_job_token is not None:
                if job.state != 'queued' or str(job.token) != str(queued_job_token):
                    return
            elif job.state in {'queued', 'running'} or job.next_attempt_at > timezone.now():
                return
            source_hash = _memory_transcript_hash(convo)
            job.desired_transcript_hash = source_hash
            if source_hash == job.completed_transcript_hash:
                job.state, job.token = 'idle', None
                job.save()
                return
            if not _user_is_globally_idle(convo.owner_id, now=_utcnow(), idle_seconds=_mem0_infer_idle_seconds()):
                job.state, job.token, job.lease_expires_at = 'pending', None, None
                job.next_attempt_at = timezone.now() + timedelta(seconds=_mem0_infer_idle_seconds())
                job.save()
                return
            token = job.token or uuid.uuid4()
            job.state, job.token = 'running', token
            job.lease_expires_at = timezone.now() + timedelta(seconds=LEASE_SECONDS)
            job.next_attempt_at = job.lease_expires_at
            job.save()
        try:
            _run_conversation_memory_creation(convo)
        except Exception as exc:
            _failed_attempt(conversation_id, token, 'running', exc)
            raise
        with transaction.atomic():
            job = ConversationMemoryJob.objects.select_for_update().filter(pk=conversation_id, token=token, state='running').first()
            if job is None:
                return
            job.completed_transcript_hash = source_hash
            job.desired_transcript_hash = _memory_transcript_hash(convo)
            job.state = 'idle' if job.completed_transcript_hash == job.desired_transcript_hash else 'pending'
            job.token, job.lease_expires_at, job.last_error = None, None, ''
            job.next_attempt_at = timezone.now() + timedelta(seconds=_mem0_infer_idle_seconds())
            job.save()


def recover_memory_jobs(*, limit=25):
    due = list(ConversationMemoryJob.objects.exclude(desired_transcript_hash=F('completed_transcript_hash'))
               .filter(next_attempt_at__lte=timezone.now()).order_by('next_attempt_at', 'pk')
               .values_list('pk', flat=True)[:min(100, max(1, int(limit)))])
    return sum(dispatch_memory_job(pk) for pk in due)
