"""Bounded provider reconciliation and erased-copy recall controls."""
from hashlib import sha256
from unittest.mock import patch
import pytest
from apps.chat.models import Message
from apps.memory.models import UserMemoryFact
from aquillm import memory
from .test_profile_memory_lifecycle import source, promote, lifecycle

pytestmark = pytest.mark.django_db(transaction=True)


def test_provider_receipt_is_reconciled_once_and_filtered_before_injection(source, monkeypatch):
    from apps.memory.models import ProfileMemoryReconciliationJob
    from aquillm.tasks import reconcile_profile_memory_task
    from lib.memory.mem0 import operations
    from lib.memory.mem0.search_parsing import parse_mem0_search_items
    service = lifecycle()
    fact = UserMemoryFact.objects.create(user=source[0], fact='I prefer telescope A.', origin='explicit_request',
        source_conversation_id=source[1].pk, source_message_uuid=source[2].message_uuid,
        source_content_digest=sha256(source[2].content.encode()).hexdigest(), provider_memory_ids=['known-id'], provider_receipt_complete=True, reconciliation_status='linked')
    deleted = []
    class Provider:
        def get(self, identity):
            return {'id': identity, 'metadata': {'user_id': str(source[0].pk), 'conversation_id': source[1].pk,
                'source_message_uuid': str(source[2].message_uuid), 'source_content_digest': sha256(source[2].content.encode()).hexdigest()}}
        def delete(self, identity):
            deleted.append(identity)
            return {'message': 'deleted'}
    monkeypatch.setattr(operations, 'get_mem0_oss', lambda: Provider())
    with patch.object(reconcile_profile_memory_task, 'delay'):
        service.revoke_profile_fact(user=source[0], fact_id=fact.pk, expected_revision=1)
    fact.refresh_from_db()
    assert fact.reconciliation_status == 'pending'
    items = parse_mem0_search_items([
        {'id': 'known-id', 'memory': 'Remote paraphrase', 'metadata': {'user_id': str(source[0].pk)}},
        {'id': 'other-id', 'memory': 'Other safe context', 'metadata': {'user_id': str(source[0].pk)}},
    ], top_k=4, exclude_conversation_id=None, user_id=str(source[0].pk))
    assert service.filter_reconciled_memories(source[0], items) == [items[1]]
    monkeypatch.setattr(memory, 'use_mem0', lambda: True)
    monkeypatch.setattr(memory, 'search_mem0_episodic_memories', lambda **kwargs: items)
    assert memory.get_episodic_memories(source[0], 'My telescope preference?') == [items[1]]
    reconcile_profile_memory_task.run(fact_id=fact.pk)
    reconcile_profile_memory_task.run(fact_id=fact.pk)
    fact.refresh_from_db()
    job = ProfileMemoryReconciliationJob.objects.get(fact=fact)
    assert deleted == ['known-id'] and fact.reconciliation_status == 'complete'
    assert job.attempts == 1 and job.state == 'complete'


@pytest.mark.parametrize('failure', ['timeout', 'unsupported', 'foreign', 'missing_identity'])
def test_reconciliation_failure_never_claims_external_deletion(source, monkeypatch, failure):
    from apps.memory.models import ProfileMemoryReconciliationJob
    from aquillm.tasks import reconcile_profile_memory_task
    from lib.memory.mem0 import operations
    fact = UserMemoryFact.objects.create(user=source[0], fact='I prefer telescope A.',
        provider_memory_ids=[] if failure == 'missing_identity' else ['known-id'], reconciliation_status='linked')
    deleted = []
    class Provider:
        def get(self, identity):
            if failure == 'timeout':
                raise TimeoutError()
            return {'id': identity, 'metadata': {'user_id': 'other' if failure == 'foreign' else str(source[0].pk)}}
        def delete(self, identity):
            if failure == 'unsupported':
                raise NotImplementedError()
            deleted.append(identity)
    monkeypatch.setattr(operations, 'get_mem0_oss', lambda: Provider())
    with patch.object(reconcile_profile_memory_task, 'delay'):
        lifecycle().revoke_profile_fact(user=source[0], fact_id=fact.pk, expected_revision=1)
    reconcile_profile_memory_task.run(fact_id=fact.pk)
    fact.refresh_from_db()
    job = ProfileMemoryReconciliationJob.objects.get(fact=fact)
    assert fact.fact == '' and deleted == []
    assert fact.reconciliation_status in {'pending', 'failed'}
    assert job.state in {'pending', 'failed'} and job.attempts == 1


def test_known_source_provider_copy_is_filtered_even_before_profile_lookup(source, monkeypatch):
    from lib.memory.types import RetrievedEpisodicMemory
    monkeypatch.setattr(memory, 'extract_stable_facts', lambda *args: [])
    promote(source)
    item = RetrievedEpisodicMemory('Remote paraphrase', conversation_id=source[1].pk,
        source_message_uuid=str(source[2].message_uuid), source_content_digest=sha256(source[2].content.encode()).hexdigest())
    Message.objects.filter(pk=source[2].pk).update(content='Edited origin.')
    assert lifecycle().filter_reconciled_memories(source[0], [item]) == []
    assert UserMemoryFact.objects.get().fact == ''


def test_known_local_episode_cannot_reintroduce_revoked_profile_source(source, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(memory, 'extract_stable_facts', lambda *args: [])
    monkeypatch.setattr(memory, 'use_mem0', lambda: False)
    promote(source)
    fact = UserMemoryFact.objects.get()
    lifecycle().revoke_profile_fact(user=source[0], fact_id=fact.pk, expected_revision=1)
    item = SimpleNamespace(content='User: Remember that I prefer telescope A.\nAssistant: Noted.', conversation_id=source[1].pk, assistant_message_uuid='local-identity')
    monkeypatch.setattr(memory, '_get_episodic_memories_pgvector', lambda *args, **kwargs: [item])
    assert memory.get_episodic_memories(source[0], 'My telescope preference?') == []


@pytest.mark.parametrize('provider_identity', ['unknown-id', None])
@pytest.mark.parametrize('origin', ['legacy_unknown', 'explicit_request'])
def test_unknown_legacy_provider_copies_are_not_reported_as_deleted_or_reinjected(source, provider_identity, origin):
    from aquillm.tasks import reconcile_profile_memory_task
    from lib.memory.types import RetrievedEpisodicMemory
    provenance = {'source_conversation_id': source[1].pk, 'source_message_uuid': source[2].message_uuid,
        'source_content_digest': sha256(source[2].content.encode()).hexdigest()} if origin == 'explicit_request' else {}
    old = UserMemoryFact.objects.create(user=source[0], fact='I prefer telescope A.', origin=origin,
        reconciliation_status='pending' if provenance else 'not_needed', **provenance)
    with patch.object(reconcile_profile_memory_task, 'delay'):
        lifecycle().revoke_profile_fact(user=source[0], fact_id=old.pk, expected_revision=1)
    old.refresh_from_db()
    assert old.reconciliation_status == 'pending'
    assert lifecycle().filter_reconciled_memories(source[0], [RetrievedEpisodicMemory('A paraphrase with unknown origin', provider_memory_id=provider_identity)]) == []
    reconcile_profile_memory_task.run(fact_id=old.pk)
    old.refresh_from_db()
    assert old.reconciliation_status == 'failed' and old.fact == ''


def test_partial_provider_timeout_retry_is_idempotent_and_bounded(source, monkeypatch):
    from apps.memory.models import ProfileMemoryReconciliationJob
    from django.utils import timezone
    from aquillm.tasks import reconcile_profile_memory_task
    from lib.memory.mem0 import operations
    fact = UserMemoryFact.objects.create(user=source[0], fact='I prefer telescope A.',
        provider_memory_ids=['first', 'second'], provider_receipt_complete=True, reconciliation_status='linked')
    deleted, unavailable = [], [True]
    class Provider:
        def get(self, identity):
            if identity in deleted:
                return None
            if identity == 'second' and unavailable[0]:
                raise TimeoutError()
            return {'id': identity, 'metadata': {'user_id': str(source[0].pk)}}
        def delete(self, identity):
            deleted.append(identity)
            return {'message': 'deleted'}
    monkeypatch.setattr(operations, 'get_mem0_oss', lambda: Provider())
    with patch.object(reconcile_profile_memory_task, 'delay'), patch.object(reconcile_profile_memory_task, 'apply_async'):
        lifecycle().revoke_profile_fact(user=source[0], fact_id=fact.pk, expected_revision=1)
        reconcile_profile_memory_task.run(fact_id=fact.pk)
        job = ProfileMemoryReconciliationJob.objects.get(fact=fact)
        assert job.state == 'pending' and job.attempts == 1 and job.last_error == 'provider_unavailable'
        assert deleted == ['first']
        unavailable[0] = False
        ProfileMemoryReconciliationJob.objects.filter(fact=fact).update(next_attempt_at=timezone.now())
        reconcile_profile_memory_task.run(fact_id=fact.pk)
        reconcile_profile_memory_task.run(fact_id=fact.pk)
    job.refresh_from_db()
    fact.refresh_from_db()
    assert deleted == ['first', 'second'] and job.attempts == 2
    assert job.state == fact.reconciliation_status == 'complete' and fact.fact == ''


def test_provider_timeout_stops_after_three_attempts_and_recovery_is_bounded(source, monkeypatch):
    from apps.memory.models import ProfileMemoryReconciliationJob
    from apps.memory.jobs import recover_profile_reconciliations
    from django.utils import timezone
    from aquillm.tasks import reconcile_profile_memory_task
    from lib.memory.mem0 import operations
    fact = UserMemoryFact.objects.create(user=source[0], fact='I prefer telescope A.',
        provider_memory_ids=['known'], reconciliation_status='linked')
    class Provider:
        def get(self, identity):
            raise TimeoutError()
        def delete(self, identity):
            raise AssertionError('unverified deletion')
    monkeypatch.setattr(operations, 'get_mem0_oss', lambda: Provider())
    with patch.object(reconcile_profile_memory_task, 'delay'), patch.object(reconcile_profile_memory_task, 'apply_async'):
        lifecycle().revoke_profile_fact(user=source[0], fact_id=fact.pk, expected_revision=1)
        for _ in range(4):
            ProfileMemoryReconciliationJob.objects.filter(fact=fact).update(next_attempt_at=timezone.now())
            reconcile_profile_memory_task.run(fact_id=fact.pk)
    job = ProfileMemoryReconciliationJob.objects.get(fact=fact)
    assert job.attempts == 3 and job.state == 'failed'
    for index in range(26):
        erased = UserMemoryFact.objects.create(user=source[0], fact='', status='revoked', origin='manual')
        ProfileMemoryReconciliationJob.objects.create(fact=erased)
    with patch.object(reconcile_profile_memory_task, 'delay') as dispatch:
        recover_profile_reconciliations()
    assert dispatch.call_count == 25
