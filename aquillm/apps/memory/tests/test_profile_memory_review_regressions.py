"""Review counterexamples at actual consent, prompt and durable provider boundaries."""
from datetime import timedelta
from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest
from django.utils import timezone
from apps.chat.models import Message
from apps.memory.models import UserMemoryFact, ProfileMemoryReconciliationJob
from aquillm import memory, tasks
from lib.memory.mem0 import operations
from lib.memory.mem0.search_parsing import parse_mem0_search_items
from .test_profile_memory_lifecycle import source, promote, lifecycle

pytestmark = pytest.mark.django_db(transaction=True)
HYPOTHETICAL = 'If I wrote the following, what would happen?\nRemember that I prefer telescope A.'


def test_multiline_hypothetical_never_reaches_provider_or_local_promotion(source, monkeypatch):
    source[2].content = HYPOTHETICAL
    source[2].save()
    Message.objects.create(conversation=source[1], role='assistant', content='Noted.', sequence_number=1)
    monkeypatch.setattr(memory, 'use_mem0', lambda: True)
    monkeypatch.setattr(memory, 'MEM0_DUAL_WRITE_LOCAL', False)
    monkeypatch.setattr(memory, 'extract_stable_facts', lambda *args: ['I prefer telescope A.'])
    captured = []
    class Provider:
        def add(self, payload, **kwargs):
            captured.append(payload)
            return {'results': []}
    monkeypatch.setattr(operations, 'get_mem0_oss', lambda: Provider())
    with patch.object(tasks.promote_profile_facts_task, 'delay'):
        memory.create_episodic_memories_for_conversation(source[1])
    promoted = promote(source)
    assert (captured, promoted, UserMemoryFact.objects.exists()) == ([], 0, False)


def test_proven_affirmative_before_hypothetical_keeps_only_its_payload(source, monkeypatch):
    source[2].content = 'Please remember that I prefer telescope B.\n' + HYPOTHETICAL
    source[2].save()
    Message.objects.create(conversation=source[1], role='assistant', content='Noted.', sequence_number=1)
    monkeypatch.setattr(memory, 'use_mem0', lambda: True)
    monkeypatch.setattr(memory, 'MEM0_DUAL_WRITE_LOCAL', False)
    monkeypatch.setattr(memory, 'extract_stable_facts', lambda *args: [])
    captured = []
    class Provider:
        def add(self, payload, **kwargs):
            captured.append(payload)
            return {'results': []}
    monkeypatch.setattr(operations, 'get_mem0_oss', lambda: Provider())
    with patch.object(tasks.promote_profile_facts_task, 'delay') as enqueue:
        memory.create_episodic_memories_for_conversation(source[1])
    assert enqueue.call_args is not None
    tasks.promote_profile_facts_task.run(**enqueue.call_args.kwargs)
    assert captured == [[{'role': 'user', 'content': 'I prefer telescope B.'}]]
    assert UserMemoryFact.objects.get().fact == 'I prefer telescope B.'


@pytest.mark.parametrize('provenance', ['missing_digest', 'malformed_uuid', 'malformed_digest', 'unproven_revision', 'uppercase_uuid'])
def test_partial_provider_provenance_cannot_reintroduce_erased_copy_into_prompt(source, monkeypatch, provenance):
    digest = sha256(source[2].content.encode()).hexdigest()
    old = UserMemoryFact.objects.create(user=source[0], fact='I prefer telescope A.', origin='explicit_request',
        source_conversation_id=source[1].pk, source_message_uuid=source[2].message_uuid,
        source_content_digest=digest, reconciliation_status='pending', provider_memory_ids=['known'])
    with patch.object(tasks.reconcile_profile_memory_task, 'delay'):
        lifecycle().revoke_profile_fact(user=source[0], fact_id=old.pk, expected_revision=1)
    metadata = {'user_id': str(source[0].pk), 'conversation_id': source[1].pk,
        'source_message_uuid': str(source[2].message_uuid), 'source_content_digest': digest}
    if provenance == 'missing_digest':
        metadata.pop('source_content_digest')
    elif provenance == 'malformed_uuid':
        metadata['source_message_uuid'] = 'not-a-uuid'
    elif provenance == 'malformed_digest':
        metadata['source_content_digest'] = 'g' * 64
    elif provenance == 'unproven_revision':
        metadata['source_content_digest'] = 'f' * 64
    else:
        metadata['source_message_uuid'] = str(source[2].message_uuid).upper()
    independent = Message.objects.create(conversation=source[1], role='user', content='Independent context', sequence_number=2)
    parsed = parse_mem0_search_items([
        {'id': 'unknown', 'memory': 'Discarded preference paraphrase', 'metadata': metadata},
        {'id': 'independent', 'memory': 'Safe independent context', 'metadata': {
            'user_id': str(source[0].pk), 'conversation_id': source[1].pk,
            'source_message_uuid': str(independent.message_uuid),
            'source_content_digest': sha256(independent.content.encode()).hexdigest()}},
    ], top_k=5, exclude_conversation_id=None, user_id=str(source[0].pk))
    monkeypatch.setattr(memory, 'use_mem0', lambda: True)
    monkeypatch.setattr(memory, 'search_mem0_episodic_memories', lambda **kwargs: parsed)
    monkeypatch.setattr(memory, 'get_last_user_message_text', lambda convo: 'Telescope preference?')
    monkeypatch.setattr(memory, '_remaining_memory_chars', lambda *args: 4096)
    convo = SimpleNamespace(system='', messages=[])
    memory.augment_conversation_with_memory(convo, source[0], 'Base system')
    assert 'Discarded preference paraphrase' not in convo.system
    assert 'Safe independent context' in convo.system


@pytest.mark.parametrize('malformed', [[], {}])
def test_malformed_event_before_add_keeps_durable_id_and_honest_erasure(source, monkeypatch, malformed):
    deleted, metadata = [], {}
    class Provider:
        def add(self, payload, **kwargs):
            metadata.update(kwargs['metadata'])
            return {'results': [{'event': malformed}, {'event': 'ADD', 'id': 'known'}]}
        def get(self, identity):
            return {'id': identity, 'metadata': metadata}
        def delete(self, identity):
            deleted.append(identity)
            return {'message': 'deleted'}
    monkeypatch.setattr(operations, 'get_mem0_oss', lambda: Provider())
    monkeypatch.setattr(memory, 'use_mem0', lambda: True)
    monkeypatch.setattr(memory, 'MEM0_DUAL_WRITE_LOCAL', False)
    Message.objects.create(conversation=source[1], role='assistant', content='Noted.', sequence_number=1)
    with patch.object(tasks.promote_profile_facts_task, 'delay'), patch.object(tasks.reconcile_profile_memory_task, 'delay'):
        memory.create_episodic_memories_for_conversation(source[1])
        fact = UserMemoryFact.objects.get()
        assert fact.provider_memory_ids == ['known'] and not fact.provider_receipt_complete
        lifecycle().revoke_profile_fact(user=source[0], fact_id=fact.pk, expected_revision=1)
        tasks.reconcile_profile_memory_task.run(fact_id=fact.pk)
    fact.refresh_from_db()
    assert deleted == ['known'] and fact.fact == '' and fact.reconciliation_status == 'failed'
    parsed = parse_mem0_search_items([{'id': 'unknown', 'memory': 'Unattributed paraphrase',
        'metadata': {'user_id': str(source[0].pk)}}], 5, None, str(source[0].pk))
    assert lifecycle().filter_reconciled_memories(source[0], parsed) == []


@pytest.mark.parametrize('graph_mode', ['disabled', 'enabled', 'fallback'])
def test_graph_attempt_cannot_claim_vector_receipt_covers_graph_erasure(source, monkeypatch, graph_mode):
    deleted, metadata, attempts = [], {}, []
    monkeypatch.setenv('MEM0_GRAPH_ENABLED', '0' if graph_mode == 'disabled' else '1')
    monkeypatch.setenv('MEM0_GRAPH_ADD_ENABLED', '1')
    monkeypatch.setenv('MEM0_GRAPH_FAIL_OPEN', '1')
    class Provider:
        def add(self, payload, **kwargs):
            attempts.append(kwargs['enable_graph'])
            metadata.update(kwargs['metadata'])
            if graph_mode == 'fallback' and kwargs['enable_graph']:
                raise RuntimeError('graph failed after possible partial write')
            return {'results': [{'event': 'ADD', 'id': 'known'}]}
        def get(self, identity):
            return {'id': identity, 'metadata': metadata}
        def delete(self, identity):
            deleted.append(identity)
            return {'message': 'deleted'}
    monkeypatch.setattr(operations, 'get_mem0_oss', lambda: Provider())
    monkeypatch.setattr(memory, 'use_mem0', lambda: True)
    monkeypatch.setattr(memory, 'MEM0_DUAL_WRITE_LOCAL', False)
    Message.objects.create(conversation=source[1], role='assistant', content='Noted.', sequence_number=1)
    with patch.object(tasks.promote_profile_facts_task, 'delay'), patch.object(tasks.reconcile_profile_memory_task, 'delay'):
        memory.create_episodic_memories_for_conversation(source[1])
        fact = UserMemoryFact.objects.get()
        assert fact.provider_memory_ids == ['known']
        assert fact.provider_receipt_complete is (graph_mode == 'disabled')
        lifecycle().revoke_profile_fact(user=source[0], fact_id=fact.pk, expected_revision=1)
        tasks.reconcile_profile_memory_task.run(fact_id=fact.pk)
    fact.refresh_from_db()
    assert attempts == ([True, False] if graph_mode == 'fallback' else [graph_mode == 'enabled'])
    assert deleted == ['known']
    assert fact.reconciliation_status == ('complete' if graph_mode == 'disabled' else 'failed')


def test_expired_third_claim_recovery_does_not_make_fourth_provider_attempt(source, monkeypatch):
    fact = UserMemoryFact.objects.create(user=source[0], fact='', status='revoked', origin='manual',
        provider_memory_ids=['known'], provider_receipt_complete=True, reconciliation_status='pending')
    job = ProfileMemoryReconciliationJob.objects.create(fact=fact, state='running', attempts=3,
        token=uuid4(), lease_expires_at=timezone.now() - timedelta(seconds=1), next_attempt_at=timezone.now())
    calls = []
    monkeypatch.setattr(operations, 'reconcile_known_profile_copies', lambda **kwargs: calls.append(kwargs))
    from apps.memory.jobs import recover_profile_reconciliations
    with patch.object(tasks.reconcile_profile_memory_task, 'delay') as dispatch:
        recover_profile_reconciliations()
        tasks.reconcile_profile_memory_task.run(**dispatch.call_args.kwargs)
    fact.refresh_from_db()
    job.refresh_from_db()
    assert calls == []
    assert job.attempts == 3 and job.state == fact.reconciliation_status == 'failed'
    assert job.token is None and job.lease_expires_at is None


def test_complete_live_revision_of_same_source_remains_eligible(source, monkeypatch):
    old_digest = sha256(source[2].content.encode()).hexdigest()
    old = UserMemoryFact.objects.create(user=source[0], fact='I prefer telescope A.', origin='explicit_request',
        source_conversation_id=source[1].pk, source_message_uuid=source[2].message_uuid,
        source_content_digest=old_digest, reconciliation_status='pending')
    with patch.object(tasks.reconcile_profile_memory_task, 'delay'):
        lifecycle().revoke_profile_fact(user=source[0], fact_id=old.pk, expected_revision=1)
    Message.objects.filter(pk=source[2].pk).update(content='Independent current revision')
    metadata = {'user_id': str(source[0].pk), 'conversation_id': source[1].pk,
        'source_message_uuid': str(source[2].message_uuid).upper(),
        'source_content_digest': sha256(b'Independent current revision').hexdigest().upper()}
    items = parse_mem0_search_items([{'id': 'independent', 'memory': 'Current independent context',
        'metadata': metadata}], 5, None, str(source[0].pk))
    assert lifecycle().filter_reconciled_memories(source[0], items) == items
