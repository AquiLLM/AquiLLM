"""Provider receipt completeness is independent of deleting an identified subset."""
from unittest.mock import patch
import pytest
from apps.chat.models import Message
from apps.memory.models import UserMemoryFact
from aquillm import memory, tasks
from lib.memory.mem0 import operations
from .test_profile_memory_lifecycle import source, lifecycle

pytestmark = pytest.mark.django_db(transaction=True)


def test_mixed_receipt_deletes_known_subset_without_confirming_unknown_coverage(source, monkeypatch):
    deleted, metadata = [], {}
    class Provider:
        def add(self, payload, **kwargs):
            metadata.update(kwargs['metadata'])
            return {'results': [{'id': 'known', 'event': 'ADD'}, {'event': 'ADD'}]}
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
        lifecycle().revoke_profile_fact(user=source[0], fact_id=fact.pk, expected_revision=1)
        tasks.reconcile_profile_memory_task.run(fact_id=fact.pk)
    fact.refresh_from_db()
    assert deleted == ['known'] and fact.fact == ''
    assert fact.reconciliation_status == 'failed'
    assert fact.provider_receipt_complete is False
    from lib.memory.types import RetrievedEpisodicMemory
    assert lifecycle().filter_reconciled_memories(source[0], [RetrievedEpisodicMemory('Unidentified paraphrase', provider_memory_id='unknown')]) == []


def test_legacy_default_is_unknown_but_no_write_and_manual_correction_are_local_only(source, monkeypatch):
    from .test_profile_memory_lifecycle import promote
    legacy = UserMemoryFact.objects.create(user=source[0], fact='I prefer legacy optics.')
    assert legacy.provider_receipt_complete is False and legacy.origin == 'legacy_unknown'
    monkeypatch.setattr(memory, 'extract_stable_facts', lambda *args: [])
    promote(source)
    fact = UserMemoryFact.objects.get(origin='explicit_request')
    assert fact.provider_receipt_complete is True and fact.reconciliation_status == 'not_needed'
    new = lifecycle().correct_profile_fact(user=source[0], fact_id=fact.pk, replacement='I prefer telescope B.', expected_revision=1)
    assert new.provider_receipt_complete is True and new.reconciliation_status == 'not_needed'
    lifecycle().revoke_profile_fact(user=source[0], fact_id=new.pk, expected_revision=1)
    new.refresh_from_db()
    assert new.reconciliation_status == 'not_needed'


@pytest.mark.parametrize('result', [None, {}, {'results': None}, {'results': [None]},
    {'results': [{'event': 'ADD'}]}, {'results': [{'event': 'UPDATE', 'id': ''}]},
    {'results': [{'event': 'ADD', 'id': 'x' * 257}]},
    {'results': [{'event': 'ADD', 'id': 'known'}, {'event': []}]},
    {'results': [{'event': 'ADD', 'id': 'known'}, {'event': {}}]},
    {'results': [{'event': 'ADD', 'id': 'known'}, {'event': 'unknown'}]}])
def test_malformed_receipts_never_claim_complete_coverage(result):
    from lib.memory.mem0._receipts import capture_provider_receipt
    identities, complete = capture_provider_receipt(result)
    assert complete is False
    if isinstance(result, dict) and isinstance(result.get('results'), list) and result['results'] and result['results'][0] == {'event': 'ADD', 'id': 'known'}:
        assert identities == ['known']


def test_receipt_bounds_retain_identified_subset_without_losing_unknown_coverage():
    from lib.memory.mem0._receipts import capture_provider_receipt
    ids, complete = capture_provider_receipt({'results': [{'event': 'ADD', 'id': str(i)} for i in range(51)]})
    assert ids == [str(i) for i in range(50)] and complete is False
    assert capture_provider_receipt({'results': []}) == ([], True)
    assert capture_provider_receipt({'results': [{'event': 'NONE'}]}) == ([], True)
    assert capture_provider_receipt({'results': [{'event': 'ADD', 'id': 'known'}]}) == (['known'], True)


@pytest.mark.parametrize('events', [[], [{'event': 'NONE'}]])
def test_valid_no_write_results_do_not_fabricate_external_reconciliation(source, monkeypatch, events):
    class Provider:
        def add(self, *args, **kwargs):
            return {'results': events}
    monkeypatch.setattr(operations, 'get_mem0_oss', lambda: Provider())
    monkeypatch.setattr(memory, 'use_mem0', lambda: True)
    monkeypatch.setattr(memory, 'MEM0_DUAL_WRITE_LOCAL', False)
    Message.objects.create(conversation=source[1], role='assistant', content='Noted.', sequence_number=1)
    with patch.object(tasks.promote_profile_facts_task, 'delay') as enqueue:
        memory.create_episodic_memories_for_conversation(source[1])
    tasks.promote_profile_facts_task.run(**enqueue.call_args.kwargs)
    fact = UserMemoryFact.objects.get()
    assert fact.provider_receipt_complete is True and fact.reconciliation_status == 'not_needed'
    lifecycle().revoke_profile_fact(user=source[0], fact_id=fact.pk, expected_revision=1)
    fact.refresh_from_db()
    assert fact.reconciliation_status == 'not_needed'
