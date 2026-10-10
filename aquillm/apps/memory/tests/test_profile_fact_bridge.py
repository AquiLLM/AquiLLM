"""Real source enqueue and mocked external Mem0 payload boundaries."""
from hashlib import sha256
from unittest.mock import patch
import pytest
from django.contrib.auth import get_user_model
from apps.chat.models import Message, WSConversation
from apps.memory.models import UserMemoryFact
from aquillm import memory, tasks
from lib.memory.mem0 import operations

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.fixture
def turn(monkeypatch):
    user = get_user_model().objects.create(username='bridge-owner')
    convo = WSConversation.objects.create(owner=user)
    message = Message.objects.create(conversation=convo, role='user', content='Remember that I prefer concise updates.', sequence_number=0)
    Message.objects.create(conversation=convo, role='assistant', content='Assistant invented preference: telescope C.', sequence_number=1)
    monkeypatch.setattr(memory, 'use_mem0', lambda: True)
    monkeypatch.setattr(memory, 'MEM0_DUAL_WRITE_LOCAL', False)
    return user, convo, message


def test_source_bound_enqueue_contains_no_raw_user_or_assistant_text(turn, monkeypatch):
    captured = []
    class Provider:
        def add(self, payload, **kwargs):
            captured.append((payload, kwargs))
            return {'results': []}
    monkeypatch.setattr(operations, 'get_mem0_oss', lambda: Provider())
    monkeypatch.setattr(memory, 'extract_stable_facts', lambda *args: [])
    with patch.object(tasks.promote_profile_facts_task, 'delay') as enqueue:
        memory.create_episodic_memories_for_conversation(turn[1])
    receipt = enqueue.call_args.kwargs
    assert set(receipt) == {'user_id', 'source_conversation_id', 'source_message_uuid', 'source_content_digest', 'provider_memory_ids', 'provider_wrote', 'provider_receipt_complete'}
    assert receipt['source_message_uuid'] == str(turn[2].message_uuid)
    assert receipt['source_content_digest'] == sha256(turn[2].content.encode()).hexdigest()
    assert not UserMemoryFact.objects.exists()
    assert captured[0][0] == [{'role': 'user', 'content': 'I prefer concise updates.'}]
    assert captured[0][1]['infer'] is True
    assert captured[0][1]['metadata']['source_message_uuid'] == str(turn[2].message_uuid)
    tasks.promote_profile_facts_task.run(**receipt)
    assert UserMemoryFact.objects.get().fact == 'I prefer concise updates.'


def test_delayed_worker_rereads_current_source_and_legacy_task_cannot_promote(turn, monkeypatch):
    monkeypatch.setattr(operations, 'get_mem0_oss', lambda: type('Provider', (), {'add': lambda *args, **kwargs: {'results': []}})())
    with patch.object(tasks.promote_profile_facts_task, 'delay') as enqueue:
        memory.create_episodic_memories_for_conversation(turn[1])
    Message.objects.filter(pk=turn[2].pk).update(content='Edited user statement.')
    tasks.promote_profile_facts_task.run(**enqueue.call_args.kwargs)
    tasks.promote_profile_facts_task.run(user_id=turn[0].pk, user_content='Remember that I prefer telescope C.', assistant_content='')
    assert not UserMemoryFact.objects.exists()


def test_tool_output_cannot_supply_profile_provenance_or_mem0_inference(turn, monkeypatch):
    Message.objects.filter(pk=turn[2].pk).update(role='tool')
    monkeypatch.setattr(operations, 'get_mem0_oss', lambda: (_ for _ in ()).throw(AssertionError('tool-derived inference')))
    with patch.object(tasks.promote_profile_facts_task, 'delay') as enqueue:
        memory.create_episodic_memories_for_conversation(turn[1])
    enqueue.assert_not_called()
    assert not UserMemoryFact.objects.exists()


@pytest.mark.parametrize('user_content', ['Do not remember that I prefer concise updates.', 'The paper says "Remember that concise updates are best."', 'If I ask you to remember that I prefer concise updates, what happens?'])
def test_direct_mem0_wrapper_also_rejects_excluded_user_directives(monkeypatch, user_content):
    monkeypatch.setattr(operations, 'get_mem0_oss', lambda: (_ for _ in ()).throw(AssertionError('excluded inference')))
    assert operations.add_mem0_memory_with_client('1', user_content, 'Assistant inferred preference', 1, 'a', strict=True) is False


def test_direct_messages_use_only_proven_affirmative_user_payload(monkeypatch):
    captured = []
    class Provider:
        def add(self, payload, **kwargs):
            captured.append(payload)
            return {'results': []}
    monkeypatch.setattr(operations, 'get_mem0_oss', lambda: Provider())
    operations.add_mem0_messages('1', [
        {'role': 'user', 'content': 'Do not remember that I prefer telescope A. Please remember that I prefer telescope B.'},
        {'role': 'assistant', 'content': 'I prefer telescope C.'},
    ], 1, 'a', strict=True)
    assert captured == [[{'role': 'user', 'content': 'I prefer telescope B.'}]]


def test_ordinary_turn_preserves_episodic_inference_without_local_profile_promotion(turn, monkeypatch):
    captured = []
    class Provider:
        def add(self, payload, **kwargs):
            captured.append((payload, kwargs))
            return {'results': [{'id': 'episode-only', 'event': 'ADD'}]}
    monkeypatch.setattr(operations, 'get_mem0_oss', lambda: Provider())
    Message.objects.filter(pk=turn[2].pk).update(content='We compared telescope lenses today.')
    with patch.object(tasks.promote_profile_facts_task, 'delay') as enqueue:
        memory.create_episodic_memories_for_conversation(turn[1])
    assert captured[0][0] == [
        {'role': 'user', 'content': 'We compared telescope lenses today.'},
        {'role': 'assistant', 'content': 'Assistant invented preference: telescope C.'},
    ]
    assert captured[0][1]['infer'] is True
    enqueue.assert_not_called()
    assert not UserMemoryFact.objects.exists()


def test_excluded_remember_turn_keeps_existing_local_episodic_retention(turn, monkeypatch):
    from apps.memory.models import EpisodicMemory
    monkeypatch.setattr(memory, 'use_mem0', lambda: False)
    Message.objects.filter(pk=turn[2].pk).update(content='Do not remember that I prefer telescope A.')
    with patch.object(tasks.promote_profile_facts_task, 'delay') as enqueue:
        memory.create_episodic_memories_for_conversation(turn[1])
    episode = EpisodicMemory.objects.get()
    assert episode.user_id == turn[0].pk and episode.conversation_id == turn[1].pk
    assert 'Do not remember that I prefer telescope A.' in episode.content
    enqueue.assert_not_called()
    assert not UserMemoryFact.objects.exists()
