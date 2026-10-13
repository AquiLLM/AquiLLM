"""External transports and real durable-job driver for P1 memory scenario 12."""
from copy import deepcopy
from datetime import timedelta
from hashlib import sha256
from types import SimpleNamespace

from asgiref.sync import async_to_sync
from django.utils import timezone

from apps.chat.models import Message, WSConversation
from apps.memory.jobs import request_memory_job
from apps.memory.models import ConversationMemoryJob
from aquillm import memory, tasks
from aquillm.llm import Conversation, OpenAIInterface, UserMessage
from lib.memory import config
from lib.memory.extraction import stable_facts
from lib.memory.mem0 import operations


class ExternalMemoryTransport:
    """Keep stale search copies even after acknowledged vector deletion."""

    def __init__(self):
        self.adds, self.searches, self.gets, self.deletes = [], [], [], []
        self.receipts, self.rows = {}, []

    def add(self, payload, **kwargs):
        self.adds.append((deepcopy(payload), deepcopy(kwargs)))
        identity = f'provider-copy-{len(self.adds)}'
        self.receipts[identity] = {'id': identity, 'metadata': deepcopy(kwargs['metadata'])}
        return {'results': [{'event': 'ADD', 'id': identity}]}

    def search(self, **kwargs):
        self.searches.append(deepcopy(kwargs))
        return {'results': deepcopy(self.rows)}

    def get(self, identity):
        self.gets.append(identity)
        return deepcopy(self.receipts[identity])

    def delete(self, identity):
        self.deletes.append(identity)
        return {'message': 'deleted'}


class BrokerTransport:
    """Capture broker envelopes, then deliver through the actual Celery tasks."""

    def __init__(self, monkeypatch):
        self.jobs, self.promotions, self.reconciliations = [], [], []
        monkeypatch.setattr(tasks.create_conversation_memories_task, 'apply_async',
                            lambda **kw: self.jobs.append(deepcopy(kw)))
        monkeypatch.setattr(tasks.promote_profile_facts_task, 'delay',
                            lambda **kw: self.promotions.append(deepcopy(kw)))
        monkeypatch.setattr(tasks.reconcile_profile_memory_task, 'delay',
                            lambda **kw: self.reconciliations.append(deepcopy(kw)))

    def infer(self, conversation):
        # Real persisted activity is old enough for the unchanged global-idle gate.
        WSConversation.objects.filter(owner=conversation.owner).update(
            updated_at=timezone.now() - timedelta(hours=1))
        before = len(self.jobs)
        request_memory_job(conversation.pk)
        assert len(self.jobs) == before + 1
        envelope = self.jobs[-1]
        job = ConversationMemoryJob.objects.get(conversation=conversation)
        assert job.state == 'queued' and str(job.token) == envelope['kwargs']['queued_job_token']
        assert envelope['countdown'] >= 30
        tasks.create_conversation_memories_task.run(**envelope['kwargs'])
        job.refresh_from_db()
        assert job.state == 'idle' and job.token is None and job.attempts == 1
        assert job.desired_transcript_hash == job.completed_transcript_hash != ''
        # A replayed broker envelope must not rerun provider work.
        tasks.create_conversation_memories_task.run(**envelope['kwargs'])
        return envelope

    def deliver_promotions(self):
        for kwargs in self.promotions:
            assert 'user_content' not in kwargs and 'assistant_content' not in kwargs
            tasks.promote_profile_facts_task.run(**kwargs)


def install_transports(monkeypatch):
    provider = ExternalMemoryTransport()
    monkeypatch.setattr(config, 'MEMORY_BACKEND', 'mem0')
    monkeypatch.setattr(memory, 'MEM0_DUAL_WRITE_LOCAL', False)
    monkeypatch.setenv('MEM0_GRAPH_ENABLED', '1')
    monkeypatch.setenv('MEM0_GRAPH_ADD_ENABLED', '1')
    # Declare the synthetic SDK model's capacity; unknown capacity correctly omits memory.
    monkeypatch.setenv('OPENAI_CONTEXT_LIMIT', '32768')
    monkeypatch.setattr(operations, 'get_mem0_oss', lambda: provider)
    monkeypatch.setattr(operations, 'get_mem0_oss_vector', lambda: provider)
    extraction_requests = []

    def post(url, **kwargs):
        extraction_requests.append(deepcopy(kwargs['json']))
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            'choices': [{'message': {'content': '{"facts":["Assistant invented preference."]}'}}]})

    monkeypatch.setattr(stable_facts.requests, 'post', post)
    return provider, BrokerTransport(monkeypatch), extraction_requests


def source_turn(owner, text):
    conversation = WSConversation.objects.create(owner=owner)
    source = Message.objects.create(conversation=conversation, role='user', content=text, sequence_number=0)
    Message.objects.create(conversation=conversation, role='assistant',
                           content='Assistant invented preference.', sequence_number=1)
    return conversation, source


def attributed_row(source, text, identity):
    return {'id': identity, 'memory': text, 'metadata': {
        'user_id': str(source.conversation.owner_id), 'conversation_id': source.conversation_id,
        'source_message_uuid': str(source.message_uuid),
        'source_content_digest': sha256(source.content.encode()).hexdigest()}}


class SubsequentPrompt:
    """Reuse the same conversation while actual augmentation rebuilds its system."""

    def __init__(self):
        self.requests = []
        self.conversation = Conversation(system='', messages=[UserMessage(content='What should I use?')])
        client = SimpleNamespace(chat=SimpleNamespace(completions=self), base_url='http://fake-models:8000/v1')
        self.llm = OpenAIInterface(client, model='qwen3.5:27b')

    async def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(tool_calls=[], content='Acknowledged.'), finish_reason='stop')],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2))

    def capture(self, owner):
        assert memory._remaining_memory_chars(self.conversation, 'Base system.') > 0
        memory.augment_conversation_with_memory(self.conversation, owner, 'Base system.')
        async_to_sync(self.llm.get_message)(system=self.conversation.system,
            messages=[{'role': 'user', 'content': self.conversation.messages[0].content}], max_tokens=128)
        request = self.requests[-1]
        assert request['messages'][0]['role'] == 'system'
        return request['messages'][0]['content']
