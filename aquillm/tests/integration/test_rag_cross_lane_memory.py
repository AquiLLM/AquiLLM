"""P1 scenario 12: consent, ownership, corrections and revocation to SDK prompts."""
import json
from hashlib import sha256

import pytest
from django.contrib.auth import get_user_model
from django.test import Client

from apps.memory.models import ProfileMemoryReconciliationJob, UserMemoryFact
from aquillm import tasks
from .cross_lane_memory_support import (
    SubsequentPrompt, attributed_row, install_transports, source_turn,
)

pytestmark = pytest.mark.django_db(transaction=True)
URL = '/api/profile-memory/'
NEGATED = 'Do not remember that I prefer amber telescopes.'
ORIGINAL = 'I prefer cobalt telescopes.'
CORRECTED = 'I prefer jade telescopes.'
FOREIGN = 'I prefer violet telescopes.'
SAFE = 'Independent observatory hours are available.'
PARAPHRASES = ('Cobalt equipment suits this user.', 'Their choice is the blue instrument.')


def edit(client, method, fact_id, revision, **body):
    return getattr(client, method)(URL + str(fact_id) + '/', content_type='application/json',
                                  data=json.dumps({'expected_revision': revision, **body}))


def assert_absent(prompt, *values):
    for value in values:
        assert value not in prompt


def test_memory_consent_owner_correction_revocation_reaches_subsequent_sdk(monkeypatch):
    provider, broker, extraction = install_transports(monkeypatch)
    owner = get_user_model().objects.create_user(username='p1-memory-owner')
    other = get_user_model().objects.create_user(username='p1-memory-other')
    client, other_client = Client(), Client()
    client.force_login(owner)
    other_client.force_login(other)
    negative_conversation, negative_source = source_turn(owner, NEGATED)
    broker.infer(negative_conversation)
    assert provider.adds == broker.promotions == []
    assert not UserMemoryFact.objects.filter(user=owner).exists()

    # An independently owned affirmative source must work without leaking owners.
    foreign_conversation, foreign_source = source_turn(other, 'Please remember that ' + FOREIGN)
    broker.infer(foreign_conversation)
    broker.deliver_promotions()
    foreign_fact = UserMemoryFact.objects.get(user=other)
    assert foreign_fact.fact == FOREIGN
    assert len(provider.adds) == 1
    assert provider.adds[0][0] == [{'role': 'user', 'content': FOREIGN}]
    assert provider.adds[0][1]['metadata']['user_id'] == str(other.pk)
    assert client.get(URL).json()['facts'] == []
    assert edit(client, 'delete', foreign_fact.pk, 1, user_id=other.pk).status_code == 404

    _, independent_source = source_turn(owner, SAFE)
    provider.rows = [attributed_row(independent_source, SAFE, 'independent'),
                     attributed_row(foreign_source, FOREIGN, foreign_fact.provider_memory_ids[0])]
    prompt = SubsequentPrompt()
    initial = prompt.capture(owner)
    assert SAFE in initial
    assert_absent(initial, 'amber telescopes', FOREIGN, 'Assistant invented preference.')
    foreign_prompt = SubsequentPrompt().capture(other)
    assert FOREIGN in foreign_prompt
    assert_absent(foreign_prompt, SAFE, 'amber telescopes')

    affirmative_conversation, affirmative_source = source_turn(owner, 'Please remember that ' + ORIGINAL)
    broker.infer(affirmative_conversation)
    broker.deliver_promotions()
    original = UserMemoryFact.objects.get(user=owner)
    assert original.fact == ORIGINAL and original.status == 'active'
    assert original.source_conversation_id == affirmative_conversation.pk
    assert str(original.source_message_uuid) == str(affirmative_source.message_uuid)
    assert original.source_content_digest == sha256(affirmative_source.content.encode()).hexdigest()
    assert original.provider_memory_ids == ['provider-copy-2']
    assert original.provider_receipt_complete is False  # Graph coverage is unobservable.
    assert len(provider.adds) == 2
    assert provider.adds[1][0] == [{'role': 'user', 'content': ORIGINAL}]
    assert all(call[1]['enable_graph'] is True for call in provider.adds)
    assert extraction and all('Assistant invented preference.' not in call['messages'][-1]['content']
                              for call in extraction)
    assert not UserMemoryFact.objects.filter(fact='Assistant invented preference.').exists()

    # Keep known, source-attributed unknown and unprovable provider copies stale.
    provider.rows.extend([
        attributed_row(affirmative_source, ORIGINAL, 'provider-copy-2'),
        attributed_row(affirmative_source, PARAPHRASES[0], 'unknown-attributed'),
        {'id': 'unknown-unattributed', 'memory': PARAPHRASES[1], 'metadata': {'user_id': str(owner.pk)}},
    ])
    before_correction = prompt.capture(owner)
    assert all(value in before_correction for value in (SAFE, ORIGINAL, *PARAPHRASES))
    assert_absent(before_correction, FOREIGN, 'amber telescopes')
    assert edit(other_client, 'patch', original.pk, 1, replacement=CORRECTED).status_code == 404
    assert edit(client, 'patch', original.pk, 2, replacement=CORRECTED).status_code == 409
    response = edit(client, 'patch', original.pk, 1, replacement=CORRECTED)
    assert response.status_code == 200
    corrected_row = response.json()['fact']
    corrected = UserMemoryFact.objects.get(pk=corrected_row['id'])
    original.refresh_from_db()
    assert original.fact == '' and original.status == 'superseded' and original.revision == 2
    assert corrected.fact == CORRECTED and corrected.supersedes_id == original.pk
    assert ORIGINAL not in response.content.decode()
    job = ProfileMemoryReconciliationJob.objects.get(fact=original)
    assert job.state == 'pending' and job.attempts == 0
    assert broker.reconciliations == [{'fact_id': original.pk}]

    # Suppression precedes remote reconciliation, and overwrites the cached system.
    after_correction = prompt.capture(owner)
    assert SAFE in after_correction and CORRECTED in after_correction
    assert_absent(after_correction, ORIGINAL, *PARAPHRASES, FOREIGN, 'amber telescopes')
    assert provider.gets == provider.deletes == []
    tasks.reconcile_profile_memory_task.run(**broker.reconciliations[0])
    job.refresh_from_db()
    original.refresh_from_db()
    assert provider.gets == provider.deletes == ['provider-copy-2']
    assert job.state == original.reconciliation_status == 'failed'
    assert job.last_error == 'unidentified_provider_copies' and job.attempts == 1
    assert original.provider_receipt_complete is False

    response = edit(client, 'delete', corrected.pk, corrected.revision)
    assert response.status_code == 200 and CORRECTED not in response.content.decode()
    corrected.refresh_from_db()
    assert corrected.fact == '' and corrected.status == 'revoked'
    assert client.get(URL).json()['facts'] == []
    assert other_client.get(URL).json()['facts'][0]['fact'] == FOREIGN
    after_revocation = prompt.capture(owner)
    assert SAFE in after_revocation
    assert_absent(after_revocation, ORIGINAL, CORRECTED, *PARAPHRASES, FOREIGN, 'amber telescopes')
    assert len(provider.rows) == 5  # Transport still returns every stale copy.
    assert all(call['query'] == 'What should I use?' for call in provider.searches)
    assert provider.searches[-1]['user_id'] == str(owner.pk)

    # Delayed promotion receipts and duplicate durable-worker envelopes cannot revive it.
    broker.deliver_promotions()
    tasks.create_conversation_memories_task.run(**broker.jobs[-1]['kwargs'])
    assert len(provider.adds) == 2
    assert not UserMemoryFact.objects.filter(user=owner).exclude(fact='').exists()
    final = prompt.capture(owner)
    assert SAFE in final
    assert_absent(final, ORIGINAL, CORRECTED, *PARAPHRASES, FOREIGN, 'amber telescopes')
    assert UserMemoryFact.objects.get(user=other).fact == FOREIGN
