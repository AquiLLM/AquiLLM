"""Narrative words cannot grant persistence consent or lose their prefix."""
from hashlib import sha256

import pytest
from django.contrib.auth import get_user_model

from apps.chat.models import Message, WSConversation
from apps.memory.models import UserMemoryFact
from apps.memory.services.profile_facts import promote_source_payload
from lib.memory.extraction.stable_facts import clean_stable_facts

pytestmark = pytest.mark.django_db(transaction=True)


def persist(text):
    user = get_user_model().objects.create(username="directive-token-owner")
    conversation = WSConversation.objects.create(owner=user)
    message = Message.objects.create(
        conversation=conversation, role="user", content=text, sequence_number=0)
    count = promote_source_payload(
        user=user, user_content=text, source_conversation_id=conversation.pk,
        source_message_uuid=str(message.message_uuid),
        source_content_digest=sha256(text.encode()).hexdigest())
    return count, message


@pytest.mark.parametrize("text", [
    "Remembering my childhood makes me happy.",
    "Remembered events are sometimes unreliable.",
    "Rememberers often share stories together.",
    "Also remembering my childhood makes me happy.",
])
def test_narrative_prefix_is_not_consent(text):
    count, _ = persist(text)
    assert count == 0
    assert not UserMemoryFact.objects.exists()
    assert clean_stable_facts([text]) == [text]


@pytest.mark.parametrize("text, payload", [
    ("Remember that I prefer telescope A.", "I prefer telescope A."),
    ("Please remember: I prefer telescope A.", "I prefer telescope A."),
    ("Could you remember this going forward: I prefer telescope A.", "I prefer telescope A."),
])
def test_standalone_directive_retains_exact_source_span(text, payload):
    count, message = persist(text)
    assert count == 1
    fact = UserMemoryFact.objects.get()
    assert fact.fact == payload
    assert fact.source_message_uuid == message.message_uuid
    assert text[fact.source_span_start:fact.source_span_end] == payload
