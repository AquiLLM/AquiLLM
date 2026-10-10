"""Conversation outage preserves keyword chunks without per-window retries."""
from unittest.mock import Mock
import pytest
from django.contrib.auth import get_user_model
from apps.chat.models import WSConversation, Message, ConversationChunk
from apps.chat.services import conversation_indexing as indexing
from apps.chat.tasks.conversation_indexing import index_conversation_task
from aquillm.utils import EmbeddingUpstreamUnavailableError
from lib.embeddings.utils import EmbeddingContractError

pytestmark = pytest.mark.django_db

@pytest.fixture
def conversation():
    user = get_user_model().objects.create(username="synthetic-outage")
    convo = WSConversation.objects.create(owner=user, name="Synthetic")
    Message.objects.create(conversation=convo, sequence_number=0, role="user", content="Synthetic message " * 400)
    return convo


def test_outage_has_one_batch_and_keyword_chunks(conversation, monkeypatch):
    batch = Mock(side_effect=EmbeddingUpstreamUnavailableError("offline"))
    single = Mock()
    monkeypatch.setattr(indexing, "get_embeddings", batch)
    monkeypatch.setattr(indexing, "get_embedding", single)
    assert indexing.index_conversation(conversation.pk) > 0
    assert batch.call_count == 1
    single.assert_not_called()
    conversation.refresh_from_db()
    assert not conversation.index_complete
    assert not ConversationChunk.objects.filter(conversation=conversation, embedding__isnull=False).exists()


@pytest.mark.parametrize("failure", [EmbeddingContractError("malformed"), TypeError("bug")])
def test_permanent_failures_do_not_retry_task(conversation, monkeypatch, failure):
    monkeypatch.setattr(indexing, "get_embeddings", Mock(side_effect=failure))
    retry = Mock()
    monkeypatch.setattr(index_conversation_task, "retry", retry)
    with pytest.raises(type(failure)):
        index_conversation_task.run(conversation.pk)
    retry.assert_not_called()
    assert not ConversationChunk.objects.filter(conversation=conversation).exists()


def test_keyword_outage_task_retry_chain_is_six_attempts(conversation, monkeypatch):
    from celery.exceptions import Retry, MaxRetriesExceededError
    provider = Mock(side_effect=EmbeddingUpstreamUnavailableError("offline"))
    monkeypatch.setattr(indexing, "get_embeddings", provider)
    for number in range(6):
        index_conversation_task.push_request(retries=number, called_directly=False, is_eager=True)
        try:
            with pytest.raises(Retry if number < 5 else MaxRetriesExceededError):
                index_conversation_task.run(conversation.pk)
        finally:
            index_conversation_task.pop_request()
    assert provider.call_count == 6
    conversation.refresh_from_db()
    assert not conversation.index_complete
    assert conversation.chunks.exists()
