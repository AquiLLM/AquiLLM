"""Actual citation repair cannot publish an unsupported number at any final sink."""
import pytest

from apps.chat.services.rag_turn import preservation_turn
from apps.documents.tests.representation_support import retained as retained, upload as upload
from lib.embeddings.tests.test_document_provider import transport as transport
from .cross_lane_publication_support import (
    Provider, SOURCE, consumer_for, publish, retrieve,
    close_publication_worker as close_publication_worker,
    publication_source as publication_source,
)

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["legacy", "source"])
async def test_valid_repaired_citation_cannot_carry_invented_quantity(publication_source, monkeypatch, mode):
    source = publication_source
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", mode)
    citation = f"[doc:{source.doc.id} chunk:{source.chunk.pk}]"
    fabricated = "Capacity is 999 litres at 12 V. " + citation
    provider = Provider(["Capacity is 999 litres [doc:missing chunk:999999].", fabricated])
    consumer = consumer_for(source)
    async with preservation_turn(consumer, max_func_calls=3):
        message = await retrieve(consumer, provider)
        assert len(provider.requests) == 2 and not provider.answers
        assert all(SOURCE in str(request["messages"]) for request in provider.requests)
        assert citation in str(provider.requests[-1]["messages"])
        expected = ("Support for an answer is uncertain in the selected evidence. "
            "These are attributed excerpts, not a conclusion about the question:\n\n"
            f"> α😀 Capacity is 18 litres at 20 C. {citation} "
            f"The capacity applies only below 1 Pa. {citation}")
        assert message.content == expected and message._validated_answer.outcome == "partial"
        saved, events = await publish(source, consumer)
        assert saved == expected
        assert all("999 litres" not in text and "12 V" not in text for _, text in events)
        assert fabricated not in consumer.convo.model_dump_json()
