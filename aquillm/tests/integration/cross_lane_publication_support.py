"""Real lexical retrieval, provider request and durable answer sink fixtures."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import pytest_asyncio
from channels.db import database_sync_to_async
from django.db import connections

from apps.chat.consumers.chat_publish import end_spin_publish
from apps.chat.refs import CollectionsRef
from apps.chat.services.rag_pipeline import run_direct_rag_turn
from apps.documents.services import embedding_index_store
from apps.documents.tests.representation_support import activate
from aquillm.message_adapters import save_conversation_to_db
from aquillm.models import WSConversation
from lib.embeddings import config
from lib.llm.providers.final_stream import send_final_answer_stream
from lib.llm.providers.openai import OpenAIInterface
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import UserMessage

SOURCE = "α😀 Capacity is 18 litres at 20 C. The capacity applies only below 1 Pa."
REVISED = "α😀 Capacity is 24 litres at 30 C. The capacity applies only below 2 Pa."
QUESTION = "What capacity and operating conditions are stated in the selected document?"


@pytest.fixture
def publication_source(retained, transport, monkeypatch, settings):
    doc, output, _ = retained
    for key, value in {
        "RAG_DIRECT_ENABLED": "1", "RAG_DIRECT_MAX_QUERIES": "1",
        "RAG_DIRECT_TOP_K": "3", "RAG_EVIDENCE_SELECTION_MODE": "legacy",
        "RAG_RERANK_TEXT_MODE": "legacy", "RAG_DOCUMENT_CAPACITY_MODE": "legacy",
        "RAG_FOLLOWUP_EVIDENCE_ENABLED": "0", "RAG_ITERATIVE_RETRIEVAL_ENABLED": "0",
        "RAG_QUERY_REWRITE_ENABLED": "0", "RAG_APPEND_CITATION_SOURCES": "0",
        "OPENAI_CONTEXT_LIMIT": "32768", "OPENAI_STREAM_RESPONSES": "0",
        "CONTEXT_PACKER_ENABLED": "0", "TOKEN_EFFICIENCY_ENABLED": "0",
        "LLM_STREAM_FINAL_ANSWER_ONLY": "1",
    }.items():
        monkeypatch.setenv(key, value)
    settings.KG_OVERLAY_ENABLED = False
    settings.DOCUMENT_EMBEDDING_SPACE_KEY = transport[0].space_key
    embedding_index_store.persist_embedding_space(transport[0], using="default")
    monkeypatch.setattr(config, "get_local_embed_config", lambda: ("http://fake-models:8000/v1", "synthetic-test-key", transport[0].model))
    assert activate(doc, output, text=SOURCE, parser="p1-publication-a") == "activated"
    doc.refresh_from_db()
    chunk = doc.chunks.get()
    assert chunk.search_vector is not None and chunk.representation_generation_id is not None
    return SimpleNamespace(doc=doc, user=doc.ingested_by, output=output, chunk=chunk,
                           db=WSConversation.objects.create(owner=doc.ingested_by))


@pytest_asyncio.fixture(autouse=True)
async def close_publication_worker(transactional_db):
    yield
    await database_sync_to_async(connections.close_all)()


class Provider:
    """Replace the external SDK transport only; all request/recovery code is real."""
    def __init__(self, answers):
        self.answers, self.requests = list(answers), []
        self.llm = OpenAIInterface(SimpleNamespace(base_url="http://fake-models:8000/v1",
            chat=SimpleNamespace(completions=self)), "qwen3.5:27b")

    async def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        assert self.answers, "unexpected provider request"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            tool_calls=[], content=self.answers.pop(0)), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20))


def consumer_for(source, question=QUESTION):
    convo = Conversation(system="Use the selected source.", messages=[UserMessage(content=question)])
    return SimpleNamespace(user=source.user,
        col_ref=CollectionsRef([source.doc.collection_id]), convo=convo)


async def retrieve(consumer, provider):
    assert await run_direct_rag_turn(consumer, provider.llm, consumer.convo) == "handled"
    assert consumer.convo[-1]._validated_answer is not None
    return consumer.convo[-1]


async def publish(source, consumer):
    events = []
    message = consumer.convo[-1]
    async def stream(payload):
        events.append(("stream", payload["content"]))
    async def send(**kwargs):
        payload = json.loads(kwargs["text_data"])
        events.append(("refresh", payload["delta"]["messages"][0]["content"]))
    await send_final_answer_stream(stream, message_uuid=str(message.message_uuid),
        content=message.content, stop_reason=message.stop_reason, usage=message.usage, message=message)
    await end_spin_publish(SimpleNamespace(convo=consumer.convo,
        _spin_epoch=len(consumer.convo)-1, send=send), consumer.convo)
    await database_sync_to_async(save_conversation_to_db)(consumer.convo, source.db)
    stored = await database_sync_to_async(lambda: source.db.db_messages.get(
        message_uuid=message.message_uuid).content)()
    assert events == [("stream", message.content), ("refresh", message.content)]
    assert stored == message.content
    return stored, events


def cited_reply(text, citation):
    return text.replace(". ", ". " + citation + " ") + " " + citation
