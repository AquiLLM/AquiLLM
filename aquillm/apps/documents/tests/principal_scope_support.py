"""Real disjoint principals and an already captured populated parent turn."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from django.contrib.auth.models import User

from apps.collections.models import Collection, CollectionPermission
from apps.documents.models import RawTextDocument, TextChunk
from apps.documents.services.representation_media_acquisition import initialize_media_runtime
from lib.llm.providers.openai import OpenAIInterface
from lib.llm.turn_context import TurnContext
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


@pytest.fixture
def principals(monkeypatch):
    from apps.documents.services import chunk_search
    monkeypatch.setattr(chunk_search, "load_query_embedding", lambda *args: None)
    for name in ("RAG_RERANK_TEXT_MODE", "RAG_DOCUMENT_CAPACITY_MODE", "RAG_EVIDENCE_TEXT_MODE"):
        monkeypatch.setenv(name, "legacy")
    monkeypatch.setenv("RAG_FOLLOWUP_EVIDENCE_ENABLED", "0")
    monkeypatch.setenv("RAG_ITERATIVE_RETRIEVAL_ENABLED", "0")
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "65536")
    alice = User.objects.create_user(username="original-alice")
    bob = User.objects.create_user(username="replacement-bob")
    documents, chunks = [], []
    for owner, label in ((alice, "Original"), (alice, "Alice requested"), (bob, "Bob requested")):
        collection = Collection.objects.create(name=label)
        CollectionPermission.objects.create(user=owner, collection=collection, permission="VIEW")
        document = RawTextDocument(title=label, full_text=label + " source",
            collection=collection, ingested_by=owner)
        document.save(dont_rechunk=True)
        chunk = TextChunk.objects.bulk_create([TextChunk(doc_id=document.id, content=document.full_text,
            chunk_number=0, start_position=0, end_position=len(document.full_text))])[0]
        documents.append(document)
        chunks.append(chunk)
    consumer = SimpleNamespace(user=alice, col_ref=SimpleNamespace(collections=[documents[0].collection_id]))
    turn = TurnContext(TurnBudget(TurnLimits()))
    parent = initialize_media_runtime(consumer, turn)
    assert parent.authorization.reauthorization_capability._principal is alice
    assert parent.authorization.selected_document_ids == frozenset((documents[0].id,))
    return SimpleNamespace(alice=alice, bob=bob, documents=documents, chunks=chunks,
        same_pk_principal=User.objects.get(pk=alice.pk),
        consumer=consumer, turn=turn, parent=parent,
        snapshot=parent._frozen_retrieval_scope.representation)


def sdk_for(document, chunk):
    citation = f"[doc:{document.id} chunk:{chunk.pk}]"
    create = AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=chunk.content + " " + citation, tool_calls=None),
        finish_reason="stop")], usage=None))
    provider = OpenAIInterface(SimpleNamespace(base_url="http://fake-models:8000/v1",
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))), "test")
    return provider, create, citation


def change_principal(case, change):
    if change == "different_user":
        case.consumer.user = case.bob
    elif change == "same_pk_object":
        case.consumer.user = case.same_pk_principal
        assert case.consumer.user is not case.alice
    elif change == "original_pk":
        case.alice.pk = case.bob.pk
