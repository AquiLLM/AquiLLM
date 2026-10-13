"""Public synthetic corpus with actual I3/I4 publication and vocabulary APIs."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import pytest_asyncio
from channels.db import database_sync_to_async
from django.db import connections

from apps.collections.models import Collection, CollectionPermission
from apps.collections.tests.test_vocabulary_scope import entry, publish, service
from apps.documents.models import RawTextDocument, TextChunk
from apps.documents.services import embedding_index_store as store
from apps.documents.services.representation_store import freeze_active_representations
from apps.documents.tasks.embedding_index import build_embedding_generation, rebuild_document_embeddings
from apps.documents.tests.representation_support import activate, reference
from aquillm.models import WSConversation
from aquillm.task_ingest_uploaded import run_ingest_uploaded_file
from lib.embeddings import config

QUESTION = "Explain BRX-9 response"
REWRITE = "Describe BRX-9 response"
EXPANSION = "baroreceptor experimental unit"
PRIVATE = "UNSELECTED-ALIAS-SENTINEL"
TEXTS = (
    "BRX-9 response is conditional. The response applies only below 1 Pa.",
    "The baroreceptor experimental unit response is conditional. It applies only below 1 Pa.",
)


@pytest.fixture
def query_corpus(upload, transport, monkeypatch, settings):
    for key, value in {
        "RAG_DIRECT_ENABLED": "1", "RAG_DIRECT_MAX_QUERIES": "3",
        "RAG_DIRECT_TOP_K": "1", "RAG_EVIDENCE_SELECTION_MODE": "legacy",
        "RAG_RERANK_TEXT_MODE": "legacy", "RAG_DOCUMENT_CAPACITY_MODE": "legacy",
        "RAG_ITERATIVE_RETRIEVAL_ENABLED": "0", "RAG_RERANK_SHADOW_SCORING_ENABLED": "0",
        "RAG_APPEND_CITATION_SOURCES": "0", "OPENAI_CONTEXT_LIMIT": "32768",
        "OPENAI_STREAM_RESPONSES": "0", "CONTEXT_PACKER_ENABLED": "0",
        "TOKEN_EFFICIENCY_ENABLED": "0", "LLM_STREAM_FINAL_ANSWER_ONLY": "1",
        "MEM0_ENABLED": "0", "APP_RERANK_PROVIDER": "local",
        "APP_RERANK_BASE_URL": "http://synthetic-r5-score:8000/v1",
        "APP_RERANK_MODEL": "r5-constant-score", "APP_RERANK_MODEL_REVISION": "",
    }.items():
        monkeypatch.setenv(key, value)
    settings.KG_OVERLAY_ENABLED = False
    settings.RAG_CACHE_ENABLED = False
    settings.DOCUMENT_EMBEDDING_SPACE_KEY = transport[0].space_key
    monkeypatch.setattr(config, "get_local_embed_config", lambda: (
        "http://fake-models:8000/v1", "synthetic-test-key", transport[0].model))
    store.persist_embedding_space(transport[0], using="default")
    item = upload(tuple((f"source-{i}.txt", text.encode()) for i, text in enumerate(TEXTS)))
    run_ingest_uploaded_file(item.pk)
    outputs = list(item.outputs.filter(status="success").order_by("pk"))
    assert len(outputs) == 2
    docs, chunks = [], []
    for index, output in enumerate(outputs):
        doc = RawTextDocument.objects.get(pk=output.document_pkid)
        doc.title = "Red" if index == 0 else "Expanded"
        doc.save(dont_rechunk=True)
        assert activate(doc, output, text=TEXTS[index], parser="r5-query-v1") == "activated"
        doc.refresh_from_db()
        chunk = doc.chunks.get()
        assert chunk.representation_generation_id is not None and chunk.search_vector is not None
        snapshot = freeze_active_representations((reference(doc),), using="default")
        with patch.object(build_embedding_generation, "delay"):
            key = rebuild_document_embeddings(snapshot.selections[0],
                space_key=transport[0].space_key, using="default")
        assert build_embedding_generation.run(key, using="default", timeout_ms=120000) == "activated"
        docs.append(doc)
        chunks.append(chunk)
    actor, selected = docs[0].ingested_by, docs[0].collection
    CollectionPermission.objects.filter(user=actor, collection=selected).update(permission="MANAGE")
    vocabulary, version = publish(actor, selected, [entry("BRX-9", EXPANSION)])
    pin = service().attach_vocabulary(actor=actor, collection=selected, version=version, expected_revision=0)
    foreign = Collection.objects.create(name="unselected synthetic source")
    CollectionPermission.objects.create(user=actor, collection=foreign, permission="MANAGE")
    hidden_doc = RawTextDocument.objects.create(collection=foreign, title="Unselected",
        full_text=PRIVATE, ingested_by=actor)
    TextChunk.objects.bulk_create([TextChunk(doc_id=hidden_doc.id, chunk_number=0,
        content=PRIVATE, start_position=0, end_position=len(PRIVATE))])
    _, hidden = publish(actor, foreign, [entry("BRX-9", PRIVATE)])
    service().attach_vocabulary(actor=actor, collection=foreign, version=hidden, expected_revision=0)
    return SimpleNamespace(user=actor, collection=selected, docs=docs, chunks=chunks,
        outputs=outputs, vocabulary=vocabulary, version=version, pin=pin,
        hidden=hidden, db=WSConversation.objects.create(owner=actor))


@pytest_asyncio.fixture(autouse=True)
async def close_query_worker(transactional_db):
    yield
    await database_sync_to_async(connections.close_all)()


def controls(monkeypatch, arm, followup, *, iterative=False):
    for key, enabled in {
        "RAG_VOCABULARY_QUERY_ENABLED": arm in {"vocabulary", "interaction"},
        "RAG_QUERY_REWRITE_ENABLED": arm in {"rewrite", "interaction"},
        "RAG_FOLLOWUP_EVIDENCE_ENABLED": followup,
        "RAG_ITERATIVE_RETRIEVAL_ENABLED": iterative,
    }.items():
        monkeypatch.setenv(key, str(int(enabled)))
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source" if followup else "legacy")


def mutate(corpus, change):
    if change == "detach":
        service().detach_vocabulary(actor=corpus.user, attachment=corpus.pin,
            expected_revision=corpus.pin.revision)
    elif change == "repin":
        draft = service().save_draft(actor=corpus.user, vocabulary=corpus.vocabulary,
            expected_revision=2, entries=[entry("BRX-9", "changed curator expansion")])
        version = service().publish_vocabulary(actor=corpus.user,
            vocabulary=corpus.vocabulary, expected_revision=draft.revision)
        service().attach_vocabulary(actor=corpus.user, collection=corpus.collection,
            version=version, expected_revision=corpus.pin.revision)
    elif change == "revoke":
        CollectionPermission.objects.filter(user=corpus.user, collection=corpus.collection).delete()
    elif change == "generation":
        assert activate(corpus.docs[0], corpus.outputs[0], text=TEXTS[0],
            parser="r5-query-replacement") == "activated"
        corpus.docs[0].refresh_from_db()
        assert corpus.docs[0].chunks.get().pk != corpus.chunks[0].pk
        assert TextChunk.objects.filter(pk=corpus.chunks[0].pk, content=TEXTS[0]).exists()
    else:
        raise AssertionError(change)


def inherit_from_parent(corpus):
    service().detach_vocabulary(actor=corpus.user, attachment=corpus.pin,
        expected_revision=corpus.pin.revision)
    parent = Collection.objects.create(name="explicit vocabulary ancestor")
    CollectionPermission.objects.create(user=corpus.user, collection=parent, permission="MANAGE")
    corpus.collection.parent = parent
    corpus.collection.save(update_fields=["parent"])
    vocabulary, version = publish(corpus.user, parent, [entry("BRX-9", EXPANSION)])
    pin = service().attach_vocabulary(actor=corpus.user, collection=parent,
        version=version, expected_revision=0, inherit_descendants=True)
    corpus.vocabulary, corpus.version, corpus.pin = vocabulary, version, pin
    return parent
