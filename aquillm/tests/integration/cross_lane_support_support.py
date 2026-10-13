"""Actual activated sources and compatible dense indexes for support controls."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from apps.collections.models import Collection, CollectionPermission
from apps.documents.models import RawTextDocument, TextChunk
from apps.documents.services.representation_store import freeze_active_representations
from apps.documents.tasks.embedding_index import build_embedding_generation, rebuild_document_embeddings
from apps.documents.tests.representation_support import activate, reference
from apps.ingestion.models import IngestionBatch
from aquillm.task_ingest_uploaded import run_ingest_uploaded_file

QUESTION = "What is the capacity?"
EXACT = "The capacity is 18 litres at 20 C."
IRRELEVANT = "This excerpt describes telescope optics."
CONTRADICTORY = (
    "The capacity is 18 litres at 20 C. The capacity is 12 litres at 40 C. "
    "These measurements conflict and neither is an accepted specification."
)
DISTRACTOR = "The telescope mirror is polished. Its optical alignment requires inspection."
UNCERTAIN = (
    "Support for an answer is uncertain in the selected evidence. "
    "These are attributed excerpts, not a conclusion about the question:"
)


def build_dense(doc, space):
    frozen = freeze_active_representations((reference(doc),), using="default")
    with patch.object(build_embedding_generation, "delay"):
        key = rebuild_document_embeddings(frozen.selections[0], space_key=space.space_key, using="default")
    assert build_embedding_generation.run(key, using="default", timeout_ms=120000) == "activated"
    chunk = doc.chunks.get()
    assert chunk.representation_generation_id is not None and chunk.search_vector is not None
    return chunk


def replace_target(corpus, text):
    assert activate(corpus.doc, corpus.output, text=text, parser="p1-support-" + str(len(text))) == "activated"
    corpus.doc.refresh_from_db()
    corpus.chunk = build_dense(corpus.doc, corpus.space)
    assert corpus.chunk.content == text


@pytest.fixture
def support_corpus(publication_source, upload, transport, monkeypatch, settings):
    source = publication_source
    for key, value in {
        "RAG_DIRECT_TOP_K": "2", "RAG_DIRECT_MAX_QUERIES": "3",
        "RAG_EVIDENCE_SELECTION_MODE": "adaptive", "RAG_VOCABULARY_QUERY_ENABLED": "0",
        "APP_RERANK_PROVIDER": "local", "APP_RERANK_BASE_URL": "http://p1-score:8000/v1",
        "APP_RERANK_MODEL": "p1-hostile-score", "APP_RERANK_MODEL_REVISION": "",
    }.items():
        monkeypatch.setenv(key, value)
    settings.RAG_CACHE_ENABLED = False
    corpus = SimpleNamespace(**vars(source), space=transport[0])
    replace_target(corpus, EXACT)
    other = Collection.objects.create(name="independent distractor collection")
    CollectionPermission.objects.create(user=source.user, collection=other, permission="EDIT")
    item = upload((("distractor.txt", DISTRACTOR.encode()),))
    item.batch = IngestionBatch.objects.create(user=source.user, collection=other)
    item.save(update_fields=["batch"])
    run_ingest_uploaded_file(item.pk)
    output = item.outputs.get(status="success")
    doc = RawTextDocument.objects.get(pk=output.document_pkid)
    assert activate(doc, output, text=DISTRACTOR, parser="p1-support-distractor") == "activated"
    doc.refresh_from_db()
    corpus.distractor, corpus.distractor_chunk = doc, build_dense(doc, corpus.space)
    empty_collection = Collection.objects.create(name="authorized empty source")
    CollectionPermission.objects.create(user=source.user, collection=empty_collection, permission="VIEW")
    corpus.empty = RawTextDocument.objects.create(collection=empty_collection, ingested_by=source.user,
        title="Empty owner", full_text="")
    assert not TextChunk.objects.filter(doc_id=corpus.empty.id).exists()
    return corpus


def citation(chunk):
    return f"[doc:{chunk.doc_id} chunk:{chunk.pk}]"


def literal_excerpt(text, reference):
    # Explicit frozen-case attribution, independent of the renderer under test.
    assert text in (EXACT, IRRELEVANT, CONTRADICTORY, DISTRACTOR)
    if text == CONTRADICTORY:
        return (
            f"> The capacity is 18 litres at 20 C. {reference} "
            f"The capacity is 12 litres at 40 C. {reference} "
            "These measurements conflict and neither is an accepted "
            f"specification. {reference}"
        )
    return "> " + text + " " + reference
