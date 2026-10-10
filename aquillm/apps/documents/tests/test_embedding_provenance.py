"""Stored receipts must follow their vectors through every supported writer."""

from io import StringIO
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from django.core.management import call_command

from lib.embeddings.provenance import make_result, valid_provenance


@pytest.fixture(autouse=True)
def no_legacy_transport(monkeypatch):
    # Baseline writers still use these APIs; keep RED safe and finite too.
    from aquillm import utils

    monkeypatch.setattr(utils, "get_embedding", lambda *_, **__: [0.1] * 1024)
    monkeypatch.setattr(
        utils, "get_multimodal_embedding", lambda *_, **__: [0.2] * 1024
    )


def result(value=0.1, text="text"):
    return make_result(
        [value] * 1024,
        provider="local-openai",
        route="openai-embeddings",
        role="search_document",
        prepared_input=text,
        original_input=text,
        model="fixture-model",
    )


def chunk(**kwargs):
    from apps.documents.models import TextChunk

    return TextChunk(
        content="text",
        doc_id=uuid4(),
        chunk_number=0,
        start_position=0,
        end_position=4,
        **kwargs,
    )


@pytest.mark.django_db
def test_text_save_persists_new_pair_and_clears_receipt_on_replacement(monkeypatch):
    from aquillm import utils
    from apps.documents.models import TextChunk

    response = result()
    monkeypatch.setattr(
        utils, "get_embedding_result", lambda *_, **__: response, raising=False
    )
    row = chunk()
    row.save()
    row.refresh_from_db()
    assert row.embedding_provenance == response.provenance
    assert valid_provenance(row.embedding, row.embedding_provenance)
    row.embedding = [0.2] * 1024
    row.save(update_fields=["embedding"])
    row.refresh_from_db()
    assert row.embedding_provenance is None
    response2 = result(0.3)
    row.embedding, row.embedding_provenance = response2.vector, response2.provenance
    TextChunk.objects.bulk_update([row], ["embedding", "embedding_provenance"])
    row.refresh_from_db()
    assert (
        valid_provenance(row.embedding, row.embedding_provenance)
        == response2.provenance
    )


@pytest.mark.django_db
def test_orm_bulk_and_update_clear_stale_receipts_and_preserve_valid_pairs():
    from apps.documents.models import TextChunk

    response = result()
    row = chunk(embedding=response.vector, embedding_provenance=response.provenance)
    TextChunk.objects.bulk_create([row])
    row.refresh_from_db()
    assert valid_provenance(row.embedding, row.embedding_provenance)
    row.embedding = [0.2] * 1024
    TextChunk.objects.bulk_update([row], ["embedding"])
    row.refresh_from_db()
    assert row.embedding_provenance is None
    TextChunk.objects.filter(pk=row.pk).update(
        embedding=response.vector, embedding_provenance=response.provenance
    )
    row.refresh_from_db()
    assert row.embedding_provenance == response.provenance
    TextChunk.objects.filter(pk=row.pk).update(embedding=[0.3] * 1024)
    row.refresh_from_db()
    assert row.embedding_provenance is None
    with pytest.raises(ValueError):
        TextChunk.objects.filter(pk=row.pk).update(
            embedding_provenance=response.provenance
        )


@pytest.mark.django_db
@pytest.mark.parametrize("known", [True, False])
def test_duplicate_text_preserves_known_or_unknown_receipt(monkeypatch, known):
    from apps.documents.models import RawTextDocument, TextChunk
    from apps.documents.tasks import chunking
    from ._chunk_graph_lifecycle_support import persist_document
    from django.contrib.auth.models import User
    from apps.collections.models import Collection

    user = User.objects.create_user(username="receipt-copy")
    collection = Collection.objects.create(name="receipt-copy")
    source = persist_document(
        RawTextDocument,
        user=user,
        collection=collection,
        text="text",
        ingestion_complete=True,
    )
    target_collection = Collection.objects.create(name="receipt-copy-target")
    target = persist_document(
        RawTextDocument,
        user=user,
        collection=target_collection,
        text="text",
        ingestion_complete=False,
    )
    response = result()
    row = chunk(
        embedding=response.vector,
        embedding_provenance=response.provenance if known else None,
    )
    row.doc_id = source.id
    TextChunk.objects.bulk_create([row])
    copied = chunking._duplicate_chunks(target, using="default")
    TextChunk.objects.bulk_create(copied)
    saved = TextChunk.objects.get(doc_id=target.id)
    assert saved.embedding_provenance == (response.provenance if known else None)
    assert saved.embedding is not None


def test_document_batch_and_target_image_use_matching_fresh_receipts(monkeypatch):
    from aquillm import utils
    from apps.documents.tasks import chunking
    from apps.documents.services import chunk_embeddings

    text_result, image_result = result(0.1), result(0.2, "target-image")
    monkeypatch.setattr(chunking, "get_embeddings", lambda *_, **__: [text_result])
    monkeypatch.setattr(
        utils,
        "get_multimodal_embedding_result",
        lambda *_, **__: image_result,
        raising=False,
    )
    monkeypatch.setattr(
        chunk_embeddings, "image_data_url", lambda _: "data:image/png;base64,target"
    )
    monkeypatch.setattr(
        chunking, "notify_ingest_monitor_progress", lambda *_, **__: None
    )
    text, image = chunk(), chunk(modality="image")
    rows = chunking._embed_chunks(SimpleNamespace(id=uuid4()), [text], image)
    assert [r.embedding for r in rows] == [text_result.vector, image_result.vector]
    assert [r.embedding_provenance for r in rows] == [
        text_result.provenance,
        image_result.provenance,
    ]


@pytest.mark.django_db
def test_conversation_indexing_persists_pairs_and_unknown_external_vectors(monkeypatch):
    from apps.chat.models import ConversationChunk, Message, WSConversation
    from apps.chat.services import conversation_indexing
    from django.contrib.auth.models import User

    user = User.objects.create_user(username="receipt-chat")
    convo = WSConversation.objects.create(owner=user, name="receipt-chat")
    Message.objects.create(
        conversation=convo, role="user", content="hello", sequence_number=0
    )
    response = result()
    monkeypatch.setattr(
        conversation_indexing, "get_embeddings", lambda *_, **__: [response]
    )
    assert conversation_indexing.index_conversation(convo.pk) == 1
    row = ConversationChunk.objects.get(conversation=convo)
    assert row.embedding_provenance == response.provenance
    assert valid_provenance(row.embedding, row.embedding_provenance)
    row.embedding = [0.2] * 1024
    row.save(update_fields=["embedding"])
    row.refresh_from_db()
    assert row.embedding_provenance is None


@pytest.mark.django_db
def test_receipt_only_save_cannot_claim_unwritten_vector():
    response = result()
    row = chunk(embedding=response.vector, embedding_provenance=response.provenance)
    row.save()
    replacement = result(0.2)
    row.embedding, row.embedding_provenance = replacement.vector, replacement.provenance
    with pytest.raises(ValueError):
        row.save(update_fields=["embedding_provenance"])
    row.refresh_from_db()
    assert (
        valid_provenance(row.embedding, row.embedding_provenance) == response.provenance
    )


@pytest.mark.django_db
def test_unsupported_positional_save_does_not_write_vector_or_receipt():
    response = result()
    row = chunk(embedding=response.vector, embedding_provenance=response.provenance)
    row.save()
    row.embedding = [0.2] * 1024
    with pytest.raises(TypeError):
        row.save(False, False, None, ["embedding"])
    row.refresh_from_db()
    assert (
        valid_provenance(row.embedding, row.embedding_provenance) == response.provenance
    )


@pytest.mark.django_db
def test_positional_bulk_conflict_update_clears_replaced_vector_receipt():
    from apps.documents.models import TextChunk

    response = result()
    row = chunk(embedding=response.vector, embedding_provenance=response.provenance)
    row.save()
    replacement = chunk(embedding=[0.2] * 1024)
    replacement.doc_id = row.doc_id
    TextChunk.objects.bulk_create(
        [replacement], None, False, True, ["embedding"], ["doc_id", "chunk_number"]
    )
    row.refresh_from_db()
    assert row.embedding_provenance is None


@pytest.mark.django_db
def test_audit_is_bounded_read_only_and_returns_only_counts():
    from apps.documents.models import TextChunk

    response = result()
    TextChunk.objects.bulk_create(
        [
            chunk(embedding=response.vector, embedding_provenance=response.provenance),
            chunk(embedding=response.vector),
        ]
    )
    output = StringIO()
    call_command("audit_embedding_provenance", limit=1, stdout=output)
    report = json.loads(output.getvalue())
    assert report["documents"]["scanned"] == 1
    assert report["documents"]["has_more"] is True
    assert report["documents"]["known"] == 1
    assert TextChunk.objects.count() == 2
