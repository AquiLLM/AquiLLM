from __future__ import annotations

import uuid

import pytest

from ._chunk_graph_lifecycle_support import (
    configure_chunking_runtime,
    database_required,
    persist_chunk,
    persist_document,
    run_chunk_task,
)


def _snapshot(document):
    from apps.documents.models import TextChunk

    return tuple(
        TextChunk.objects.filter(doc_id=document.id).values_list(
            "pk", "content", "chunk_number", "metadata"
        )
    )


@pytest.mark.django_db(transaction=True)
@database_required
@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("doc_id", "not-a-uuid"),
        ("expected_source_hash", "not-a-sha256"),
        ("concrete_model_label", "Not.A.Canonical.Model"),
        ("document_pkid", 0),
    ),
)
def test_invalid_four_scalar_identity_fails_before_any_mutation(
    monkeypatch,
    field,
    invalid_value,
):
    from django.contrib.auth.models import User
    from django.core.exceptions import ObjectDoesNotExist, ValidationError

    from apps.collections.models import Collection
    from apps.documents.models import RawTextDocument
    from apps.knowledge_graph.services import builds

    chunking = configure_chunking_runtime(monkeypatch)
    user = User.objects.create_user(username=f"invalid-ref-{field}-{uuid.uuid4()}")
    collection = Collection.objects.create(name=f"invalid ref {field} {uuid.uuid4()}")
    document = persist_document(
        RawTextDocument,
        user=user,
        collection=collection,
        text="Invalid task identity cannot touch this document.",
        ingestion_complete=False,
    )
    persist_chunk(document, content="identity fence")
    before = _snapshot(document)
    arguments = {
        "doc_id": str(document.id),
        "expected_source_hash": document.full_text_hash,
        "concrete_model_label": document._meta.label_lower,
        "document_pkid": int(document.pkid),
    }
    arguments[field] = invalid_value
    publications = []
    monkeypatch.setattr(
        builds, "enqueue_document_build", lambda *args: publications.append(args)
    )

    with pytest.raises((ValueError, ValidationError, ObjectDoesNotExist)):
        chunking.create_chunks.run(**arguments)

    document.refresh_from_db()
    assert document.ingestion_complete is False
    assert _snapshot(document) == before
    assert publications == []


@pytest.mark.django_db(transaction=True)
@database_required
@pytest.mark.parametrize(
    "reference_override",
    (
        {"document_pkid": 2**62},
        {"concrete_model_label": "apps_documents.imageuploaddocument"},
    ),
)
def test_mismatched_exact_document_reference_fails_closed_without_mutation(
    monkeypatch,
    reference_override,
):
    from django.contrib.auth.models import User
    from django.core.exceptions import ObjectDoesNotExist, ValidationError

    from apps.collections.models import Collection
    from apps.documents.models import RawTextDocument
    from apps.knowledge_graph.services import builds

    chunking = configure_chunking_runtime(monkeypatch)
    user = User.objects.create_user(username=f"mismatched-ref-{uuid.uuid4()}")
    collection = Collection.objects.create(name=f"mismatched ref {uuid.uuid4()}")
    document = persist_document(
        RawTextDocument,
        user=user,
        collection=collection,
        text="A valid UUID is insufficient without its exact concrete row.",
        ingestion_complete=False,
    )
    persist_chunk(document, content="exact reference fence")
    before = _snapshot(document)
    arguments = {
        "doc_id": str(document.id),
        "expected_source_hash": document.full_text_hash,
        "concrete_model_label": document._meta.label_lower,
        "document_pkid": int(document.pkid),
    }
    arguments.update(reference_override)
    publications = []
    monkeypatch.setattr(
        builds, "enqueue_document_build", lambda *args: publications.append(args)
    )

    with pytest.raises((ValueError, ValidationError, ObjectDoesNotExist)):
        chunking.create_chunks.run(**arguments)

    document.refresh_from_db()
    assert document.ingestion_complete is False
    assert _snapshot(document) == before
    assert publications == []


@pytest.mark.django_db(transaction=True)
@database_required
def test_ambiguous_cross_table_uuid_fails_closed_before_chunk_mutation(monkeypatch):
    from django.contrib.auth.models import User
    from django.core.exceptions import ObjectDoesNotExist, ValidationError

    from apps.collections.models import Collection
    from apps.documents.models import ImageUploadDocument, RawTextDocument
    from apps.knowledge_graph.services import builds

    chunking = configure_chunking_runtime(monkeypatch)
    user = User.objects.create_user(username=f"ambiguous-ref-{uuid.uuid4()}")
    raw_collection = Collection.objects.create(name=f"ambiguous raw {uuid.uuid4()}")
    image_collection = Collection.objects.create(name=f"ambiguous image {uuid.uuid4()}")
    text = "The UUID is intentionally ambiguous across concrete tables."
    raw_document = persist_document(
        RawTextDocument,
        user=user,
        collection=raw_collection,
        text=text,
        ingestion_complete=False,
    )
    image_document = persist_document(
        ImageUploadDocument,
        user=user,
        collection=image_collection,
        text=text,
        ingestion_complete=False,
        id=raw_document.id,
        image_file="ingestion_images/ambiguous.png",
    )
    persist_chunk(raw_document, content="ambiguous UUID fence")
    before = _snapshot(raw_document)
    publications = []
    monkeypatch.setattr(
        builds, "enqueue_document_build", lambda *args: publications.append(args)
    )

    with pytest.raises((ValueError, ValidationError, ObjectDoesNotExist)):
        run_chunk_task(chunking, raw_document)

    raw_document.refresh_from_db()
    image_document.refresh_from_db()
    assert raw_document.ingestion_complete is False
    assert image_document.ingestion_complete is False
    assert _snapshot(raw_document) == before
    assert publications == []
