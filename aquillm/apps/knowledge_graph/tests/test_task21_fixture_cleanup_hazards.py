from __future__ import annotations

from uuid import uuid4

import pytest
from django.contrib.auth.models import Group, User
from django.core.management import CommandError
from django.test import override_settings

from apps.knowledge_graph.tests.task21_fixture_test_support import (
    VISIBLE_USERNAME,
    cleanup,
    database_counts,
    seed,
    strict_eval_environment,
)

_STRICT_EVAL_ENVIRONMENT = strict_eval_environment


def _all_counts():
    from apps.documents.models import DocumentFigure

    return {**database_counts(), "figures": DocumentFigure.objects.count()}


@pytest.mark.django_db(transaction=True)
@override_settings(DEBUG=True)
@pytest.mark.parametrize(
    "hazard",
    (
        "child_collection",
        "extra_document",
        "duplicate_document_uuid",
        "extra_chunk",
        "foreign_child_figure",
        "user_reference",
        "user_document_reference",
        "user_duplicate_uuid_foreign_collection",
    ),
)
def test_cleanup_fences_cascade_hazard_before_any_deletion(
    tmp_path, monkeypatch, hazard
) -> None:
    from apps.collections.models import Collection
    from apps.documents.models import RawTextDocument, TextChunk

    manifest_path = tmp_path / "fixture.json"
    payload, _output, _observed = seed(manifest_path, monkeypatch)
    collection_id = payload["authorized_scope"][0]["collection_id"]
    if hazard == "child_collection":
        Collection.objects.create(
            name="foreign-child-collection", parent_id=collection_id
        )
    elif hazard == "extra_document":
        visible = User.objects.get(username=VISIBLE_USERNAME)
        text = "Foreign synthetic document text."
        RawTextDocument.objects.bulk_create(
            [
                RawTextDocument(
                    id=uuid4(),
                    title="Foreign synthetic document",
                    full_text=text,
                    full_text_hash=RawTextDocument.hash_fn(text),
                    collection_id=collection_id,
                    ingested_by=visible,
                    ingestion_complete=True,
                )
            ]
        )
    elif hazard == "duplicate_document_uuid":
        document_id = next(
            row["document_id"]
            for row in payload["documents"].values()
            if payload["collections"][row["collection_symbol"]]["collection_id"]
            == collection_id
        )
        visible = User.objects.get(username=VISIBLE_USERNAME)
        text = "Foreign duplicate logical UUID text."
        RawTextDocument.objects.bulk_create(
            [
                RawTextDocument(
                    id=document_id,
                    title="Foreign duplicate logical UUID",
                    full_text=text,
                    full_text_hash=RawTextDocument.hash_fn(text),
                    collection_id=collection_id,
                    ingested_by=visible,
                    ingestion_complete=True,
                )
            ]
        )
    elif hazard == "extra_chunk":
        document_id = next(iter(payload["documents"].values()))["document_id"]
        TextChunk.objects.bulk_create(
            [
                TextChunk(
                    doc_id=document_id,
                    content="Foreign synthetic chunk.",
                    start_position=10_000,
                    end_position=10_024,
                    chunk_number=999,
                    modality=TextChunk.Modality.TEXT,
                    metadata={"foreign": True},
                    embedding=[0.0] * 1024,
                )
            ]
        )
    elif hazard == "foreign_child_figure":
        from django.contrib.contenttypes.models import ContentType

        from apps.documents.models import DocumentFigure

        parent = RawTextDocument.objects.filter(collection_id=collection_id).first()
        foreign_user = User.objects.create_user(username="foreign-figure-owner")
        foreign_collection = Collection.objects.create(name="foreign-figure-collection")
        text = "Foreign child figure text."
        DocumentFigure.objects.bulk_create(
            [
                DocumentFigure(
                    title="Foreign child figure",
                    full_text=text,
                    full_text_hash=DocumentFigure.hash_fn(text),
                    collection=foreign_collection,
                    ingested_by=foreign_user,
                    image_file="document_figures/foreign.png",
                    parent_content_type=ContentType.objects.get_for_model(
                        RawTextDocument
                    ),
                    parent_object_id=parent.id,
                    parent_object_pkid=parent.pkid,
                    parent_raw_text_document=parent,
                )
            ]
        )
    elif hazard == "user_reference":
        group = Group.objects.create(name="foreign-fixture-user-reference")
        User.objects.get(username=VISIBLE_USERNAME).groups.add(group)
    else:
        foreign_collection = Collection.objects.create(
            name=f"foreign-principal-document-{hazard}"
        )
        visible = User.objects.get(username=VISIBLE_USERNAME)
        document_id = (
            next(iter(payload["documents"].values()))["document_id"]
            if hazard == "user_duplicate_uuid_foreign_collection"
            else uuid4()
        )
        text = f"Foreign principal document {hazard}."
        RawTextDocument.objects.bulk_create(
            [
                RawTextDocument(
                    id=document_id,
                    title="Foreign principal document",
                    full_text=text,
                    full_text_hash=RawTextDocument.hash_fn(text),
                    collection=foreign_collection,
                    ingested_by=visible,
                    ingestion_complete=True,
                )
            ]
        )
    before = _all_counts()
    with pytest.raises(CommandError, match="topology"):
        cleanup(manifest_path, payload)
    assert _all_counts() == before
    assert manifest_path.exists()
