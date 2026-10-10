"""Keep the app split's model metadata faithful to the existing document tables."""

from datetime import timedelta

import pytest
from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.db.migrations.loader import MigrationLoader
from django.utils import timezone

from apps.collections.models import Collection
from apps.documents.models import DocumentFigure, TextChunk
from apps.documents.models.document_types import DESCENDED_FROM_DOCUMENT


@pytest.fixture
def document_fields(db):
    return {
        "collection": Collection.objects.create(name="Metadata regression"),
        "ingested_by": get_user_model().objects.create_user(username="metadata-owner"),
        "title": "Synthetic document",
        "full_text": "Synthetic document content",
        "full_text_hash": "same-content-hash",
        "ingestion_complete": False,
    }


@pytest.mark.django_db
@pytest.mark.parametrize("model", DESCENDED_FROM_DOCUMENT, ids=lambda model: model.__name__)
def test_document_validation_rejects_content_already_in_the_collection(model, document_fields):
    # Bulk insertion avoids extraction/publication side effects, preserving the
    # real table constraints and Django's real model-validation query.
    model.objects.bulk_create([model(**document_fields)])
    duplicate = model(**document_fields)

    with pytest.raises(ValidationError, match="already exists"):
        duplicate.validate_constraints()


@pytest.mark.django_db
@pytest.mark.parametrize(
    "model",
    [model for model in DESCENDED_FROM_DOCUMENT if model is not DocumentFigure],
    ids=lambda model: model.__name__,
)
def test_documents_order_newest_first_then_title(model, document_fields):
    documents = model.objects.bulk_create([
        model(**{**document_fields, "title": title, "full_text_hash": title})
        for title in ("A older", "Zulu recent", "Alpha recent")
    ])
    recent = timezone.now()
    selected = model.objects.filter(pkid__in=[document.pkid for document in documents])
    selected.update(ingestion_date=recent)
    selected.filter(pkid=documents[0].pkid).update(ingestion_date=recent - timedelta(days=1))

    assert list(selected.values_list("title", flat=True)) == [
        "Alpha recent", "Zulu recent", "A older"
    ]


@pytest.mark.parametrize("model,fields", [
    (DocumentFigure, ["parent_content_type", "parent_object_id"]),
    (DocumentFigure, ["source_format"]),
    (TextChunk, ["doc_id", "start_position", "end_position"]),
])
def test_index_metadata_preserves_the_legacy_physical_name(model, fields):
    loader = MigrationLoader(None)
    legacy = loader.project_state([("aquillm", "0017_document_figure_model")])
    current = loader.project_state()
    legacy_index = next(index for index in legacy.models[
        ("aquillm", model._meta.model_name)
    ].options["indexes"] if index.fields == fields)
    current_index = next(index for index in current.models[
        ("apps_documents", model._meta.model_name)
    ].options["indexes"] if index.fields == fields)
    runtime_index = next(index for index in model._meta.indexes if index.fields == fields)

    assert current_index.name == legacy_index.name
    assert runtime_index.name == legacy_index.name


def test_migration_field_metadata_matches_runtime_serialization():
    state = MigrationLoader(None).project_state()
    for model, field_name in [
        *((model, "collection") for model in DESCENDED_FROM_DOCUMENT),
        (TextChunk, "doc_id"),
    ]:
        migrated = state.models[("apps_documents", model._meta.model_name)].fields[field_name]
        live = model._meta.get_field(field_name)
        assert migrated.deconstruct()[1:] == live.deconstruct()[1:]
