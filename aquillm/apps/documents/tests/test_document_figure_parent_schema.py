from __future__ import annotations

import importlib
import uuid

import pytest
from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import FieldDoesNotExist, ValidationError
from django.db import models


PARENT_FIELDS = {
    "HandwrittenNotesDocument": "parent_handwritten_notes_document",
    "ImageUploadDocument": "parent_image_upload_document",
    "MediaUploadDocument": "parent_media_upload_document",
    "PDFDocument": "parent_pdf_document",
    "RawTextDocument": "parent_raw_text_document",
    "TeXDocument": "parent_tex_document",
    "VTTDocument": "parent_vtt_document",
}


def test_each_supported_parent_has_a_db_backed_child_figure_relation():
    from apps.documents import models as document_models
    from apps.documents.models import Document, DocumentFigure

    with pytest.raises(FieldDoesNotExist):
        Document._meta.get_field("child_figures")

    for model_name, field_name in PARENT_FIELDS.items():
        parent_model = getattr(document_models, model_name)
        field = DocumentFigure._meta.get_field(field_name)
        relation = parent_model._meta.get_field("child_figures")

        assert isinstance(field, models.ForeignKey)
        assert field.null is True
        assert field.remote_field.model is parent_model
        assert field.remote_field.on_delete is models.CASCADE
        assert relation.field is field


def test_document_figure_does_not_inherit_a_self_referential_reverse_relation():
    from apps.documents.models import DocumentFigure

    with pytest.raises(FieldDoesNotExist):
        DocumentFigure._meta.get_field("child_figures")


def test_figure_parent_fields_must_be_all_set_or_all_null():
    from apps.documents.models import DocumentFigure

    constraint = next(
        constraint
        for constraint in DocumentFigure._meta.constraints
        if constraint.name == "documentfigure_parent_fields_paired"
    )

    assert isinstance(constraint, models.CheckConstraint)

    figure = DocumentFigure(parent_object_id=uuid.uuid4())
    with pytest.raises(ValidationError) as exc_info:
        figure.clean()

    assert set(exc_info.value.message_dict) == {
        "parent_content_type",
        "parent_object_id",
        "parent_object_pkid",
    }


def test_parent_property_stores_both_parent_keys_and_content_type(monkeypatch):
    from apps.documents.models import DocumentFigure, RawTextDocument

    content_type = ContentType(
        pk=41,
        app_label="apps_documents",
        model="rawtextdocument",
    )
    monkeypatch.setattr(
        ContentType.objects,
        "get_for_model",
        lambda value, for_concrete_model=False: content_type,
    )
    parent_id = uuid.uuid4()
    parent = RawTextDocument(pkid=29, id=parent_id)
    parent._state.adding = False
    figure = DocumentFigure()

    figure.parent_document = parent

    assert figure.parent_content_type is content_type
    assert figure.parent_object_pkid == 29
    assert figure.parent_object_id == parent_id
    assert figure.parent_raw_text_document is parent
    assert sum(
        getattr(figure, f"{field_name}_id") is not None
        for field_name in PARENT_FIELDS.values()
    ) == 1

    figure.parent_document = None
    assert figure.parent_content_type is None
    assert figure.parent_object_pkid is None
    assert figure.parent_object_id is None
    assert all(
        getattr(figure, f"{field_name}_id") is None
        for field_name in PARENT_FIELDS.values()
    )


def test_parent_property_requires_a_saved_document_instance():
    from apps.documents.models import DocumentFigure, RawTextDocument

    figure = DocumentFigure()

    with pytest.raises(ValueError, match="saved Document"):
        figure.parent_document = RawTextDocument(id=uuid.uuid4())

    with pytest.raises(ValueError, match="saved Document"):
        figure.parent_document = RawTextDocument(pkid=29, id=uuid.uuid4())

    with pytest.raises(TypeError, match="Document instance"):
        figure.parent_document = object()


def test_document_figure_cannot_be_used_as_another_figures_parent():
    from apps.documents.models import DocumentFigure

    parent = DocumentFigure(pkid=29, id=uuid.uuid4())
    parent._state.adding = False

    with pytest.raises(TypeError, match="supported parent document type"):
        DocumentFigure().parent_document = parent


def test_parent_property_resolves_only_a_coherent_typed_owner(monkeypatch):
    from apps.documents.models import DocumentFigure, RawTextDocument

    parent_id = uuid.uuid4()
    parent = RawTextDocument(pkid=29, id=parent_id)
    parent._state.adding = False
    content_type = ContentType(
        pk=41,
        app_label="apps_documents",
        model="rawtextdocument",
    )
    figure = DocumentFigure(
        parent_content_type=content_type,
        parent_object_pkid=29,
        parent_object_id=parent_id,
        parent_raw_text_document=parent,
    )

    assert figure.parent_document is parent

    figure.parent_object_id = uuid.uuid4()
    assert figure.parent_document is None


def test_parent_property_rejects_a_runtime_legacy_triple_without_typed_ownership():
    from apps.documents.models import DocumentFigure

    content_type = ContentType(
        pk=41,
        app_label="apps_documents",
        model="rawtextdocument",
    )
    figure = DocumentFigure(
        parent_content_type=content_type,
        parent_object_pkid=29,
        parent_object_id=uuid.UUID(int=29),
    )

    assert figure.parent_document is None
    with pytest.raises(ValidationError, match="typed parent ownership"):
        figure.clean()


def test_parent_validation_rejects_multiple_or_mismatched_typed_owners(monkeypatch):
    from apps.documents.models import DocumentFigure, PDFDocument, RawTextDocument

    content_type = ContentType(
        pk=41,
        app_label="apps_documents",
        model="rawtextdocument",
    )
    monkeypatch.setattr(
        ContentType.objects,
        "get_for_model",
        lambda value, for_concrete_model=False: content_type,
    )
    raw_parent = RawTextDocument(pkid=29, id=uuid.uuid4())
    raw_parent._state.adding = False
    pdf_parent = PDFDocument(pkid=31, id=uuid.uuid4())
    pdf_parent._state.adding = False
    figure = DocumentFigure()
    figure.parent_document = raw_parent
    figure.parent_pdf_document = pdf_parent

    with pytest.raises(ValidationError, match="exactly one typed parent"):
        figure.clean()

    figure.parent_pdf_document = None
    figure.parent_object_id = uuid.uuid4()
    with pytest.raises(ValidationError, match="logical UUID"):
        figure.clean()


def test_parent_ownership_constraint_ties_zero_or_one_owner_to_the_triple():
    from apps.documents.models import DocumentFigure

    constraint = next(
        constraint
        for constraint in DocumentFigure._meta.constraints
        if constraint.name == "documentfigure_parent_owner_coherent"
    )

    assert isinstance(constraint, models.CheckConstraint)


def test_parent_partial_save_expands_to_the_complete_typed_identity(monkeypatch):
    from apps.documents.models import Document, DocumentFigure, PDFDocument

    content_type = ContentType(
        pk=43,
        app_label="apps_documents",
        model="pdfdocument",
    )
    monkeypatch.setattr(
        ContentType.objects,
        "get_for_model",
        lambda value, for_concrete_model=False: content_type,
    )
    parent = PDFDocument(pkid=29, id=uuid.uuid4())
    parent._state.adding = False
    figure = DocumentFigure()
    figure.parent_document = parent
    calls = []
    monkeypatch.setattr(
        Document,
        "save",
        lambda self, *args, **kwargs: calls.append(kwargs),
    )

    figure.save(
        dont_rechunk=True,
        update_fields=(field for field in ("parent_object_id", "source_format")),
    )

    assert set(calls[0]["update_fields"]) == {
        "parent_content_type",
        "parent_object_pkid",
        "parent_object_id",
        *PARENT_FIELDS.values(),
        "source_format",
    }


def test_parent_identity_bulk_mutations_are_rejected_before_the_database():
    from apps.documents.models import DocumentFigure

    figure = DocumentFigure(pkid=41)
    with pytest.raises(ValidationError, match="parent ownership"):
        DocumentFigure.objects.all().update(parent_object_id=uuid.uuid4())
    with pytest.raises(ValidationError, match="parent ownership"):
        DocumentFigure.objects.bulk_update(
            [figure],
            ["parent_raw_text_document"],
        )
    with pytest.raises(ValidationError, match="typed parent ownership"):
        DocumentFigure.objects.bulk_create(
            [
                DocumentFigure(
                    parent_content_type_id=41,
                    parent_object_pkid=29,
                    parent_object_id=uuid.uuid4(),
                )
            ]
        )
    with pytest.raises(ValidationError, match="parent ownership"):
        DocumentFigure.objects.bulk_create(
            [DocumentFigure(pkid=41)],
            update_conflicts=True,
            unique_fields=["pkid"],
            update_fields=["parent_object_id"],
        )


class _ResolverQuery:
    def __init__(self, rows, calls):
        self.rows = rows
        self.calls = calls

    def using(self, alias):
        self.calls.append(("using", alias))
        return self

    def filter(self, **kwargs):
        self.calls.append(("filter", kwargs))
        if "id" in kwargs:
            rows = [row for row in self.rows if row[1] == kwargs["id"]]
        elif "id__in" in kwargs:
            rows = [row for row in self.rows if row[1] in kwargs["id__in"]]
        elif "pkid__in" in kwargs:
            rows = [row for row in self.rows if row[0] in kwargs["pkid__in"]]
        else:
            rows = [row for row in self.rows if row[0] == kwargs["pkid"]]
        return _ResolverQuery(rows, self.calls)

    def values_list(self, *_fields):
        return self

    def __getitem__(self, item):
        return self.rows[item]

    def __iter__(self):
        return iter(self.rows)

    def first(self):
        return self.rows[0] if self.rows else None


def test_parent_migration_batches_exact_and_legacy_resolution():
    migration = importlib.import_module(
        "apps.documents.migrations.0004_documentfigure_concrete_parent_key"
    )
    exact_id = uuid.uuid4()
    legacy_id = uuid.UUID(int=29)
    legacy_logical_id = uuid.uuid4()
    calls = []
    parent_model = type(
        "ParentModel",
        (),
        {
            "_base_manager": _ResolverQuery(
                [(87, exact_id), (29, legacy_logical_id)],
                calls,
            )
        },
    )

    resolved = migration._resolve_parent_rows(
        parent_model,
        {exact_id, legacy_id},
        using="archive",
    )

    assert resolved == {
        exact_id: (87, exact_id),
        legacy_id: (29, legacy_logical_id),
    }
    assert calls == [
        ("using", "archive"),
        ("filter", {"id__in": {exact_id, legacy_id}}),
        ("using", "archive"),
        ("filter", {"pkid__in": {29}}),
    ]


def test_parent_migration_fails_closed_on_duplicate_logical_uuid():
    migration = importlib.import_module(
        "apps.documents.migrations.0004_documentfigure_concrete_parent_key"
    )
    duplicate_id = uuid.UUID(int=29)
    calls = []
    parent_model = type(
        "ParentModel",
        (),
        {
            "_base_manager": _ResolverQuery(
                [
                    (87, duplicate_id),
                    (88, duplicate_id),
                    (29, uuid.uuid4()),
                ],
                calls,
            )
        },
    )

    resolved = migration._resolve_parent_rows(
        parent_model,
        {duplicate_id},
        using="default",
    )

    assert resolved == {}
    assert calls == [
        ("using", "default"),
        ("filter", {"id__in": {duplicate_id}}),
    ]


def test_parent_migration_binds_the_exact_typed_owner_field():
    migration = importlib.import_module(
        "apps.documents.migrations.0004_documentfigure_concrete_parent_key"
    )
    parent_id = uuid.uuid4()
    calls = []
    parent_model = type(
        "HistoricalRawTextDocument",
        (),
        {
            "_meta": type("Meta", (), {"model_name": "rawtextdocument"})(),
            "_base_manager": _ResolverQuery([(29, parent_id)], calls),
        },
    )
    figure = type(
        "HistoricalFigure",
        (),
        {
            "parent_content_type_id": 41,
            "parent_object_id": parent_id,
            "parent_object_pkid": None,
        },
    )()
    updates = []

    class FigureManager:
        def using(self, alias):
            assert alias == "default"
            return self

        def bulk_update(self, figures, fields):
            updates.append((tuple(figures), tuple(fields)))

    historical_figure = type(
        "HistoricalDocumentFigure",
        (),
        {"_base_manager": FigureManager()},
    )

    migration._backfill_figure_batch(
        historical_figure,
        [figure],
        {41: parent_model},
        using="default",
    )

    assert figure.parent_raw_text_document_id == 29
    assert figure.parent_object_pkid == 29
    assert figure.parent_object_id == parent_id
    assert all(
        getattr(figure, f"{field_name}_id") is None
        for field_name in migration._OWNER_FIELDS
        if field_name != "parent_raw_text_document"
    )
    assert updates == [((figure,), migration._UPDATE_FIELDS)]
