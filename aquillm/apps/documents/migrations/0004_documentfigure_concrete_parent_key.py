import uuid

import django.db.models.deletion
from django.db import migrations, models


_DOCUMENT_MODEL_NAMES = frozenset(
    {
        "handwrittennotesdocument",
        "imageuploaddocument",
        "mediauploaddocument",
        "pdfdocument",
        "rawtextdocument",
        "texdocument",
        "vttdocument",
    }
)
_MAX_BIGINT = 2**63 - 1
_OWNER_FIELD_BY_MODEL_NAME = {
    "handwrittennotesdocument": "parent_handwritten_notes_document",
    "imageuploaddocument": "parent_image_upload_document",
    "mediauploaddocument": "parent_media_upload_document",
    "pdfdocument": "parent_pdf_document",
    "rawtextdocument": "parent_raw_text_document",
    "texdocument": "parent_tex_document",
    "vttdocument": "parent_vtt_document",
}
_OWNER_FIELDS = tuple(_OWNER_FIELD_BY_MODEL_NAME.values())
_UPDATE_FIELDS = (
    "parent_content_type",
    "parent_object_pkid",
    "parent_object_id",
    *_OWNER_FIELDS,
)


def _parent_owner_constraint_condition():
    provenance_null = {
        "parent_content_type__isnull": True,
        "parent_object_pkid__isnull": True,
        "parent_object_id__isnull": True,
    }
    provenance_present = {
        "parent_content_type__isnull": False,
        "parent_object_pkid__isnull": False,
        "parent_object_id__isnull": False,
    }
    condition = models.Q(
        **provenance_null,
        **{f"{field_name}__isnull": True for field_name in _OWNER_FIELDS},
    )
    for selected_field in _OWNER_FIELDS:
        owner_presence = {
            f"{field_name}__isnull": field_name != selected_field
            for field_name in _OWNER_FIELDS
        }
        condition |= (
            models.Q(**provenance_present, **owner_presence)
            & models.Q(parent_object_pkid=models.F(selected_field))
        )
    return condition


def _document_model_for_content_type(apps, content_type):
    if (
        content_type is None
        or content_type.app_label != "apps_documents"
        or content_type.model not in _DOCUMENT_MODEL_NAMES
    ):
        return None
    try:
        return apps.get_model(content_type.app_label, content_type.model)
    except LookupError:
        return None


def _resolve_parent_rows(parent_model, stored_ids, *, using):
    """Resolve logical UUIDs in bulk, then unambiguous legacy integer keys."""
    stored_ids = set(stored_ids)
    if not stored_ids:
        return {}

    exact_by_id = {}
    exact_rows = (
        parent_model._base_manager.using(using)
        .filter(id__in=stored_ids)
        .values_list("pkid", "id")
    )
    for parent_pkid, logical_id in exact_rows:
        exact_by_id.setdefault(logical_id, []).append((parent_pkid, logical_id))

    resolved = {}
    legacy_pkids = {}
    for stored_id in stored_ids:
        exact_matches = exact_by_id.get(stored_id, ())
        if len(exact_matches) == 1:
            resolved[stored_id] = exact_matches[0]
            continue
        if exact_matches:
            # Duplicate logical UUIDs are ambiguous. Never reinterpret them as
            # the legacy UUID-encoded integer form.
            continue

        legacy_pkid = stored_id.int
        if 0 < legacy_pkid <= _MAX_BIGINT:
            legacy_pkids[stored_id] = legacy_pkid

    if not legacy_pkids:
        return resolved

    legacy_rows = {
        parent_pkid: (parent_pkid, logical_id)
        for parent_pkid, logical_id in (
            parent_model._base_manager.using(using)
            .filter(pkid__in=set(legacy_pkids.values()))
            .values_list("pkid", "id")
        )
    }
    for stored_id, legacy_pkid in legacy_pkids.items():
        legacy_match = legacy_rows.get(legacy_pkid)
        if legacy_match is not None:
            resolved[stored_id] = legacy_match
    return resolved


def _backfill_figure_batch(DocumentFigure, figures, parent_models, *, using):
    stored_ids_by_model = {}
    for figure in figures:
        for owner_field in _OWNER_FIELDS:
            setattr(figure, f"{owner_field}_id", None)
        parent_model = parent_models.get(figure.parent_content_type_id)
        if parent_model is not None and figure.parent_object_id is not None:
            stored_ids_by_model.setdefault(parent_model, set()).add(
                figure.parent_object_id
            )

    resolved_by_model = {
        parent_model: _resolve_parent_rows(
            parent_model,
            stored_ids,
            using=using,
        )
        for parent_model, stored_ids in stored_ids_by_model.items()
    }

    for figure in figures:
        parent_model = parent_models.get(figure.parent_content_type_id)
        resolved = None
        if parent_model is not None and figure.parent_object_id is not None:
            resolved = resolved_by_model[parent_model].get(figure.parent_object_id)

        if resolved is None:
            figure.parent_content_type_id = None
            figure.parent_object_pkid = None
            figure.parent_object_id = None
        else:
            parent_pkid, logical_id = resolved
            figure.parent_object_pkid = parent_pkid
            figure.parent_object_id = logical_id
            owner_field = _OWNER_FIELD_BY_MODEL_NAME[parent_model._meta.model_name]
            setattr(figure, f"{owner_field}_id", parent_pkid)

    DocumentFigure._base_manager.using(using).bulk_update(
        figures,
        _UPDATE_FIELDS,
    )


def backfill_concrete_parent_keys(apps, schema_editor):
    DocumentFigure = apps.get_model("apps_documents", "DocumentFigure")
    ContentType = apps.get_model("contenttypes", "ContentType")
    using = schema_editor.connection.alias

    content_type_ids = set(
        DocumentFigure._base_manager.using(using)
        .exclude(parent_content_type_id=None)
        .values_list("parent_content_type_id", flat=True)
        .distinct()
    )
    if len(content_type_ids) > len(_DOCUMENT_MODEL_NAMES):
        raise RuntimeError("Figure parent content-type scope is not supported.")
    content_types = {
        content_type.pk: content_type
        for content_type in ContentType._base_manager.using(using).filter(
            pk__in=content_type_ids
        )
    }
    parent_models = {
        content_type_id: _document_model_for_content_type(apps, content_type)
        for content_type_id, content_type in content_types.items()
    }

    pending_updates = []
    figures = (
        DocumentFigure._base_manager.using(using)
        .all()
        .only("pkid", "parent_content_type_id", "parent_object_id")
        .iterator(chunk_size=500)
    )
    for figure in figures:
        pending_updates.append(figure)

        if len(pending_updates) == 500:
            _backfill_figure_batch(
                DocumentFigure,
                pending_updates,
                parent_models,
                using=using,
            )
            pending_updates.clear()

    if pending_updates:
        _backfill_figure_batch(
            DocumentFigure,
            pending_updates,
            parent_models,
            using=using,
        )


def restore_legacy_parent_keys(apps, schema_editor):
    DocumentFigure = apps.get_model("apps_documents", "DocumentFigure")
    using = schema_editor.connection.alias

    pending_updates = []
    figures = (
        DocumentFigure._base_manager.using(using)
        .all()
        .only("pkid", "parent_content_type_id", "parent_object_pkid")
        .iterator(chunk_size=500)
    )
    for figure in figures:
        parent_pkid = figure.parent_object_pkid
        if (
            figure.parent_content_type_id is None
            or parent_pkid is None
            or not 0 < parent_pkid <= _MAX_BIGINT
        ):
            figure.parent_content_type_id = None
            figure.parent_object_id = None
        else:
            figure.parent_object_id = uuid.UUID(int=parent_pkid)
        pending_updates.append(figure)

        if len(pending_updates) == 500:
            DocumentFigure._base_manager.using(using).bulk_update(
                pending_updates,
                ("parent_content_type", "parent_object_id"),
            )
            pending_updates.clear()

    if pending_updates:
        DocumentFigure._base_manager.using(using).bulk_update(
            pending_updates,
            ("parent_content_type", "parent_object_id"),
        )


class Migration(migrations.Migration):
    dependencies = [
        ("apps_documents", "0003_document_source_url"),
    ]

    operations = [
        migrations.AddField(
            model_name="documentfigure",
            name="parent_object_pkid",
            field=models.BigIntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="documentfigure",
            name="parent_handwritten_notes_document",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="child_figures",
                to="apps_documents.handwrittennotesdocument",
            ),
        ),
        migrations.AddField(
            model_name="documentfigure",
            name="parent_image_upload_document",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="child_figures",
                to="apps_documents.imageuploaddocument",
            ),
        ),
        migrations.AddField(
            model_name="documentfigure",
            name="parent_media_upload_document",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="child_figures",
                to="apps_documents.mediauploaddocument",
            ),
        ),
        migrations.AddField(
            model_name="documentfigure",
            name="parent_pdf_document",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="child_figures",
                to="apps_documents.pdfdocument",
            ),
        ),
        migrations.AddField(
            model_name="documentfigure",
            name="parent_raw_text_document",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="child_figures",
                to="apps_documents.rawtextdocument",
            ),
        ),
        migrations.AddField(
            model_name="documentfigure",
            name="parent_tex_document",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="child_figures",
                to="apps_documents.texdocument",
            ),
        ),
        migrations.AddField(
            model_name="documentfigure",
            name="parent_vtt_document",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="child_figures",
                to="apps_documents.vttdocument",
            ),
        ),
        migrations.RunPython(
            backfill_concrete_parent_keys,
            restore_legacy_parent_keys,
        ),
        migrations.AddIndex(
            model_name="documentfigure",
            index=models.Index(
                fields=["parent_content_type", "parent_object_pkid"],
                name="docfigure_parent_pkid_idx",
            ),
        ),
        migrations.AddConstraint(
            model_name="documentfigure",
            constraint=models.CheckConstraint(
                condition=(
                    models.Q(
                        parent_content_type__isnull=True,
                        parent_object_pkid__isnull=True,
                        parent_object_id__isnull=True,
                    )
                    | models.Q(
                        parent_content_type__isnull=False,
                        parent_object_pkid__isnull=False,
                        parent_object_id__isnull=False,
                    )
                ),
                name="documentfigure_parent_fields_paired",
            ),
        ),
        migrations.AddConstraint(
            model_name="documentfigure",
            constraint=models.CheckConstraint(
                condition=_parent_owner_constraint_condition(),
                name="documentfigure_parent_owner_coherent",
            ),
        ),
    ]
