from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import ValidationError
from django.core.validators import FileExtensionValidator
from django.db import models

from ..document import Document
from .image import IMAGE_UPLOAD_EXTENSIONS


_SUPPORTED_PARENT_MODEL_LABELS = frozenset(
    {
        "apps_documents.handwrittennotesdocument",
        "apps_documents.imageuploaddocument",
        "apps_documents.mediauploaddocument",
        "apps_documents.pdfdocument",
        "apps_documents.rawtextdocument",
        "apps_documents.texdocument",
        "apps_documents.vttdocument",
    }
)

_PARENT_FIELD_BY_MODEL_LABEL = {
    "apps_documents.handwrittennotesdocument": (
        "parent_handwritten_notes_document"
    ),
    "apps_documents.imageuploaddocument": "parent_image_upload_document",
    "apps_documents.mediauploaddocument": "parent_media_upload_document",
    "apps_documents.pdfdocument": "parent_pdf_document",
    "apps_documents.rawtextdocument": "parent_raw_text_document",
    "apps_documents.texdocument": "parent_tex_document",
    "apps_documents.vttdocument": "parent_vtt_document",
}
_PARENT_OWNER_FIELDS = tuple(_PARENT_FIELD_BY_MODEL_LABEL.values())
_PARENT_PROVENANCE_FIELDS = (
    "parent_content_type",
    "parent_object_pkid",
    "parent_object_id",
)
_PARENT_PERSISTENCE_FIELDS = (*_PARENT_PROVENANCE_FIELDS, *_PARENT_OWNER_FIELDS)
_PARENT_WRITE_NAMES = frozenset(
    (
        *_PARENT_PERSISTENCE_FIELDS,
        *(f"{field_name}_id" for field_name in _PARENT_OWNER_FIELDS),
        "parent_content_type_id",
    )
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
        **{f"{field_name}__isnull": True for field_name in _PARENT_OWNER_FIELDS},
    )
    for selected_field in _PARENT_OWNER_FIELDS:
        owner_presence = {
            f"{field_name}__isnull": field_name != selected_field
            for field_name in _PARENT_OWNER_FIELDS
        }
        condition |= (
            models.Q(**provenance_present, **owner_presence)
            & models.Q(parent_object_pkid=models.F(selected_field))
        )
    return condition


class DocumentFigureQuerySet(models.QuerySet):
    def update(self, **kwargs):
        if _PARENT_WRITE_NAMES.intersection(kwargs):
            raise ValidationError(
                "Figure parent ownership must be changed through parent_document."
            )
        return super().update(**kwargs)

    def bulk_update(self, objs, fields, batch_size=None):
        if _PARENT_WRITE_NAMES.intersection(fields):
            raise ValidationError(
                "Figure parent ownership cannot be changed with bulk_update."
            )
        return super().bulk_update(objs, fields, batch_size=batch_size)

    def bulk_create(self, objs, **kwargs):
        objs = tuple(objs)
        update_fields = tuple(kwargs.get("update_fields") or ())
        if _PARENT_WRITE_NAMES.intersection(update_fields):
            raise ValidationError(
                "Figure parent ownership cannot be changed with a bulk upsert."
            )
        for figure in objs:
            figure._validate_parent_ownership()
        return super().bulk_create(objs, **kwargs)


class DocumentFigure(Document):
    """
    Figure/image extracted from any document format.

    The logical UUID is retained for API/provenance use while the concrete
    primary key powers Django's reverse generic relation and deletion
    collector.  ``parent_document`` deliberately resolves both values so a
    mismatched pair cannot point at the wrong document.
    """
    image_file = models.FileField(
        upload_to="document_figures/",
        validators=[FileExtensionValidator(IMAGE_UPLOAD_EXTENSIONS)],
    )
    
    parent_content_type = models.ForeignKey(
        ContentType,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
    )
    parent_object_id = models.UUIDField(null=True, blank=True)
    parent_object_pkid = models.BigIntegerField(null=True, blank=True)
    parent_handwritten_notes_document = models.ForeignKey(
        "HandwrittenNotesDocument",
        on_delete=models.CASCADE,
        related_name="child_figures",
        null=True,
        blank=True,
    )
    parent_image_upload_document = models.ForeignKey(
        "ImageUploadDocument",
        on_delete=models.CASCADE,
        related_name="child_figures",
        null=True,
        blank=True,
    )
    parent_media_upload_document = models.ForeignKey(
        "MediaUploadDocument",
        on_delete=models.CASCADE,
        related_name="child_figures",
        null=True,
        blank=True,
    )
    parent_pdf_document = models.ForeignKey(
        "PDFDocument",
        on_delete=models.CASCADE,
        related_name="child_figures",
        null=True,
        blank=True,
    )
    parent_raw_text_document = models.ForeignKey(
        "RawTextDocument",
        on_delete=models.CASCADE,
        related_name="child_figures",
        null=True,
        blank=True,
    )
    parent_tex_document = models.ForeignKey(
        "TeXDocument",
        on_delete=models.CASCADE,
        related_name="child_figures",
        null=True,
        blank=True,
    )
    parent_vtt_document = models.ForeignKey(
        "VTTDocument",
        on_delete=models.CASCADE,
        related_name="child_figures",
        null=True,
        blank=True,
    )
    objects = DocumentFigureQuerySet.as_manager()

    def _typed_parent_fields(self):
        return tuple(
            field_name
            for field_name in _PARENT_OWNER_FIELDS
            if getattr(self, f"{field_name}_id") is not None
        )

    @property
    def parent_document(self):
        """Return the exact DB-owned parent only when provenance is coherent."""
        owner_fields = self._typed_parent_fields()
        if len(owner_fields) != 1:
            return None
        field_name = owner_fields[0]
        parent = getattr(self, field_name)
        if parent is None or not self._is_supported_parent_document_model(type(parent)):
            return None
        if _PARENT_FIELD_BY_MODEL_LABEL[type(parent)._meta.label_lower] != field_name:
            return None
        if (
            self.parent_content_type_id is None
            or self.parent_object_pkid != parent.pkid
            or self.parent_object_id != parent.id
        ):
            return None
        content_type = self.parent_content_type
        if (
            content_type.app_label != parent._meta.app_label
            or content_type.model != parent._meta.model_name
        ):
            return None
        return parent

    @parent_document.setter
    def parent_document(self, value):
        for field_name in _PARENT_OWNER_FIELDS:
            setattr(self, field_name, None)
        if value is None:
            self.parent_content_type = None
            self.parent_object_pkid = None
            self.parent_object_id = None
            return

        if not isinstance(value, Document):
            raise TypeError("parent_document must be a Document instance")
        if not self._is_supported_parent_document_model(type(value)):
            raise TypeError("parent_document must use a supported parent document type")
        if (
            value._state.adding
            or value.pkid is None
            or value.id is None
        ):
            raise ValueError("parent_document must be a saved Document")

        self.parent_content_type = ContentType.objects.get_for_model(
            value,
            for_concrete_model=False,
        )
        self.parent_object_pkid = value.pkid
        self.parent_object_id = value.id
        setattr(
            self,
            _PARENT_FIELD_BY_MODEL_LABEL[value._meta.label_lower],
            value,
        )

    @staticmethod
    def _is_supported_parent_document_model(model) -> bool:
        return (
            isinstance(model, type)
            and issubclass(model, Document)
            and not model._meta.abstract
            and model._meta.label_lower in _SUPPORTED_PARENT_MODEL_LABELS
        )

    def clean(self):
        super().clean()
        self._validate_parent_ownership()

    def _validate_parent_ownership(self):
        owner_fields = self._typed_parent_fields()
        parent_fields = {
            "parent_content_type": self.parent_content_type_id,
            "parent_object_pkid": self.parent_object_pkid,
            "parent_object_id": self.parent_object_id,
        }
        populated = [value is not None for value in parent_fields.values()]
        if not owner_fields:
            if not any(populated):
                return
            if not all(populated):
                message = (
                    "Parent content type, concrete key, and logical UUID "
                    "must be set together."
                )
                raise ValidationError({field: message for field in parent_fields})
            if all(populated):
                raise ValidationError(
                    {
                        "parent_document": (
                            "Parent provenance requires typed parent ownership."
                        )
                    }
                )
        if len(owner_fields) != 1:
            raise ValidationError(
                {"parent_document": "exactly one typed parent owner is required."}
            )
        if not all(populated):
            message = "Parent content type, concrete key, and logical UUID must be set together."
            raise ValidationError({field: message for field in parent_fields})

        owner_field = owner_fields[0]
        parent = getattr(self, owner_field)
        parent_model = type(parent)
        if (
            not self._is_supported_parent_document_model(parent_model)
            or _PARENT_FIELD_BY_MODEL_LABEL[parent_model._meta.label_lower]
            != owner_field
        ):
            raise ValidationError(
                {"parent_content_type": "Parent must be a concrete Document model."}
            )
        content_type = self.parent_content_type
        if (
            content_type.app_label != parent_model._meta.app_label
            or content_type.model != parent_model._meta.model_name
        ):
            raise ValidationError(
                {"parent_content_type": "Parent content type must match typed owner."}
            )
        if self.parent_object_pkid != parent.pkid:
            raise ValidationError(
                {"parent_object_pkid": "Parent concrete key must match typed owner."}
            )
        if self.parent_object_id != parent.id:
            message = "Parent logical UUID must match typed owner."
            raise ValidationError(
                {
                    "parent_object_id": message,
                }
            )

    def save(self, *args, **kwargs):
        update_fields = kwargs.get("update_fields")
        if update_fields is not None:
            requested = tuple(update_fields)
            if _PARENT_WRITE_NAMES.intersection(requested):
                requested = tuple(
                    sorted(set(requested).union(_PARENT_PERSISTENCE_FIELDS))
                )
            kwargs["update_fields"] = requested
        self._validate_parent_ownership()
        return super().save(*args, **kwargs)
    
    source_format = models.CharField(
        max_length=20,
        db_index=True,
        help_text="Source format: pdf, docx, pptx, xlsx, ods, epub"
    )
    figure_index = models.PositiveIntegerField(
        default=0,
        help_text="Index of this figure within the source document"
    )
    extracted_caption = models.TextField(
        blank=True,
        default="",
        help_text="Caption text extracted from nearby content"
    )
    location_metadata = models.JSONField(
        default=dict,
        help_text="Format-specific location info (page_number, slide_number, etc.)"
    )
    
    source_content_type = models.CharField(max_length=150, blank=True, default="")
    ocr_provider = models.CharField(max_length=64, blank=True, default="")
    ocr_model = models.CharField(max_length=200, blank=True, default="")

    class Meta:
        app_label = 'apps_documents'
        db_table = 'aquillm_documentfigure'
        ordering = ['source_format', 'figure_index']
        constraints = [
            models.CheckConstraint(
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
            models.CheckConstraint(
                condition=_parent_owner_constraint_condition(),
                name="documentfigure_parent_owner_coherent",
            ),
        ]
        indexes = [
            models.Index(fields=['parent_content_type', 'parent_object_id']),
            models.Index(
                fields=["parent_content_type", "parent_object_pkid"],
                name="docfigure_parent_pkid_idx",
            ),
            models.Index(fields=['source_format']),
        ]
