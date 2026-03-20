from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType
from django.core.validators import FileExtensionValidator
from django.db import models

from ..document import Document
from .image import IMAGE_UPLOAD_EXTENSIONS


class DocumentFigure(Document):
    """
    Figure/image extracted from any document format.
    Uses GenericForeignKey to link to any parent document type.
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
    parent_document = GenericForeignKey('parent_content_type', 'parent_object_id')
    
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
        indexes = [
            models.Index(fields=['parent_content_type', 'parent_object_id']),
            models.Index(fields=['source_format']),
        ]
