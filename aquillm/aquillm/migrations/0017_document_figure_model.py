import django.core.validators
import django.db.models.deletion
import uuid
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("aquillm", "0016_multimodal_ingestion_media_docs"),
        ("contenttypes", "0002_remove_content_type_name"),
    ]

    operations = [
        migrations.CreateModel(
            name="DocumentFigure",
            fields=[
                ("pkid", models.BigAutoField(editable=False, primary_key=True, serialize=False)),
                ("id", models.UUIDField(db_index=True, default=uuid.uuid4, editable=False)),
                ("title", models.CharField(max_length=200)),
                ("full_text", models.TextField()),
                ("full_text_hash", models.CharField(db_index=True, max_length=64)),
                ("ingestion_date", models.DateTimeField(auto_now_add=True)),
                ("ingestion_complete", models.BooleanField(default=True)),
                (
                    "image_file",
                    models.FileField(
                        upload_to="document_figures/",
                        validators=[
                            django.core.validators.FileExtensionValidator(
                                ["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp", "heic", "heif", "gif"]
                            )
                        ],
                    ),
                ),
                ("parent_object_id", models.UUIDField(blank=True, null=True)),
                (
                    "source_format",
                    models.CharField(
                        db_index=True,
                        help_text="Source format: pdf, docx, pptx, xlsx, ods, epub",
                        max_length=20,
                    ),
                ),
                (
                    "figure_index",
                    models.PositiveIntegerField(
                        default=0,
                        help_text="Index of this figure within the source document",
                    ),
                ),
                (
                    "extracted_caption",
                    models.TextField(
                        blank=True,
                        default="",
                        help_text="Caption text extracted from nearby content",
                    ),
                ),
                (
                    "location_metadata",
                    models.JSONField(
                        default=dict,
                        help_text="Format-specific location info (page_number, slide_number, etc.)",
                    ),
                ),
                ("source_content_type", models.CharField(blank=True, default="", max_length=150)),
                ("ocr_provider", models.CharField(blank=True, default="", max_length=64)),
                ("ocr_model", models.CharField(blank=True, default="", max_length=200)),
                (
                    "collection",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="%(class)s_documents",
                        to="aquillm.collection",
                    ),
                ),
                (
                    "ingested_by",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.RESTRICT,
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
                (
                    "parent_content_type",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        to="contenttypes.contenttype",
                    ),
                ),
            ],
            options={
                "ordering": ["source_format", "figure_index"],
            },
        ),
        migrations.AddIndex(
            model_name="documentfigure",
            index=models.Index(
                fields=["parent_content_type", "parent_object_id"],
                name="aquillm_doc_parent__a1b2c3_idx",
            ),
        ),
        migrations.AddIndex(
            model_name="documentfigure",
            index=models.Index(
                fields=["source_format"],
                name="aquillm_doc_source__d4e5f6_idx",
            ),
        ),
        migrations.AddConstraint(
            model_name="documentfigure",
            constraint=models.UniqueConstraint(
                fields=("collection", "full_text_hash"),
                name="documentfigure_document_collection_unique",
            ),
        ),
    ]
