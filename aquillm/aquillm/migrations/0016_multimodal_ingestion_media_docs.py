import django.core.validators
import django.db.models.deletion
import uuid
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("aquillm", "0015_unified_ingestion_status"),
    ]

    operations = [
        migrations.CreateModel(
            name="ImageUploadDocument",
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
                        upload_to="ingestion_images/",
                        validators=[
                            django.core.validators.FileExtensionValidator(
                                ["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp", "heic", "heif", "gif"]
                            )
                        ],
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
                    models.ForeignKey(on_delete=django.db.models.deletion.RESTRICT, to=settings.AUTH_USER_MODEL),
                ),
            ],
            options={
                "ordering": ["-ingestion_date", "title"],
                "abstract": False,
            },
        ),
        migrations.CreateModel(
            name="MediaUploadDocument",
            fields=[
                ("pkid", models.BigAutoField(editable=False, primary_key=True, serialize=False)),
                ("id", models.UUIDField(db_index=True, default=uuid.uuid4, editable=False)),
                ("title", models.CharField(max_length=200)),
                ("full_text", models.TextField()),
                ("full_text_hash", models.CharField(db_index=True, max_length=64)),
                ("ingestion_date", models.DateTimeField(auto_now_add=True)),
                ("ingestion_complete", models.BooleanField(default=True)),
                (
                    "media_file",
                    models.FileField(
                        upload_to="ingestion_media/",
                        validators=[
                            django.core.validators.FileExtensionValidator(
                                [
                                    "mp3",
                                    "wav",
                                    "m4a",
                                    "aac",
                                    "flac",
                                    "ogg",
                                    "opus",
                                    "mp4",
                                    "mov",
                                    "m4v",
                                    "webm",
                                    "mkv",
                                    "avi",
                                    "mpeg",
                                    "mpg",
                                ]
                            )
                        ],
                    ),
                ),
                (
                    "media_kind",
                    models.CharField(
                        choices=[("audio", "Audio"), ("video", "Video")],
                        db_index=True,
                        default="audio",
                        max_length=16,
                    ),
                ),
                ("source_content_type", models.CharField(blank=True, default="", max_length=150)),
                ("transcribe_provider", models.CharField(blank=True, default="", max_length=64)),
                ("transcribe_model", models.CharField(blank=True, default="", max_length=200)),
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
                    models.ForeignKey(on_delete=django.db.models.deletion.RESTRICT, to=settings.AUTH_USER_MODEL),
                ),
            ],
            options={
                "ordering": ["-ingestion_date", "title"],
                "abstract": False,
            },
        ),
        migrations.AddConstraint(
            model_name="imageuploaddocument",
            constraint=models.UniqueConstraint(
                fields=("collection", "full_text_hash"),
                name="imageuploaddocument_document_collection_unique",
            ),
        ),
        migrations.AddConstraint(
            model_name="mediauploaddocument",
            constraint=models.UniqueConstraint(
                fields=("collection", "full_text_hash"),
                name="mediauploaddocument_document_collection_unique",
            ),
        ),
    ]
