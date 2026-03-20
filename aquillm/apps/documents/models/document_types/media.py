from django.core.validators import FileExtensionValidator
from django.db import models

from ..document import Document


MEDIA_UPLOAD_EXTENSIONS = [
    "mp3", "wav", "m4a", "aac", "flac", "ogg", "opus",
    "mp4", "mov", "m4v", "webm", "mkv", "avi", "mpeg", "mpg",
]


class MediaUploadDocument(Document):
    class MediaKind(models.TextChoices):
        AUDIO = "audio", "Audio"
        VIDEO = "video", "Video"

    media_file = models.FileField(
        upload_to="ingestion_media/",
        validators=[FileExtensionValidator(MEDIA_UPLOAD_EXTENSIONS)],
    )
    media_kind = models.CharField(
        max_length=16,
        choices=MediaKind.choices,
        default=MediaKind.AUDIO,
        db_index=True,
    )
    source_content_type = models.CharField(max_length=150, blank=True, default="")
    transcribe_provider = models.CharField(max_length=64, blank=True, default="")
    transcribe_model = models.CharField(max_length=200, blank=True, default="")

    class Meta:
        app_label = 'apps_documents'
        db_table = 'aquillm_mediauploaddocument'
