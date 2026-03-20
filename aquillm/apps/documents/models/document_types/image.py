from django.core.validators import FileExtensionValidator
from django.db import models

from ..document import Document


IMAGE_UPLOAD_EXTENSIONS = [
    "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp", "heic", "heif", "gif"
]


class ImageUploadDocument(Document):
    image_file = models.FileField(
        upload_to="ingestion_images/",
        validators=[FileExtensionValidator(IMAGE_UPLOAD_EXTENSIONS)],
    )
    source_content_type = models.CharField(max_length=150, blank=True, default="")
    ocr_provider = models.CharField(max_length=64, blank=True, default="")
    ocr_model = models.CharField(max_length=200, blank=True, default="")

    class Meta:
        app_label = 'apps_documents'
        db_table = 'aquillm_imageuploaddocument'
