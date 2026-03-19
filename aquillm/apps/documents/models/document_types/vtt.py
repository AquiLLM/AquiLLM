from django.core.validators import FileExtensionValidator
from django.db import models

from ..document import Document


class VTTDocument(Document):
    audio_file = models.FileField(
        upload_to='stt_audio/',
        null=True,
        validators=[FileExtensionValidator(['mp4', 'ogg', 'opus', 'm4a', 'aac'])]
    )

    class Meta:
        app_label = 'apps_documents'
        db_table = 'aquillm_vttdocument'
