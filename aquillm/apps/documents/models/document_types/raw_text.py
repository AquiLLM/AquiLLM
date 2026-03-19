from django.db import models

from ..document import Document


class RawTextDocument(Document):
    source_url = models.URLField(max_length=2000, null=True, blank=True)

    class Meta:
        app_label = 'apps_documents'
        db_table = 'aquillm_rawtextdocument'
