from django.db import models

from ..document import Document


class TeXDocument(Document):
    pdf_file = models.FileField(upload_to='pdfs/', null=True)

    class Meta:
        app_label = 'apps_documents'
        db_table = 'aquillm_texdocument'
