from django.db import models

from ..document import Document


class RawTextDocument(Document):
    source_url = models.URLField(max_length=2000, null=True, blank=True)
    # Populated by the web crawler when each visited URL is captured as a PDF
    # so the citation modal can reuse the PDF highlight UX for web content.
    rendered_pdf = models.FileField(upload_to='crawled_pdfs/', null=True, blank=True)

    class Meta:
        app_label = 'apps_documents'
        db_table = 'aquillm_rawtextdocument'
