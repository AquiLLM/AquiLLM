from django.core.validators import FileExtensionValidator
from django.db import models
from pypdf import PdfReader

from ..document import Document


class PDFDocument(Document):
    pdf_file = models.FileField(
        upload_to='pdfs/',
        max_length=500,
        validators=[FileExtensionValidator(['pdf'])]
    )
    zotero_item_key = models.CharField(
        max_length=100,
        null=True,
        blank=True,
        db_index=True,
        help_text="Zotero item key to prevent duplicate syncing"
    )

    class Meta:
        app_label = 'apps_documents'
        db_table = 'aquillm_pdfdocument'

    def save(self, *args, dont_rechunk=False, skip_text_extraction=False, **kwargs):
        if not skip_text_extraction and not dont_rechunk and not self.full_text:
            self.extract_text()
        super().save(*args, dont_rechunk=dont_rechunk, **kwargs)

    def extract_text(self):
        text = ""
        reader = PdfReader(self.pdf_file)
        for page in reader.pages:
            text += page.extract_text() + '\n'
        self.full_text = text.replace('\0', '')
