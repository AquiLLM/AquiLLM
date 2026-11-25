"""Replaceable search metadata for one current PDF owner."""

from django.db import models


class PDFNamedItemIndex(models.Model):
    document = models.OneToOneField(
        "apps_documents.PDFDocument",
        on_delete=models.CASCADE,
        primary_key=True,
        related_name="+",
    )
    generation = models.ForeignKey(
        "apps_documents.DocumentRepresentationGeneration",
        null=True,
        on_delete=models.CASCADE,
        related_name="+",
    )
    member_digest = models.CharField(max_length=64)
    binding = models.CharField(max_length=64)
    payload = models.TextField()

    class Meta:
        app_label = "apps_documents"
        db_table = "aquillm_pdf_named_item_index"
