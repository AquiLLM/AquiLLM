"""Ingestion batch models."""
from django.contrib.auth.models import User
from django.db import models
from django.utils import timezone
from uuid import uuid4


class IngestionBatch(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='ingestion_batches')
    collection = models.ForeignKey(
        'apps_collections.Collection',
        on_delete=models.CASCADE,
        related_name='ingestion_batches'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = 'apps_ingestion'
        db_table = 'aquillm_ingestionbatch'
        ordering = ['-created_at']


class IngestionBatchItem(models.Model):
    class Status(models.TextChoices):
        QUEUED = "queued", "Queued"
        PROCESSING = "processing", "Processing"
        SUCCESS = "success", "Success"
        PARTIAL = "partial", "Partial"
        ERROR = "error", "Error"

    batch = models.ForeignKey(IngestionBatch, on_delete=models.CASCADE, related_name='items')
    source_file = models.FileField(upload_to='ingestion_uploads/', max_length=500)
    original_filename = models.CharField(max_length=300)
    content_type = models.CharField(max_length=150, blank=True, default="")
    file_size_bytes = models.BigIntegerField(default=0)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.QUEUED, db_index=True)
    error_message = models.TextField(blank=True, default="")
    document_ids = models.JSONField(default=list, blank=True)
    parser_metadata = models.JSONField(default=dict, blank=True)
    manifest_complete = models.BooleanField(default=False)
    generation = models.UUIDField(default=uuid4)
    attempts = models.PositiveIntegerField(default=0)
    next_attempt_at = models.DateTimeField(default=timezone.now, db_index=True)
    failure_code = models.CharField(max_length=64, blank=True, default='')
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        app_label = 'apps_ingestion'
        db_table = 'aquillm_ingestionbatchitem'
        ordering = ['created_at']
