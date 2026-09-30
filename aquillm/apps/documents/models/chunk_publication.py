"""Transactional, exact-source intent for at-least-once chunk task publication."""
from django.db import models
from django.utils import timezone


class ChunkPublication(models.Model):
    concrete_model_label = models.CharField(max_length=100)
    document_pkid = models.PositiveBigIntegerField()
    document_id = models.UUIDField(db_index=True)
    source_hash = models.CharField(max_length=64)
    collection_ids = models.JSONField(default=list)
    attempts = models.PositiveIntegerField(default=0)
    next_attempt_at = models.DateTimeField(default=timezone.now, db_index=True)
    last_error = models.CharField(max_length=128, blank=True, default='')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = 'apps_documents'
        constraints = [models.UniqueConstraint(
            fields=['concrete_model_label', 'document_pkid', 'source_hash'],
            name='unique_chunk_publication_source',
        )]
