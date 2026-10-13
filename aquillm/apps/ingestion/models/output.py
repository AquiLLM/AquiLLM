"""Durable output identities. Database triggers also fence bulk/raw writes."""
from uuid import uuid4
from django.core.exceptions import ValidationError
from django.db import models
from django.utils import timezone

IDENTITY_FIELDS = ('item_id', 'output_key', 'source_occurrence_id',
                   'extraction_revision_id', 'parent_output_id', 'output_kind',
                   'element_key', 'payload')


class OutputQuerySet(models.QuerySet):
    def update(self, **kwargs):
        if set(kwargs).intersection(IDENTITY_FIELDS):
            raise ValidationError('Output identity is immutable.')
        return super().update(**kwargs)


class IngestionOutput(models.Model):
    item = models.ForeignKey('apps_ingestion.IngestionBatchItem', on_delete=models.CASCADE, related_name='outputs')
    output_key = models.CharField(max_length=64, unique=True)
    source_occurrence = models.ForeignKey('apps_ingestion.SourceArtifactOccurrence', on_delete=models.PROTECT, null=True)
    extraction_revision = models.ForeignKey('apps_ingestion.ExtractionRevision', on_delete=models.PROTECT, null=True)
    parent_output = models.ForeignKey('self', on_delete=models.RESTRICT, null=True)
    output_kind = models.CharField(max_length=32)
    element_key = models.CharField(max_length=200)
    payload = models.JSONField(default=dict)
    status = models.CharField(max_length=20, default='queued')
    generation = models.UUIDField(default=uuid4)
    attempt = models.PositiveIntegerField(default=0)
    next_attempt_at = models.DateTimeField(default=timezone.now, db_index=True)
    error_code = models.CharField(max_length=64, blank=True, default='')
    document_model = models.CharField(max_length=100, blank=True, default='')
    document_pkid = models.PositiveBigIntegerField(null=True)
    document_id = models.UUIDField(null=True)
    document_collection_id = models.BigIntegerField(null=True)
    document_source_hash = models.CharField(max_length=64, blank=True, default='')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    objects = OutputQuerySet.as_manager()

    class Meta:
        app_label = 'apps_ingestion'
        db_table = 'aquillm_ingestionoutput'
        ordering = ['pk']
        constraints = [
            models.UniqueConstraint(fields=['item', 'output_key'], name='ingestion_output_item_key_unique'),
            models.CheckConstraint(condition=models.Q(status__in=['queued', 'processing', 'success', 'error']), name='ingestion_output_status_valid'),
            models.CheckConstraint(condition=(
                models.Q(document_model='', document_pkid__isnull=True, document_id__isnull=True,
                         document_collection_id__isnull=True, document_source_hash='') |
                (~models.Q(document_model='') & models.Q(document_pkid__isnull=False, document_id__isnull=False,
                         document_collection_id__isnull=False) & ~models.Q(document_source_hash=''))),
                name='ingestion_output_reference_complete'),
            models.CheckConstraint(condition=~models.Q(status='success') | models.Q(document_id__isnull=False), name='ingestion_output_success_reference'),
            models.CheckConstraint(condition=models.Q(output_kind='registration_error', status='error') |
                models.Q(source_occurrence__isnull=False, extraction_revision__isnull=False), name='ingestion_output_source_required'),
        ]

    def save(self, *args, **kwargs):
        if not self._state.adding:
            previous = type(self).objects.using(kwargs.get('using') or self._state.db).get(pk=self.pk)
            if any(getattr(previous, f) != getattr(self, f) for f in IDENTITY_FIELDS):
                raise ValidationError('Output identity is immutable.')
        return super().save(*args, **kwargs)
