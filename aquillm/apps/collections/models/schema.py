from __future__ import annotations

import uuid

from django.conf import settings
from django.db import models
from django.db.models import Q


class ImmutableSchemaVersionQuerySet(models.QuerySet):
    def update(self, **kwargs):
        raise ValueError("published collection schema versions are immutable")

    def bulk_update(self, objs, fields, batch_size=None):
        raise ValueError("published collection schema versions are immutable")

    def delete(self):
        raise ValueError("published collection schema versions are immutable")


class CollectionSchemaVersion(models.Model):
    collection = models.ForeignKey(
        "apps_collections.Collection",
        on_delete=models.CASCADE,
        related_name="schema_versions",
    )
    version = models.PositiveIntegerField()
    checksum = models.CharField(max_length=64)
    definitions = models.JSONField(default=dict)
    ontology_version = models.OneToOneField(
        "apps_knowledge_graph.OntologyVersion",
        on_delete=models.PROTECT,
        related_name="collection_schema_version",
    )
    published_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
        related_name="published_collection_schema_versions",
    )
    published_at = models.DateTimeField(auto_now_add=True)
    summary = models.CharField(max_length=512, blank=True, default="")

    objects = ImmutableSchemaVersionQuerySet.as_manager()

    class Meta:
        app_label = "apps_collections"
        ordering = ("-version",)
        constraints = [
            models.CheckConstraint(
                condition=Q(version__gt=0),
                name="collection_schema_version_positive",
            ),
            models.UniqueConstraint(
                fields=("collection", "version"),
                name="collection_schema_version_unique",
            ),
            models.UniqueConstraint(
                fields=("collection", "checksum"),
                name="collection_schema_checksum_unique",
            ),
        ]

    def save(self, *args, **kwargs):
        if self.pk is not None and type(self)._base_manager.filter(pk=self.pk).exists():
            raise ValueError("published collection schema versions are immutable")
        return super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        raise ValueError("published collection schema versions are immutable")


class CollectionSchemaDraft(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    collection = models.OneToOneField(
        "apps_collections.Collection",
        on_delete=models.CASCADE,
        related_name="schema_draft",
    )
    base_version = models.ForeignKey(
        CollectionSchemaVersion,
        on_delete=models.SET_NULL,
        related_name="derived_drafts",
        null=True,
        blank=True,
    )
    revision = models.PositiveIntegerField(default=1)
    definitions = models.JSONField(default=dict)
    last_editor = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
        related_name="edited_collection_schema_drafts",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = "apps_collections"
        constraints = [
            models.CheckConstraint(
                condition=Q(revision__gt=0),
                name="collection_schema_draft_revision_positive",
            )
        ]


class CollectionSchemaGenerationRun(models.Model):
    class Status(models.TextChoices):
        QUEUED = "queued", "Queued"
        RUNNING = "running", "Running"
        SUCCEEDED = "succeeded", "Succeeded"
        FAILED = "failed", "Failed"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    collection = models.ForeignKey(
        "apps_collections.Collection",
        on_delete=models.CASCADE,
        related_name="schema_generation_runs",
    )
    requested_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        related_name="collection_schema_generation_runs",
        null=True,
        blank=True,
    )
    status = models.CharField(
        max_length=16,
        choices=Status.choices,
        default=Status.QUEUED,
    )
    source_signature = models.CharField(max_length=64)
    base_draft_id = models.UUIDField(null=True, blank=True)
    base_draft_revision = models.PositiveIntegerField(null=True, blank=True)
    lease_token = models.UUIDField(null=True, blank=True)
    lease_expires_at = models.DateTimeField(null=True, blank=True)
    statistics = models.JSONField(default=dict, blank=True)
    error_code = models.CharField(max_length=64, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = "apps_collections"
        ordering = ("-created_at",)
        constraints = [
            models.UniqueConstraint(
                fields=("collection",),
                condition=Q(status__in=("queued", "running")),
                name="collection_schema_one_active_generation",
            )
        ]
        indexes = [
            models.Index(
                fields=("collection", "status"),
                name="col_schema_run_status_idx",
            )
        ]
