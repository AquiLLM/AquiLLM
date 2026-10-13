"""Append-only byte identities, archive occurrences and extraction snapshots."""

from django.core.exceptions import ValidationError
from django.db import models


class ImmutableSourceQuerySet(models.QuerySet):
    def update(self, **kwargs):
        raise ValidationError("Source artifact records are immutable.")

    def bulk_update(self, objs, fields, batch_size=None):
        raise ValidationError("Source artifact records are immutable.")

    def bulk_create(self, objs, **kwargs):
        if kwargs.get("update_conflicts"):
            raise ValidationError("Source artifact records are immutable.")
        return super().bulk_create(objs, **kwargs)


class ImmutableSourceRecord(models.Model):
    objects = ImmutableSourceQuerySet.as_manager()

    class Meta:
        abstract = True

    def save(self, *args, **kwargs):
        if not self._state.adding:
            raise ValidationError("Source artifact records are immutable.")
        return super().save(*args, **kwargs)


class SourceArtifact(ImmutableSourceRecord):
    sha256 = models.CharField(max_length=64, unique=True)
    size_bytes = models.PositiveBigIntegerField()
    immutable_storage_key = models.CharField(max_length=500, unique=True)

    class Meta:
        app_label = "apps_ingestion"
        db_table = "aquillm_sourceartifact"


class SourceArtifactOccurrence(ImmutableSourceRecord):
    artifact = models.ForeignKey(
        SourceArtifact, on_delete=models.PROTECT, related_name="occurrences"
    )
    container = models.ForeignKey(
        SourceArtifact,
        on_delete=models.PROTECT,
        null=True,
        blank=True,
        related_name="member_occurrences",
    )
    member_path = models.TextField(null=True, blank=True)
    archive_ordinal = models.PositiveIntegerField(null=True, blank=True)
    original_filename = models.TextField()
    content_type = models.CharField(max_length=150, blank=True)
    archive_lineage = models.JSONField(default=list, blank=True)
    occurrence_key = models.CharField(max_length=64, unique=True)

    class Meta:
        app_label = "apps_ingestion"
        db_table = "aquillm_sourceartifactoccurrence"
        constraints = [
            models.CheckConstraint(
                condition=(
                    models.Q(
                        container__isnull=True,
                        member_path__isnull=True,
                        archive_ordinal__isnull=True,
                    )
                    | models.Q(
                        container__isnull=False,
                        member_path__isnull=False,
                        archive_ordinal__isnull=False,
                    )
                ),
                name="source_occurrence_container_member_complete",
            )
        ]


class ExtractionRevision(ImmutableSourceRecord):
    occurrence = models.ForeignKey(
        SourceArtifactOccurrence,
        on_delete=models.PROTECT,
        related_name="extraction_revisions",
    )
    parser_key = models.CharField(max_length=200)
    canonical_text = models.TextField(blank=True)
    canonical_sha256 = models.CharField(max_length=64)
    ordered_elements = models.JSONField(default=list, blank=True)
    quality = models.CharField(
        max_length=20,
        choices=[
            ("complete", "Complete"),
            ("partial", "Partial"),
            ("unsupported", "Unsupported"),
        ],
    )
    issue_codes = models.JSONField(default=list, blank=True)
    manifest_sha256 = models.CharField(max_length=64)

    class Meta:
        app_label = "apps_ingestion"
        db_table = "aquillm_extractionrevision"
        constraints = [
            models.UniqueConstraint(
                fields=["occurrence", "parser_key", "manifest_sha256"],
                name="source_extraction_revision_manifest_unique",
            ),
            models.CheckConstraint(
                condition=models.Q(quality__in=["complete", "partial", "unsupported"]),
                name="source_extraction_revision_quality_valid",
            ),
        ]
