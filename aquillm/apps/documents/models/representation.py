"""Immutable structural sources and a separate current representation pointer."""
from uuid import uuid4

from django.contrib.contenttypes.fields import GenericForeignKey
from django.db import models


class DocumentRepresentationState(models.Model):
    model_label = models.CharField(max_length=128)
    document_pkid = models.PositiveBigIntegerField()
    document_id = models.UUIDField(db_index=True)
    owner_content_type = models.ForeignKey("contenttypes.ContentType", on_delete=models.PROTECT)
    owner_object = GenericForeignKey("owner_content_type", "document_pkid", for_concrete_model=True)
    collection_id = models.PositiveBigIntegerField()
    active_generation = models.ForeignKey(
        "DocumentRepresentationGeneration", null=True, on_delete=models.SET_NULL, related_name="+"
    )
    legacy_source_hash = models.CharField(max_length=64, null=True)
    legacy_closed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = "apps_documents"
        db_table = "aquillm_representationstate"
        constraints = [
            models.UniqueConstraint(fields=("model_label", "document_pkid", "document_id"),
                                    name="representation_exact_owner"),
            models.UniqueConstraint(fields=("model_label", "document_pkid"),
                                    name="representation_concrete_owner"),
        ]


class LegacyDocumentSourceSnapshot(models.Model):
    key = models.UUIDField(primary_key=True, default=uuid4, editable=False)
    owner = models.ForeignKey(DocumentRepresentationState, on_delete=models.CASCADE,
                              related_name="source_snapshots")
    canonical_text = models.TextField(null=True)
    canonical_sha256 = models.CharField(max_length=64, null=True)
    captured_source_hash = models.CharField(max_length=64)
    collection_id_at_capture = models.PositiveBigIntegerField()
    source_artifact = models.ForeignKey("apps_ingestion.SourceArtifact", null=True, on_delete=models.PROTECT)
    source_occurrence = models.ForeignKey("apps_ingestion.SourceArtifactOccurrence", null=True, on_delete=models.PROTECT)
    extraction_revision = models.ForeignKey("apps_ingestion.ExtractionRevision", null=True, on_delete=models.PROTECT)
    origin_output = models.ForeignKey("apps_ingestion.IngestionOutput", null=True, on_delete=models.RESTRICT)
    source_binding = models.JSONField(null=True)
    media_binding = models.JSONField(null=True)
    source_content_type = models.CharField(max_length=150, blank=True, default="")
    canonical_status = models.CharField(max_length=16)
    canonical_failure_code = models.CharField(max_length=64, blank=True, default="")
    manifest_sha256 = models.CharField(max_length=64)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = "apps_documents"
        db_table = "aquillm_legacysourcesnapshot"
        constraints = [
            models.UniqueConstraint(fields=("owner", "captured_source_hash", "manifest_sha256"),
                                    name="legacy_source_snapshot_identity"),
            models.CheckConstraint(condition=(
                models.Q(canonical_status="verified", canonical_text__isnull=False,
                         canonical_sha256__isnull=False, canonical_failure_code="")
                | models.Q(canonical_status="unavailable", canonical_text__isnull=True,
                           canonical_sha256__isnull=True, canonical_failure_code__in=(
                               "canonical_source_unavailable", "canonical_hash_mismatch"))
            ), name="legacy_canonical_status_valid"),
        ]


class DocumentRepresentationGeneration(models.Model):
    key = models.UUIDField(primary_key=True, default=uuid4, editable=False)
    owner = models.ForeignKey(DocumentRepresentationState, on_delete=models.CASCADE,
                              related_name="generations")
    collection_id = models.PositiveBigIntegerField()
    input_kind = models.CharField(max_length=24)
    extraction_revision = models.ForeignKey("apps_ingestion.ExtractionRevision", null=True, on_delete=models.PROTECT)
    origin_output = models.ForeignKey("apps_ingestion.IngestionOutput", null=True, on_delete=models.RESTRICT)
    source_snapshot = models.ForeignKey(LegacyDocumentSourceSnapshot, null=True, on_delete=models.RESTRICT)
    source_revision_sha256 = models.CharField(max_length=64)
    parser_key = models.CharField(max_length=200)
    chunker_key = models.CharField(max_length=200)
    preparation_policy_key = models.CharField(max_length=200)
    canonical_text = models.TextField()
    canonical_sha256 = models.CharField(max_length=64)
    extraction_quality = models.CharField(max_length=20)
    issue_codes = models.JSONField(default=list)
    predecessor_source_hash = models.CharField(max_length=64)
    predecessor_collection_id = models.PositiveBigIntegerField()
    predecessor_generation_key = models.UUIDField(null=True)
    manifest = models.JSONField()
    manifest_sha256 = models.CharField(max_length=64)
    expected_count = models.PositiveIntegerField()
    status = models.CharField(max_length=16, default="building")
    lease_token = models.UUIDField(default=uuid4)
    lease_expires_at = models.DateTimeField()
    attempt = models.PositiveIntegerField(default=1)
    failure_code = models.CharField(max_length=64, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    staged_at = models.DateTimeField(null=True)
    activated_at = models.DateTimeField(null=True)
    retired_at = models.DateTimeField(null=True)
    publication_id = models.PositiveBigIntegerField(null=True)
    publication_generation = models.UUIDField(null=True)
    publication_attempt = models.PositiveIntegerField(null=True)

    class Meta:
        app_label = "apps_documents"
        db_table = "aquillm_representationgeneration"
        constraints = [
            models.UniqueConstraint(fields=("owner",), condition=models.Q(status="active"),
                                    name="representation_one_active"),
            models.CheckConstraint(condition=models.Q(status__in=(
                "building", "staged", "active", "retired", "failed", "stale")),
                name="representation_status_valid"),
            models.CheckConstraint(condition=models.Q(attempt__gt=0) & (
                models.Q(expected_count__gt=0)
                | models.Q(expected_count=0, status__in=("building", "failed", "stale"))),
                                   name="representation_positive_counts"),
            models.CheckConstraint(condition=models.Q(extraction_quality__in=("complete", "partial")),
                                   name="representation_quality_valid"),
            models.CheckConstraint(condition=(
                models.Q(input_kind="extraction", extraction_revision__isnull=False,
                         origin_output__isnull=False, source_snapshot__isnull=True)
                | models.Q(input_kind="canonical_snapshot", extraction_revision__isnull=True,
                           origin_output__isnull=True, source_snapshot__isnull=False)
            ), name="representation_input_complete"),
            models.CheckConstraint(condition=(
                models.Q(publication_id__isnull=True, publication_generation__isnull=True,
                         publication_attempt__isnull=True)
                | models.Q(publication_id__isnull=False, publication_generation__isnull=False,
                           publication_attempt__gt=0)
            ), name="representation_publication_complete"),
        ]
