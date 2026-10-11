"""Add immutable structural history without attributing files to legacy chunks."""
from uuid import uuid4

from django.db import migrations, models


def reject_unbound_generations(apps, schema_editor):
    chunks = apps.get_model("apps_documents", "TextChunk")
    if chunks.objects.using(schema_editor.connection.alias).filter(
        representation_generation__isnull=False
    ).exists():
        raise RuntimeError("unproven_representation_generation")


def refuse_used_reverse(apps, schema_editor):
    for name in ("DocumentRepresentationState", "DocumentRepresentationGeneration",
                 "LegacyDocumentSourceSnapshot"):
        if apps.get_model("apps_documents", name).objects.using(schema_editor.connection.alias).exists():
            raise RuntimeError("representation_history_in_use")


def fk(target, delete, **kw):
    return models.ForeignKey(to=target, on_delete=delete, **kw)


class Migration(migrations.Migration):
    dependencies = [
        ("apps_documents", "0012_chunk_search_projection_gin_v1"),
        ("apps_ingestion", "0004_ingestion_output_ledger_v1"),
        ("contenttypes", "0002_remove_content_type_name"),
    ]
    operations = [
        migrations.RunPython(reject_unbound_generations, migrations.RunPython.noop),
        migrations.CreateModel("DocumentRepresentationState", [
            ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
            ("model_label", models.CharField(max_length=128)),
            ("document_pkid", models.PositiveBigIntegerField()),
            ("document_id", models.UUIDField(db_index=True)),
            ("owner_content_type", fk("contenttypes.contenttype", models.PROTECT)),
            ("collection_id", models.PositiveBigIntegerField()),
            ("legacy_source_hash", models.CharField(max_length=64, null=True)),
            ("legacy_closed", models.BooleanField(default=False)),
            ("created_at", models.DateTimeField(auto_now_add=True)),
            ("updated_at", models.DateTimeField(auto_now=True)),
        ], options={"db_table": "aquillm_representationstate", "constraints": [
            models.UniqueConstraint(fields=("model_label", "document_pkid", "document_id"),
                                    name="representation_exact_owner"),
            models.UniqueConstraint(fields=("model_label", "document_pkid"),
                                    name="representation_concrete_owner"),
        ]}),
        migrations.CreateModel("LegacyDocumentSourceSnapshot", [
            ("key", models.UUIDField(primary_key=True, default=uuid4, editable=False, serialize=False)),
            ("owner", fk("apps_documents.documentrepresentationstate", models.CASCADE, related_name="source_snapshots")),
            ("canonical_text", models.TextField(null=True)),
            ("canonical_sha256", models.CharField(max_length=64, null=True)),
            ("captured_source_hash", models.CharField(max_length=64)),
            ("collection_id_at_capture", models.PositiveBigIntegerField()),
            ("source_artifact", fk("apps_ingestion.sourceartifact", models.PROTECT, null=True)),
            ("source_occurrence", fk("apps_ingestion.sourceartifactoccurrence", models.PROTECT, null=True)),
            ("extraction_revision", fk("apps_ingestion.extractionrevision", models.PROTECT, null=True)),
            ("origin_output", fk("apps_ingestion.ingestionoutput", models.RESTRICT, null=True)),
            ("source_binding", models.JSONField(null=True)),
            ("media_binding", models.JSONField(null=True)),
            ("source_content_type", models.CharField(max_length=150, blank=True, default="")),
            ("canonical_status", models.CharField(max_length=16)),
            ("canonical_failure_code", models.CharField(max_length=64, blank=True, default="")),
            ("manifest_sha256", models.CharField(max_length=64)),
            ("created_at", models.DateTimeField(auto_now_add=True)),
        ], options={"db_table": "aquillm_legacysourcesnapshot", "constraints": [
            models.UniqueConstraint(fields=("owner", "captured_source_hash", "manifest_sha256"),
                                    name="legacy_source_snapshot_identity"),
            models.CheckConstraint(condition=(
                models.Q(canonical_status="verified", canonical_text__isnull=False,
                         canonical_sha256__isnull=False, canonical_failure_code="")
                | models.Q(canonical_status="unavailable", canonical_text__isnull=True,
                           canonical_sha256__isnull=True, canonical_failure_code__in=(
                               "canonical_source_unavailable", "canonical_hash_mismatch"))
            ), name="legacy_canonical_status_valid"),
        ]}),
        migrations.CreateModel("DocumentRepresentationGeneration", [
            ("key", models.UUIDField(primary_key=True, default=uuid4, editable=False, serialize=False)),
            ("owner", fk("apps_documents.documentrepresentationstate", models.CASCADE, related_name="generations")),
            ("collection_id", models.PositiveBigIntegerField()),
            ("input_kind", models.CharField(max_length=24)),
            ("extraction_revision", fk("apps_ingestion.extractionrevision", models.PROTECT, null=True)),
            ("origin_output", fk("apps_ingestion.ingestionoutput", models.RESTRICT, null=True)),
            ("source_snapshot", fk("apps_documents.legacydocumentsourcesnapshot", models.RESTRICT, null=True)),
            ("source_revision_sha256", models.CharField(max_length=64)),
            ("parser_key", models.CharField(max_length=200)),
            ("chunker_key", models.CharField(max_length=200)),
            ("preparation_policy_key", models.CharField(max_length=200)),
            ("canonical_text", models.TextField()),
            ("canonical_sha256", models.CharField(max_length=64)),
            ("extraction_quality", models.CharField(max_length=20)),
            ("issue_codes", models.JSONField(default=list)),
            ("predecessor_source_hash", models.CharField(max_length=64)),
            ("predecessor_collection_id", models.PositiveBigIntegerField()),
            ("predecessor_generation_key", models.UUIDField(null=True)),
            ("manifest", models.JSONField()),
            ("manifest_sha256", models.CharField(max_length=64)),
            ("expected_count", models.PositiveIntegerField()),
            ("status", models.CharField(max_length=16, default="building")),
            ("lease_token", models.UUIDField(default=uuid4)),
            ("lease_expires_at", models.DateTimeField()),
            ("attempt", models.PositiveIntegerField(default=1)),
            ("failure_code", models.CharField(max_length=64, blank=True, default="")),
            ("created_at", models.DateTimeField(auto_now_add=True)),
            ("staged_at", models.DateTimeField(null=True)),
            ("activated_at", models.DateTimeField(null=True)),
            ("retired_at", models.DateTimeField(null=True)),
            ("publication_id", models.PositiveBigIntegerField(null=True)),
            ("publication_generation", models.UUIDField(null=True)),
            ("publication_attempt", models.PositiveIntegerField(null=True)),
        ], options={"db_table": "aquillm_representationgeneration", "constraints": [
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
        ]}),
        migrations.AddField("documentrepresentationstate", "active_generation",
            fk("apps_documents.documentrepresentationgeneration", models.SET_NULL, null=True, related_name="+")),
        migrations.AlterField("textchunk", "representation_generation",
            fk("apps_documents.documentrepresentationgeneration", models.CASCADE, null=True,
               db_column="representation_generation_id")),
        migrations.AddField("textchunk", "legacy_source_snapshot",
            fk("apps_documents.legacydocumentsourcesnapshot", models.CASCADE, null=True)),
        *[migrations.AddField("textchunk", name, models.CharField(max_length=16, default="unbound"))
          for name in ("legacy_context_status", "legacy_artifact_status")],
        *[migrations.AddField("textchunk", name, models.CharField(max_length=64, default="", blank=True))
          for name in ("legacy_context_failure_code", "legacy_artifact_failure_code")],
        migrations.RunPython(migrations.RunPython.noop, refuse_used_reverse),
    ]
