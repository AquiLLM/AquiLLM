"""Immutable blobs, occurrence locations and ordered extraction snapshots."""

import django.db.models.deletion
from django.db import migrations, models

IMMUTABILITY_SQL = """
CREATE FUNCTION aquillm_source_record_immutable() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF NEW IS DISTINCT FROM OLD THEN
        RAISE EXCEPTION 'Source artifact records are immutable';
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER sourceartifact_immutable BEFORE UPDATE ON aquillm_sourceartifact
FOR EACH ROW EXECUTE FUNCTION aquillm_source_record_immutable();
CREATE TRIGGER sourceoccurrence_immutable
BEFORE UPDATE ON aquillm_sourceartifactoccurrence
FOR EACH ROW EXECUTE FUNCTION aquillm_source_record_immutable();
CREATE TRIGGER extractionrevision_immutable BEFORE UPDATE ON aquillm_extractionrevision
FOR EACH ROW EXECUTE FUNCTION aquillm_source_record_immutable();
"""
REVERSE_IMMUTABILITY_SQL = """
DROP TRIGGER extractionrevision_immutable ON aquillm_extractionrevision;
DROP TRIGGER sourceoccurrence_immutable ON aquillm_sourceartifactoccurrence;
DROP TRIGGER sourceartifact_immutable ON aquillm_sourceartifact;
DROP FUNCTION aquillm_source_record_immutable();
"""


class Migration(migrations.Migration):
    dependencies = [("apps_ingestion", "0002_expand_source_file_max_length")]
    operations = [
        migrations.CreateModel(
            name="SourceArtifact",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("sha256", models.CharField(max_length=64, unique=True)),
                ("size_bytes", models.PositiveBigIntegerField()),
                (
                    "immutable_storage_key",
                    models.CharField(max_length=500, unique=True),
                ),
            ],
            options={"db_table": "aquillm_sourceartifact"},
        ),
        migrations.CreateModel(
            name="SourceArtifactOccurrence",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("member_path", models.TextField(blank=True, null=True)),
                ("archive_ordinal", models.PositiveIntegerField(blank=True, null=True)),
                ("original_filename", models.TextField()),
                ("content_type", models.CharField(blank=True, max_length=150)),
                ("archive_lineage", models.JSONField(blank=True, default=list)),
                ("occurrence_key", models.CharField(max_length=64, unique=True)),
                (
                    "artifact",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name="occurrences",
                        to="apps_ingestion.sourceartifact",
                    ),
                ),
                (
                    "container",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name="member_occurrences",
                        to="apps_ingestion.sourceartifact",
                    ),
                ),
            ],
            options={
                "db_table": "aquillm_sourceartifactoccurrence",
                "constraints": [
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
                ],
            },
        ),
        migrations.CreateModel(
            name="ExtractionRevision",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("parser_key", models.CharField(max_length=200)),
                ("canonical_text", models.TextField(blank=True)),
                ("canonical_sha256", models.CharField(max_length=64)),
                ("ordered_elements", models.JSONField(blank=True, default=list)),
                (
                    "quality",
                    models.CharField(
                        choices=[
                            ("complete", "Complete"),
                            ("partial", "Partial"),
                            ("unsupported", "Unsupported"),
                        ],
                        max_length=20,
                    ),
                ),
                ("issue_codes", models.JSONField(blank=True, default=list)),
                ("manifest_sha256", models.CharField(max_length=64)),
                (
                    "occurrence",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name="extraction_revisions",
                        to="apps_ingestion.sourceartifactoccurrence",
                    ),
                ),
            ],
            options={
                "db_table": "aquillm_extractionrevision",
                "constraints": [
                    models.UniqueConstraint(
                        fields=("occurrence", "parser_key", "manifest_sha256"),
                        name="source_extraction_revision_manifest_unique",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(
                            quality__in=["complete", "partial", "unsupported"]
                        ),
                        name="source_extraction_revision_quality_valid",
                    ),
                ],
            },
        ),
        migrations.RunSQL(IMMUTABILITY_SQL, REVERSE_IMMUTABILITY_SQL),
    ]
