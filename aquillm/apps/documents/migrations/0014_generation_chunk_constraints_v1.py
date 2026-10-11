"""Validate additive per-generation uniqueness before retiring global uniqueness."""
from django.db import migrations, models
from django.db.models import Count

from ._representation_guards import install, uninstall


def validate_rows(apps, schema_editor):
    rows = apps.get_model("apps_documents", "TextChunk").objects.using(schema_editor.connection.alias)
    for fields in (("doc_id", "representation_generation", "chunk_number"),
                   ("doc_id", "representation_generation", "start_position", "end_position")):
        if rows.values(*fields).annotate(n=Count("pk")).filter(n__gt=1).exists():
            raise RuntimeError("representation_chunk_duplicates")
    # Legacy offsets are classified at capture; they cannot block this migration.


def reverse_guard(apps, schema_editor):
    for name in ("DocumentRepresentationState", "DocumentRepresentationGeneration",
                 "LegacyDocumentSourceSnapshot"):
        if apps.get_model("apps_documents", name).objects.using(schema_editor.connection.alias).exists():
            raise RuntimeError("representation_history_in_use")
    rows = apps.get_model("apps_documents", "TextChunk").objects.using(schema_editor.connection.alias)
    for fields in (("doc_id", "chunk_number"), ("doc_id", "start_position", "end_position")):
        if rows.values(*fields).annotate(n=Count("pk")).filter(n__gt=1).exists():
            raise RuntimeError("legacy_chunk_uniqueness_unavailable")


class Migration(migrations.Migration):
    dependencies = [("apps_documents", "0013_representation_generations_v1")]
    operations = [
        migrations.RunPython(validate_rows, migrations.RunPython.noop),
        migrations.AddConstraint("textchunk", models.UniqueConstraint(
            fields=("doc_id", "representation_generation", "chunk_number"),
            name="generation_chunk_number_unique")),
        migrations.AddConstraint("textchunk", models.UniqueConstraint(
            fields=("doc_id", "representation_generation", "start_position", "end_position"),
            name="generation_chunk_position_unique")),
        migrations.AddConstraint("textchunk", models.UniqueConstraint(
            fields=("doc_id", "chunk_number"), condition=models.Q(representation_generation__isnull=True),
            name="legacy_chunk_number_unique")),
        migrations.AddConstraint("textchunk", models.UniqueConstraint(
            fields=("doc_id", "start_position", "end_position"),
            condition=models.Q(representation_generation__isnull=True),
            name="legacy_chunk_position_unique")),
        migrations.RemoveConstraint("textchunk", "uniqe_chunk_per_document"),
        migrations.RemoveConstraint("textchunk", "unique_chunk_position_per_document"),
        migrations.AddConstraint("textchunk", models.CheckConstraint(
            condition=models.Q(representation_generation__isnull=True) | models.Q(legacy_source_snapshot__isnull=True),
            name="chunk_single_source_binding")),
        migrations.RunPython(install, uninstall),
        migrations.RunPython(migrations.RunPython.noop, reverse_guard),
    ]
