"""Additive projections; reversal refuses to discard used provenance."""

from django.contrib.postgres.search import SearchVectorField
from django.db import migrations, models
from django.db.models import Q


def refuse_used_reverse(apps, schema_editor):
    chunks = apps.get_model("apps_documents", "TextChunk")
    if (
        chunks.objects.using(schema_editor.connection.alias)
        .filter(
            Q(search_vector__isnull=False)
            | Q(search_vector_binding__isnull=False)
            | ~Q(source_kind="unknown")
            | ~Q(section_path=[])
        )
        .exists()
    ):
        raise RuntimeError("chunk_search_projection_in_use")


class Migration(migrations.Migration):
    dependencies = [("apps_documents", "0010_embedding_generation_hnsw_v1")]
    operations = [
        migrations.AddField(
            "textchunk", "search_vector", SearchVectorField(null=True, editable=False)
        ),
        migrations.AddField(
            "textchunk",
            "search_vector_binding",
            models.JSONField(null=True, editable=False),
        ),
        migrations.AddField(
            "textchunk",
            "source_kind",
            models.CharField(
                max_length=32,
                default="unknown",
                choices=[
                    (kind, kind)
                    for kind in (
                        "unknown",
                        "body",
                        "heading",
                        "table",
                        "formula",
                        "figure",
                        "bibliography",
                    )
                ],
            ),
        ),
        migrations.AddField(
            "textchunk", "section_path", models.JSONField(default=list)
        ),
        migrations.RunPython(migrations.RunPython.noop, refuse_used_reverse),
    ]
