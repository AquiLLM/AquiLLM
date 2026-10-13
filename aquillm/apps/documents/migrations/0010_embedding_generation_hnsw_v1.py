"""Retry-safe concurrent index creation; invalid interrupted indexes are rebuilt."""

from django.db import migrations
from pgvector.django import HnswIndex

NAME = "dense_sidecar_hnsw_v1"


def create_index(apps, schema_editor):
    with schema_editor.connection.cursor() as cursor:
        cursor.execute(
            "SELECT indisvalid FROM pg_index WHERE indexrelid = to_regclass(%s)", [NAME]
        )
        row = cursor.fetchone()
        if row and not row[0]:
            cursor.execute("DROP INDEX CONCURRENTLY IF EXISTS dense_sidecar_hnsw_v1")
        cursor.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS dense_sidecar_hnsw_v1 "
            "ON apps_documents_chunkembedding USING hnsw (vector vector_l2_ops) "
            "WITH (m=16, ef_construction=64)"
        )


def drop_index(apps, schema_editor):
    schema_editor.execute("DROP INDEX CONCURRENTLY IF EXISTS dense_sidecar_hnsw_v1")


class Migration(migrations.Migration):
    atomic = False
    dependencies = [("apps_documents", "0009_embedding_generation_v1")]
    operations = [
        migrations.SeparateDatabaseAndState(
            database_operations=[
                migrations.RunPython(create_index, drop_index, atomic=False)
            ],
            state_operations=[
                migrations.AddIndex(
                    model_name="chunkembedding",
                    index=HnswIndex(
                        name=NAME,
                        fields=["vector"],
                        m=16,
                        ef_construction=64,
                        opclasses=["vector_l2_ops"],
                    ),
                )
            ],
        )
    ]
