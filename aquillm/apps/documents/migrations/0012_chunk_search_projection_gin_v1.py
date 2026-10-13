"""Concurrent, retry-safe GIN with exact catalog shape validation."""

from django.contrib.postgres.indexes import GinIndex
from django.db import migrations

NAME = "textchunk_search_vector_gin_v1"
CATALOG_SQL = """
SELECT i.indrelid = to_regclass('aquillm_textchunk'), a.attname, am.amname,
       op.opcname, i.indisvalid, i.indisready, i.indisunique,
       i.indnatts, i.indnkeyatts, i.indpred IS NULL, i.indexprs IS NULL
FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid
JOIN pg_am am ON am.oid = c.relam
JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = i.indkey[0]
JOIN pg_opclass op ON op.oid = i.indclass[0]
WHERE i.indexrelid = to_regclass(%s)
"""
EXPECTED = (
    True,
    "search_vector",
    "gin",
    "tsvector_ops",
    True,
    True,
    False,
    1,
    1,
    True,
    True,
)


def create_index(apps, schema_editor):
    with schema_editor.connection.cursor() as cursor:
        cursor.execute(CATALOG_SQL, [NAME])
        row = cursor.fetchone()
        if row:
            if row[:4] + row[6:] != EXPECTED[:4] + EXPECTED[6:]:
                raise RuntimeError("unexpected_chunk_search_index")
            if not row[4] or not row[5]:
                cursor.execute("DROP INDEX CONCURRENTLY textchunk_search_vector_gin_v1")
        cursor.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS textchunk_search_vector_gin_v1 "
            "ON aquillm_textchunk USING gin (search_vector)"
        )
        cursor.execute(CATALOG_SQL, [NAME])
        if cursor.fetchone() != EXPECTED:
            raise RuntimeError("invalid_chunk_search_index")


def drop_index(apps, schema_editor):
    schema_editor.execute(
        "DROP INDEX CONCURRENTLY IF EXISTS textchunk_search_vector_gin_v1"
    )


class Migration(migrations.Migration):
    atomic = False
    dependencies = [("apps_documents", "0011_chunk_search_projection_v1")]
    operations = [
        migrations.SeparateDatabaseAndState(
            database_operations=[
                migrations.RunPython(create_index, drop_index, atomic=False)
            ],
            state_operations=[
                migrations.AddIndex(
                    "textchunk", GinIndex(fields=["search_vector"], name=NAME)
                )
            ],
        )
    ]
