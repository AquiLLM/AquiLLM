"""Read-only catalog proof for the exact FTS index installed by migration 0012."""

from django.db import connections

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


def index_ready(using):
    if connections[using].vendor != "postgresql":
        return False
    with connections[using].cursor() as cursor:
        cursor.execute(CATALOG_SQL, [NAME])
        return cursor.fetchone() == (
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
