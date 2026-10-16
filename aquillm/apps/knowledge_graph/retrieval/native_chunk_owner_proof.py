"""Fresh native equality proof at the original batch chunk-owner boundary."""

from uuid import UUID

from django.db import connections

from apps.documents.models import TextChunk

from .native_seed_metadata_eligibility import environment_sql


def require_chunk_owners(metadata, *, requested, using):
    try:
        metadata.check()
        references = metadata.owner.plan.references
        if (
            type(requested) is not dict
            or not 1 <= len(requested) <= 128
            or any(
                type(pk) is not int
                or pk <= 0
                or type(doc) is not UUID
                or pk not in references
                or references[pk].doc_id != doc
                for pk, doc in requested.items()
            )
        ):
            raise ValueError("native chunk expected owners changed")
        n = len(requested)
        values = ",".join("(%s::bigint,%s::uuid)" for _ in requested)
        params = tuple(v for pair in requested.items() for v in pair)
        connection = connections[using]
        table = connection.ops.quote_name(TextChunk._meta.db_table)
        environment, ep = environment_sql(connection, ((TextChunk, ("id", "doc_id")),))
        sql = f"""WITH expected(id,doc_id) AS (VALUES {values}),
          current AS MATERIALIZED (SELECT id,doc_id FROM {table}
            WHERE id IN (SELECT id FROM expected) ORDER BY id LIMIT {n + 1})
          SELECT count(*),count(DISTINCT id),count(*) FILTER (WHERE EXISTS
            (SELECT 1 FROM expected e WHERE e.id=c.id AND e.doc_id=c.doc_id)),
            ({environment}) FROM current c"""
        rows = metadata.read(sql, (*params, *ep), using=using, maximum=1)
        if (
            len(rows) != 1
            or len(rows[0]) != 4
            or any(type(v) is not int for v in rows[0][:3])
            or rows[0] != (n, n, n, True)
            or rows[0][3] is not True
        ):
            raise ValueError("extended seed chunk ownership is stale")
        metadata.check()
    except BaseException:
        metadata.failed = True
        raise
