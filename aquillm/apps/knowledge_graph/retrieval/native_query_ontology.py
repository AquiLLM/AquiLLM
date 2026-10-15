"""Fresh bounded native artifact identity proof, followed by real YAML loading."""

from django.db import connections

from apps.knowledge_graph.models import GraphArtifact

from .native_seed_metadata_eligibility import environment_sql
from .query_ontology import _complete_query_ontology, _mixed


def load_native_ontology(metadata, *, group):
    metadata.check()
    # Original checked group inputs precede projection preparation.
    states = metadata.states(group)
    ids = tuple(sorted(s["active_artifact_id"] for s in states))
    if not ids or len(ids) > 128 or len(set(ids)) != len(ids):
        raise ValueError("native ontology input is invalid")
    count = len(ids)
    connection = connections[metadata.using]
    environment, params = environment_sql(
        connection,
        (
            (
                GraphArtifact,
                (
                    "id",
                    "ontology_version",
                    "ontology_checksum",
                    "status",
                    "scope_type",
                    "evaluation_only",
                ),
            ),
        ),
    )
    table = connection.ops.quote_name(GraphArtifact._meta.db_table)
    sql = f"""WITH current AS MATERIALIZED (
      SELECT id, ontology_version COLLATE pg_catalog."C" version,
        ontology_checksum COLLATE pg_catalog."C" checksum FROM {table}
      WHERE id=ANY(%s::bigint[]) AND scope_type=%s AND status=%s
        AND evaluation_only=FALSE
      ORDER BY id LIMIT {count + 1})
      SELECT count(*),count(DISTINCT id),count(*) FILTER
        (WHERE version IS NOT NULL AND checksum IS NOT NULL),
        count(DISTINCT (version,checksum)),min(version),min(checksum),
        ({environment}) FROM current"""
    try:
        rows = metadata.read(
            sql,
            (list(ids), "collection", "active", *params),
            using=metadata.using,
            maximum=1,
        )
        if (
            len(rows) != 1
            or len(rows[0]) != 7
            or any(type(v) is not int for v in rows[0][:4])
            or type(rows[0][6]) is not bool
        ):
            raise ValueError("malformed native ontology proof")
        total, distinct, nonnull, pairs, version, checksum, environment = rows[0]
        if environment is not True:
            raise ValueError("native ontology environment changed")
        if (total, distinct, nonnull, pairs) != (count, count, count, 1):
            return _mixed(count)
        if type(version) is not str or type(checksum) is not str:
            raise ValueError("malformed native ontology identity")
        outcome = _complete_query_ontology(
            version=version,
            checksum=checksum,
            count=count,
            using=metadata.using,
            read_budget=metadata.budget,
        )
        metadata.check()
        return outcome
    except BaseException:
        metadata.failed = True
        raise


_NATIVE_ONTOLOGY = load_native_ontology
