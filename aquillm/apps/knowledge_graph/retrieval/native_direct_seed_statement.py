"""One fresh bounded membership proof plus the original bounded candidate SQL."""

from math import isfinite

from django.db import connections

from apps.knowledge_graph.models import CollectionGraphMembershipState as State

from .direct_seed_contracts import DirectResolutionTier
from .direct_seed_queryset import _candidate_queryset
from .direct_seed_types import DirectSeedCandidateRow
from .native_seed_metadata_eligibility import environment_sql

FIELDS = (
    "collection_id",
    "active_artifact_id",
    "registry_epoch",
    "membership_checksum",
    "resolver_version",
    "resolution_config_checksum",
)


def statement(metadata, options):
    metadata.check()
    states = options["membership_states"]
    if (
        type(states) is not tuple
        or not 1 <= len(states) <= 128
        or len({s["collection_id"] for s in states}) != len(states)
        or len({s["active_artifact_id"] for s in states}) != len(states)
        or any(
            type(s) is not dict
            or set(s) != set(FIELDS)
            or any(
                type(s[f]) is not int or s[f] < (0 if f == "registry_epoch" else 1)
                for f in FIELDS[:3]
            )
            or any(type(s[f]) is not str for f in FIELDS[3:])
            for s in states
        )
    ):
        raise ValueError("native expected membership is invalid")
    connection = connections[metadata.using]
    query = _candidate_queryset(**options)
    candidate, candidate_params = query.query.get_compiler(
        using=metadata.using
    ).as_sql()
    n = len(states)
    values = ",".join(
        "(%s::bigint,%s::bigint,%s::bigint,%s::varchar,%s::varchar,%s::varchar)"
        for _ in states
    )
    params = tuple(s[f] for s in states for f in FIELDS)
    fields = ",".join(FIELDS)
    equality = " AND ".join(
        f"c.{f}=e.{f}"
        if f in FIELDS[:3]
        else f'c.{f} COLLATE pg_catalog."C"=e.{f} COLLATE pg_catalog."C"'
        for f in FIELDS
    )
    environment, environment_params = environment_sql(connection, ((State, FIELDS),))
    sql = f"""WITH expected({fields}) AS (VALUES {values}),
      current AS MATERIALIZED (SELECT {fields}
        FROM {connection.ops.quote_name(State._meta.db_table)}
        WHERE collection_id IN (SELECT collection_id FROM expected)
        ORDER BY collection_id LIMIT {n + 1}),
      counters AS MATERIALIZED (SELECT count(*) total,
        count(DISTINCT collection_id) collections,
        count(DISTINCT active_artifact_id) artifacts,
        count(*) FILTER (WHERE EXISTS (SELECT 1 FROM expected e
          WHERE {equality})) matches FROM current c),
      proof AS MATERIALIZED (SELECT *, (total={n} AND collections={n} AND artifacts={n}
        AND matches={n} AND {environment}) valid FROM counters),
      candidates AS MATERIALIZED (SELECT q.* FROM ({candidate}) q
        WHERE (SELECT valid FROM proof))
      SELECT 0::integer tag,total,collections,artifacts,matches,valid,
        NULL::bigint id,NULL::bigint artifact_id,NULL::varchar entity_type,
        NULL::bigint automatic_canonical_entity_id,NULL::bigint automatic_link_count,
        NULL::float8 similarity FROM proof
      UNION ALL SELECT 1,NULL::bigint,NULL::bigint,NULL::bigint,
        NULL::bigint,NULL::boolean,
        id,artifact_id,entity_type,automatic_canonical_entity_id,
        automatic_link_count,similarity FROM candidates
      ORDER BY tag, similarity DESC NULLS LAST, id"""
    metadata.check()
    return sql, (*params, *environment_params, *candidate_params)


def validate(rows, *, count, limit):
    if (
        type(rows) is not tuple
        or not rows
        or any(type(r) is not tuple or len(r) != 12 for r in rows)
    ):
        raise ValueError("missing native membership header")
    header = rows[0]
    if (
        type(header[0]) is not int
        or header[0] != 0
        or any(type(v) is not int for v in header[1:5])
        or header[1:5] != (count,) * 4
        or header[5] is not True
        or any(v is not None for v in header[6:])
    ):
        raise ValueError("invalid native membership proof")
    candidates = rows[1:]
    if any(
        type(r[0]) is not int
        or r[0] != 1
        or any(v is not None for v in r[1:6])
        or any(type(r[i]) is not int or r[i] <= 0 for i in (6, 7))
        or type(r[8]) is not str
        or type(r[11]) is not float
        or not isfinite(r[11])
        or (r[9] is not None and (type(r[9]) is not int or r[9] <= 0))
        or (r[10] is not None and (type(r[10]) is not int or r[10] < 0))
        for r in candidates
    ):
        raise ValueError("invalid native candidate record")
    if tuple(sorted(candidates, key=lambda r: (-r[11], r[6]))) != candidates or len(
        {r[6] for r in candidates}
    ) != len(candidates):
        raise ValueError("native candidates are not unique and ordered")
    if len(candidates) > limit:
        raise ValueError("candidate result exceeds its hard cap")
    if any((r[10] or 0) > 1 for r in candidates):
        raise ValueError("automatic membership is not unique")
    return tuple(
        DirectSeedCandidateRow(r[6], r[7], r[8], r[9], r[11]) for r in candidates
    )


def candidate_rows(metadata, **options):
    try:
        if type(options["limit"]) is not int or not 1 <= options["limit"] <= 128:
            raise ValueError("native candidate limit is outside its hard cap")
        metadata.check()
        if type(options["deadline"]) not in (int, float) or not isfinite(
            options["deadline"]
        ):
            raise ValueError("native candidate deadline must be finite")
        if options["clock"] is not metadata.budget.clock:
            raise ValueError("native candidate clock changed")
        sql, params = statement(metadata, options)
        rows = metadata.read(
            sql,
            params,
            using=metadata.using,
            maximum=options["limit"] + 2,
            deadline=min(options["deadline"], metadata.budget.deadline),
            clock=options["clock"],
            alias=options["tier"] is DirectResolutionTier.ALIAS,
        )
        result = validate(
            rows, count=len(options["membership_states"]), limit=options["limit"]
        )
        metadata.check()
        return result
    except BaseException:
        metadata.failed = True
        raise


_NATIVE_STATEMENT_METHODS = (statement, validate, candidate_rows, _candidate_queryset)
