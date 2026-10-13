"""Fresh bounded fingerprint of the exact five-value membership source relation."""
from dataclasses import dataclass
import re

from django.db import connections

SOURCE_FINGERPRINT_FORMAT = "kg-membership-source-sql-v1"


@dataclass(frozen=True, slots=True)
class MembershipSourceFingerprintV1:
    collection_id: int
    input_sha256: str
    entity_count: int
    source_rows: int
    duplicate_links: bool


# Keep framing/predicates identical to the reviewed selected-ANY cost experiment.
# In particular artifact.scope_id is deliberately NOT an eligibility predicate.
_SQL = """
WITH selected AS (
 SELECT pg_catalog.unnest(%s::bigint[]) AS collection_id
), eligible_entities AS MATERIALIZED (
 SELECT entity.id AS entity_id, entity.collection_id,
        artifact.resolver_version, artifact.resolution_config_checksum
 FROM public.apps_knowledge_graph_collectionentity entity
 JOIN public.apps_knowledge_graph_graphartifact artifact ON artifact.id = entity.artifact_id
 JOIN selected ON selected.collection_id = entity.collection_id
 WHERE entity.collection_id = ANY(%s::bigint[]) AND entity.status = 'active'
   AND artifact.status = 'active'
   AND artifact.scope_type = 'collection' AND artifact.evaluation_only = false
), eligible_links AS MATERIALIZED (
 SELECT link.id AS link_id, link.collection_entity_id, link.canonical_entity_id AS canonical_id,
        link.decision_checksum
 FROM public.apps_knowledge_graph_canonicalentitylink link
 JOIN public.apps_knowledge_graph_canonicalentity canonical ON canonical.id = link.canonical_entity_id
 JOIN eligible_entities entity ON entity.entity_id = link.collection_entity_id
 WHERE link.outcome = 'automatic' AND link.status = 'active' AND canonical.status = 'active'
), expanded AS (
 SELECT entity.*, link.link_id, link.canonical_id, link.decision_checksum,
        pg_catalog.count(link.link_id) OVER (PARTITION BY entity.entity_id) AS link_count
 FROM eligible_entities entity
 LEFT JOIN eligible_links link ON link.collection_entity_id = entity.entity_id
)
SELECT selected.collection_id, pg_catalog.count(DISTINCT expanded.entity_id) AS entity_count,
       pg_catalog.count(expanded.entity_id) AS source_rows,
       COALESCE(pg_catalog.bool_or(expanded.link_count > 1), false) AS duplicate_links,
       pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(
         'kg-membership-source-sql-v1:' || COALESCE(
           pg_catalog.jsonb_agg(pg_catalog.jsonb_build_array(
             expanded.entity_id, expanded.resolver_version, expanded.resolution_config_checksum,
             expanded.canonical_id, expanded.decision_checksum)
             ORDER BY expanded.entity_id, expanded.link_id)
             FILTER (WHERE expanded.entity_id IS NOT NULL), '[]'::jsonb)::text,
         'UTF8')), 'hex') AS input_sha256,
       pg_catalog.current_setting('transaction_isolation') AS isolation
FROM selected LEFT JOIN expanded ON expanded.collection_id = selected.collection_id
GROUP BY selected.collection_id ORDER BY selected.collection_id
"""


def load_membership_source_fingerprints(
    *, collection_ids: tuple[int, ...], using: str, check, require_fresh: bool = True,
) -> dict[int, MembershipSourceFingerprintV1]:
    """Use the caller's original operation fence and existing database deadlines."""
    check()
    if (type(collection_ids) is not tuple or len(collection_ids) > 128
            or any(type(i) is not int or not 1 <= i < 2**63 for i in collection_ids)
            or collection_ids != tuple(sorted(set(collection_ids)))):
        raise ValueError("collection_ids must be sorted unique positive signed-64 integers, max 128")
    if type(using) is not str or not using:
        raise ValueError("using must be a nonempty database alias")
    if type(require_fresh) is not bool:
        raise ValueError("require_fresh must be a boolean")
    if not collection_ids:
        check()
        return {}
    database = connections[using]
    if database.vendor != "postgresql":
        raise ValueError("membership source fingerprint requires PostgreSQL")
    check()
    with database.cursor() as cursor:
        check()
        cursor.execute(_SQL, [list(collection_ids), list(collection_ids)])
        check()
        rows = cursor.fetchmany(len(collection_ids) + 1)
        check()
    check()
    if len(rows) != len(collection_ids):
        raise ValueError("membership fingerprint collection coverage invalid")
    expected_isolation = "read committed" if require_fresh else "repeatable read"
    result = {}
    for expected, row in zip(collection_ids, rows):
        check()
        if type(row) not in (tuple, list) or len(row) != 6:
            raise ValueError("membership fingerprint row invalid")
        collection, entities, source_rows, duplicate, digest, isolation = row
        if isolation != expected_isolation:
            raise ValueError("membership fingerprint transaction isolation invalid")
        if (type(collection) is not int or collection != expected
                or type(entities) is not int or type(source_rows) is not int
                or not 0 <= entities <= source_rows or type(duplicate) is not bool
                or (entities == 0 and source_rows != 0)
                or duplicate != (source_rows > entities)
                or type(digest) is not str or re.fullmatch(r"[a-f0-9]{64}", digest) is None):
            raise ValueError("membership fingerprint value invalid")
        result[collection] = MembershipSourceFingerprintV1(
            collection, digest, entities, source_rows, duplicate,
        )
        check()
    check()
    return result


# Defining-time evidence: a lazy serving import must not canonize replacements.
_NATIVE_MEMBERSHIP_SOURCE_BINDING = (
    load_membership_source_fingerprints, _SQL, MembershipSourceFingerprintV1,
)
