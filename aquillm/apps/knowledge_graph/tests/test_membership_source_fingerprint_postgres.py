"""Real SQL differential evidence against the unchanged full membership function."""
import hashlib
import json
import os
from contextlib import contextmanager

import pytest
from django.db import connection, transaction
from django.test.utils import CaptureQueriesContext

from apps.collections.models import Collection
from apps.knowledge_graph.projection import memberships
from apps.knowledge_graph.tests.test_projection_canonical_resolver_postgres import (
    CODEC, source_rows,  # noqa: F401
)
from apps.knowledge_graph.tests.test_membership_source_fingerprint import reader

pytestmark = [pytest.mark.django_db(transaction=True), pytest.mark.skipif(
    os.environ.get("KG_REQUIRE_POSTGRES_TESTS") != "1", reason="requires PostgreSQL",
)]


def capture(ids):
    return reader().load_membership_source_fingerprints(
        collection_ids=ids, using="default", check=lambda: None,
    )


def full(ids):
    return memberships.membership_decision_checksum(
        memberships.load_automatic_membership_assignments(
            collection_ids=ids, using="default", batch_size=100, codec=CODEC,
            check=lambda: None,
        ), check=lambda: None,
    )


def expected_digest(rows):
    # PostgreSQL jsonb text uses comma/colon spaces and literal Unicode.
    text = json.dumps(rows, ensure_ascii=False, separators=(", ", ": "))
    return hashlib.sha256(("kg-membership-source-sql-v1:" + text).encode("utf-8")).hexdigest()


@contextmanager
def corrupt_source_checks(table):
    """Admit legacy malformed inputs only inside a rolled-back test transaction."""
    from psycopg2 import sql

    name = "apps_knowledge_graph_" + table
    with transaction.atomic():
        with connection.cursor() as cursor:
            cursor.execute("SELECT conname FROM pg_catalog.pg_constraint "
                           "WHERE conrelid=%s::regclass AND contype='c'", [name])
            for (constraint,) in cursor.fetchall():
                cursor.execute(sql.SQL("ALTER TABLE {} DROP CONSTRAINT {}").format(
                    sql.Identifier(name), sql.Identifier(constraint),
                ))
        try:
            yield
        finally:
            transaction.set_rollback(True)


def test_exact_source_rows_empty_collection_and_statement_count(source_rows):
    entity, canonical, link, projection, checksum = source_rows
    empty = Collection.objects.create(name="empty source proof")
    ids = tuple(sorted((entity.collection_id, empty.pk)))
    old = memberships._load_membership_source_rows(
        collection_ids=ids, using="default", batch_size=100, check=lambda: None,
    )
    with CaptureQueriesContext(connection) as queries:
        result = capture(ids)
    assert len(queries) == 1
    assert result[entity.collection_id].input_sha256 == expected_digest(old)
    assert result[entity.collection_id].entity_count == len(old) == 1
    assert not result[entity.collection_id].duplicate_links
    assert result[empty.pk].input_sha256 == expected_digest([])
    assert result[empty.pk].entity_count == result[empty.pk].source_rows == 0
    assert full((entity.collection_id,)) == checksum


@pytest.mark.parametrize("table,field,value", [
    ("collectionentity", "status", "superseded"),
    ("graphartifact", "status", "superseded"),
    ("graphartifact", "scope_type", "document"),
    ("graphartifact", "evaluation_only", True),
    ("graphartifact", "resolver_version", 'resolver-雪\\"\n'),
    ("graphartifact", "resolution_config_checksum", "d" * 64),
    ("canonicalentitylink", "status", "superseded"),
    ("canonicalentitylink", "outcome", "manual"),
    ("canonicalentitylink", "decision_checksum", "e" * 64),
    ("canonicalentity", "status", "superseded"),
])
def test_changed_full_membership_cannot_preserve_hash(source_rows, table, field, value):
    entity, canonical, link, projection, checksum = source_rows
    ids = (entity.collection_id,)
    before, old_full = capture(ids), full(ids)
    pk = {"collectionentity": entity.pk, "graphartifact": entity.artifact_id,
          "canonicalentitylink": link.pk, "canonicalentity": canonical.pk}[table]
    with corrupt_source_checks(table):
        with connection.cursor() as cursor:
            cursor.execute(f'UPDATE public.apps_knowledge_graph_{table} SET {field}=%s WHERE id=%s',
                           [value, pk])
        after = capture(ids)
        assert after[ids[0]].input_sha256 != before[ids[0]].input_sha256
        if field == "resolver_version":
            with pytest.raises(ValueError, match="canonical token"):
                full(ids)
        else:
            assert full(ids) != old_full
        rows = memberships._load_membership_source_rows(
            collection_ids=ids, using="default", batch_size=100, check=lambda: None,
        )
        assert after[ids[0]].input_sha256 == expected_digest(rows)


def test_cross_scope_artifact_mapping_is_preserved_and_null_explicit(source_rows):
    entity, canonical, link, projection, checksum = source_rows
    ids = (entity.collection_id,)
    with connection.cursor() as cursor:
        cursor.execute('UPDATE public.apps_knowledge_graph_graphartifact SET scope_id=%s WHERE id=%s',
                       ["9223372036854775807", entity.artifact_id])
        cursor.execute('DELETE FROM public.apps_knowledge_graph_canonicalentitylink WHERE id=%s', [link.pk])
    rows = memberships._load_membership_source_rows(
        collection_ids=ids, using="default", batch_size=100, check=lambda: None,
    )
    assert rows[0][3:] == (None, None)
    assert capture(ids)[ids[0]].input_sha256 == expected_digest(rows)


def test_actual_transaction_isolation_is_checked(source_rows):
    entity, *_ = source_rows
    ids = (entity.collection_id,)
    with transaction.atomic():
        with connection.cursor() as cursor:
            cursor.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        with pytest.raises(ValueError, match="isolation"):
            capture(ids)
        result = reader().load_membership_source_fingerprints(
            collection_ids=ids, using="default", check=lambda: None, require_fresh=False,
        )
        assert result[ids[0]].entity_count == 1


def test_duplicate_links_are_preserved_as_rejection_evidence(source_rows):
    entity, canonical, link, *_ = source_rows
    ids = (entity.collection_id,)
    before = capture(ids)[ids[0]]
    # Different resolver is legal to SQL but bypasses ORM registry-version checks.
    with connection.cursor() as cursor:
        cursor.execute("INSERT INTO public.apps_knowledge_graph_canonicalentitylink "
                       "(score,method,resolver_version,status,reason,metadata,created_at,"
                       "canonical_entity_id,collection_entity_id,decision_checksum,outcome) "
                       "SELECT score,method,%s,status,reason,metadata,created_at,"
                       "canonical_entity_id,collection_entity_id,decision_checksum,outcome "
                       "FROM public.apps_knowledge_graph_canonicalentitylink WHERE id=%s",
                       ["canonical-resolution-v2", link.pk])
    after = capture(ids)[ids[0]]
    assert after.duplicate_links and after.entity_count == 1 and after.source_rows == 2
    assert after.input_sha256 != before.input_sha256
    with pytest.raises(ValueError, match="not unique"):
        full(ids)


def test_orm_canonical_transition_observes_explicit_null(source_rows):
    entity, canonical, *_ = source_rows
    ids = (entity.collection_id,)
    before = capture(ids)[ids[0]]
    with transaction.atomic():
        canonical._transition_registry_status_locked((canonical.pk,), target="superseded")
    rows = memberships._load_membership_source_rows(
        collection_ids=ids, using="default", batch_size=100, check=lambda: None,
    )
    after = capture(ids)[ids[0]]
    assert rows[0][3:] == (None, None)
    assert after.input_sha256 == expected_digest(rows) != before.input_sha256


def test_large_entity_ids_preserve_integer_order_and_full_inputs(source_rows):
    entity, canonical, link, *_ = source_rows
    ids = (entity.collection_id,)
    old = capture(ids)[ids[0]]
    with transaction.atomic():
        with connection.cursor() as cursor:
            cursor.execute("UPDATE public.apps_knowledge_graph_collectionentity SET id=%s WHERE id=%s",
                           [2**63 - 1, entity.pk])
            cursor.execute("UPDATE public.apps_knowledge_graph_canonicalentitylink "
                           "SET collection_entity_id=%s WHERE id=%s", [2**63 - 1, link.pk])
        rows = memberships._load_membership_source_rows(
            collection_ids=ids, using="default", batch_size=100, check=lambda: None,
        )
        assert rows[0][0] == 2**63 - 1
        assert capture(ids)[ids[0]].input_sha256 == expected_digest(rows) != old.input_sha256
        transaction.set_rollback(True)


def test_empty_text_and_explicit_null_are_distinct(source_rows):
    entity, canonical, link, *_ = source_rows
    ids = (entity.collection_id,)
    with corrupt_source_checks("canonicalentitylink"):
        with connection.cursor() as cursor:
            cursor.execute("UPDATE public.apps_knowledge_graph_canonicalentitylink "
                           "SET decision_checksum='' WHERE id=%s", [link.pk])
        empty_text = capture(ids)[ids[0]].input_sha256
        with pytest.raises(ValueError):
            full(ids)
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM public.apps_knowledge_graph_canonicalentitylink WHERE id=%s", [link.pk])
        assert capture(ids)[ids[0]].input_sha256 != empty_text
