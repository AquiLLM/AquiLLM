"""Captured membership IDs do not become thousands of ORM expressions."""

import pytest
from django.core.exceptions import EmptyResultSet
from django.db import connection, transaction
from django.db.models.query import QuerySet

from apps.documents.services.source_deadline import source_preparation_scope
from apps.knowledge_graph.models import CanonicalEntityLink, CollectionEntity
from apps.knowledge_graph.projection import memberships
from apps.knowledge_graph.projection.membership_source_query import captured_entity_links
from apps.knowledge_graph.tests.test_projection_canonical_resolver_postgres import (
    source_rows,  # noqa: F401 - shared persisted PostgreSQL fixture
)


def test_large_captured_membership_ids_use_one_bound_array(monkeypatch):
    identifiers = tuple(range(1, 23510))
    captured = []

    def rows(queryset, **kwargs):
        if queryset.model is CollectionEntity:
            return iter((pk, "resolver-v1", "a" * 64) for pk in identifiers)
        assert queryset.model is CanonicalEntityLink
        captured.append(queryset.query.sql_with_params())
        return iter(())

    monkeypatch.setattr(QuerySet, "iterator", rows)
    result = memberships._load_membership_source_rows(
        collection_ids=(226,), using="default", batch_size=1000, check=lambda: None,
    )
    assert tuple(row[0] for row in result) == identifiers
    assert all(row[3:] == (None, None) for row in result)
    assert len(captured) == 1
    sql, params = captured[0]
    assert len(params) <= 4  # Original query has 23,512 scalar parameters.
    assert "ANY(" in sql and "::bigint[]" in sql
    assert [tuple(value) for value in params if type(value) is list] == [identifiers]


def test_non_postgres_keeps_original_lookup(monkeypatch):
    monkeypatch.setattr(connection, "vendor", "sqlite")
    ids = (1, 2, 3)
    query = CanonicalEntityLink.objects.all()
    assert captured_entity_links(query, ids).query.sql_with_params() == (
        query.filter(collection_entity_id__in=ids).query.sql_with_params()
    )


def test_empty_capture_keeps_zero_query_short_circuit():
    query = captured_entity_links(CanonicalEntityLink.objects.all(), ())
    with pytest.raises(EmptyResultSet):
        query.query.sql_with_params()


@pytest.mark.parametrize("ids", ([1, 2], (True,), ("1",), (None,)))
def test_non_native_values_keep_original_lookup(ids):
    query = CanonicalEntityLink.objects.all()
    actual = captured_entity_links(query, ids)
    expected = query.filter(collection_entity_id__in=ids)
    # EmptyResultSet for a null-only list is the same legacy behavior.
    assert str(actual.query.where) == str(expected.query.where)


@pytest.mark.django_db(transaction=True)
def test_postgres_array_preserves_captured_ids_and_live_canonical_status(request):
    entity, canonical, link, _projection, _checksum = request.getfixturevalue("source_rows")
    query = CanonicalEntityLink.objects.filter(
        outcome="automatic", status="active", canonical_entity__status="active",
    ).order_by("collection_entity_id", "pk")
    variants = ((), (entity.pk,), (entity.pk, entity.pk),
                (-(2**63), 0, entity.pk, 2**63 - 1), (entity.pk + 100000,))
    for ids in variants:
        actual = tuple(captured_entity_links(query, ids).values_list("pk", flat=True))
        expected = tuple(query.filter(collection_entity_id__in=ids).values_list("pk", flat=True))
        assert actual == expected
    assert tuple(captured_entity_links(query, (entity.pk,)).values_list("pk", flat=True)) == (link.pk,)
    with transaction.atomic():
        canonical._transition_registry_status_locked((canonical.pk,), target="superseded")
    assert tuple(captured_entity_links(query, (entity.pk,)).values_list("pk", flat=True)) == ()
    with transaction.atomic():
        canonical._transition_registry_status_locked((canonical.pk,), target="active")
    assert tuple(captured_entity_links(query, (entity.pk,)).values_list("pk", flat=True)) == (link.pk,)


def test_duplicate_active_membership_is_still_rejected(monkeypatch):
    def rows(queryset, **kwargs):
        if queryset.model is CollectionEntity:
            return iter(((7, "resolver-v1", "a" * 64),))
        return iter(((7, 10, "b" * 64), (7, 11, "c" * 64)))

    monkeypatch.setattr(QuerySet, "iterator", rows)
    with pytest.raises(ValueError, match="not unique"):
        memberships._load_membership_source_rows(
            collection_ids=(1,), using="default", batch_size=1000, check=lambda: None,
        )


@pytest.mark.parametrize("burst_size", (1, 5000))
def test_fetch_returning_after_deadline_cannot_publish_rows(monkeypatch, burst_size):
    now, queries, returned = [0.0], [], []

    def rows(queryset, *, chunk_size):
        # A generator resumes after _checked_rows' pre-fetch deadline check.
        queries.append(queryset.model)
        assert chunk_size == 5000
        burst = tuple((pk, "resolver-v1", "a" * 64) for pk in range(1, burst_size + 1))
        now[0] = 2.0
        yield from burst

    monkeypatch.setattr(QuerySet, "iterator", rows)
    with pytest.raises(TimeoutError):
        with source_preparation_scope(1.0, lambda: now[0]):
            returned.append(memberships._load_membership_source_rows(
                collection_ids=(226,), using="default", batch_size=5000,
            ))
    assert returned == []
    assert queries == [CollectionEntity]
