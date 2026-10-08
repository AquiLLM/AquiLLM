"""Registry provenance scans must avoid the corpus-wide nested-loop plan."""

import pytest
from django.db import DatabaseError, connection, transaction

from apps.collections.models import Collection
from apps.knowledge_graph.resolution import canonical
from apps.knowledge_graph.tests.test_canonical_resolution import (
    _create_active_collection_entity,
)

pytestmark = pytest.mark.django_db


def _planner_settings():
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT current_setting('join_collapse_limit'), "
            "current_setting('enable_nestloop')"
        )
        return cursor.fetchone()


@pytest.fixture
def source_entity():
    if connection.vendor != "postgresql":
        pytest.skip("PostgreSQL planner regression")
    return _create_active_collection_entity(
        Collection.objects.create(name="registry planning fixture"),
        label="Atlas",
        cluster_digit="a",
    )


def _load(artifact, entity):
    return canonical._load_locked_canonical_inputs(
        entity_rows=(entity,),
        active_artifact_ids=(artifact.pk,),
        using="default",
    )


def test_registry_scan_uses_bounded_join_plan_without_leaking_to_caller(source_entity):
    artifact, entity = source_entity
    before = _planner_settings()
    observed = []

    def observe(execute, sql, params, many, context):
        if "apps_knowledge_graph_collectionentitydocumentlink" in sql:
            observed.append(_planner_settings())
        return execute(sql, params, many, context)

    with connection.execute_wrapper(observe):
        inputs = _load(artifact, entity)

    assert [(item.entity_id, item.normalized_label) for item in inputs] == [
        (entity.pk, "atlas")
    ]
    assert observed == [("1", "off")]
    assert _planner_settings() == before


@pytest.mark.parametrize("database_failure", [False, True])
def test_registry_scan_failure_restores_caller_transaction(
    source_entity, database_failure
):
    artifact, entity = source_entity
    with transaction.atomic():
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL join_collapse_limit=7")
            cursor.execute("SET LOCAL enable_nestloop=on")

        def fail_scan(execute, sql, params, many, context):
            if "apps_knowledge_graph_collectionentitydocumentlink" in sql:
                if database_failure:
                    return execute("SELECT 1 / 0", (), False, context)
                raise ValueError("simulated provenance validation failure")
            return execute(sql, params, many, context)

        expected = DatabaseError if database_failure else ValueError
        with connection.execute_wrapper(fail_scan), pytest.raises(expected):
            _load(artifact, entity)

        assert _planner_settings() == ("7", "on")
        assert Collection.objects.filter(pk=entity.collection_id).exists()
