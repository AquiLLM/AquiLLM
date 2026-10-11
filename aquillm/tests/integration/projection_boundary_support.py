"""Actual persisted producers and restricted-role calls for the projection boundary."""
import pytest
from django.db import connections
from django.utils import timezone

from apps.knowledge_graph.models import CollectionGraphProjection
from apps.knowledge_graph.projection.state_repository import FunctionProjectionStateRepository
from tests.integration.persisted_graph_projection_support import graph_configuration
from tests.integration.persisted_graph_role_support import projection_roles
from tests.integration.persisted_graph_source_support import graph_source, upload

FIRST = "Orion uses MMLU. Original observation."


@pytest.fixture
def boundary_projection(graph_source, projection_roles):
    graph_source.activate_and_build_document(FIRST)
    artifact = graph_source.build_emitted_collection()
    row = CollectionGraphProjection.objects.get(artifact=artifact, state="pending")
    assert row.representation_snapshot is not None
    return graph_source, row


def state_repository(owner="boundary-worker", **kwargs):
    return FunctionProjectionStateRepository(
        state_using="projection_state", source_using="projection_source", owner=owner, **kwargs)


def claim(row, owner="boundary-worker"):
    repository = state_repository(owner)
    assert repository.claim(projection_id=row.pk, owner=owner, now=timezone.now(),
                            lease_seconds=60) is not None
    row.refresh_from_db()
    assert row.state == "building" and row.lease_owner == owner
    return repository


def source_call(row, *, coordinates=False, **changes):
    values = dict(projection_id=row.pk, generation_key=row.generation_key,
                  representation_digest=row.representation_digest,
                  purpose="build", lease_owner="boundary-worker")
    values.update(changes)
    name = "chunks" if coordinates else "current"
    with connections["projection_source"].cursor() as cursor:
        cursor.execute(
            f"SELECT * FROM public.kg_projection_representation_{name}_v1(%s,%s,%s,%s,%s)",
            tuple(values.values()))
        rows = cursor.fetchall()
        fields = tuple(column.name for column in cursor.description)
    return (fields, rows) if coordinates else rows[0][0]

