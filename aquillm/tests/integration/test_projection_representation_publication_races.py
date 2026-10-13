"""Final state publication cannot wait indefinitely or publish after source replacement."""
from concurrent.futures import ThreadPoolExecutor

import pytest
from django.db import OperationalError, connections, transaction

from apps.collections.models import Collection
from apps.knowledge_graph.projection.database_deadline import ProjectionOperation, projection_operation
from tests.integration.projection_boundary_support import (
    boundary_projection, claim, graph_configuration, graph_source, projection_roles,
    source_call, state_repository, upload,
)
from tests.integration.test_projection_representation_state_boundary import ready_parameters

pytestmark = pytest.mark.django_db(
    transaction=True, databases=["default", "projection_source", "projection_state"])


def test_final_state_function_wait_is_bounded_and_restores_connection(boundary_projection):
    _, row = boundary_projection
    claim(row)
    operation = ProjectionOperation(1500)

    def publish():
        alias = "projection_state"
        try:
            with connections[alias].cursor() as cursor:
                cursor.execute("SELECT set_config('lock_timeout','50ms',false)")
            repository = state_repository(operation=operation)
            with pytest.raises(OperationalError, match="lock timeout|statement timeout"):
                repository._one("ready", tuple(ready_parameters(row)))
            with connections[alias].cursor() as cursor:
                cursor.execute("SELECT current_setting('lock_timeout'),"
                               "current_setting('statement_timeout')")
                assert cursor.fetchone() == ("50ms", "0")
            assert operation.remaining_ms() > 0
        finally:
            connections[alias].close()

    with ThreadPoolExecutor(max_workers=1) as pool:
        with transaction.atomic():
            Collection.objects.select_for_update().get(pk=row.collection_id)
            completed = pool.submit(publish)
            completed.result(timeout=4)
    row.refresh_from_db()
    assert row.state == "building" and row.ready_at is None


def test_new_generation_after_coordinate_read_blocks_original_state_cas(boundary_projection):
    source, row = boundary_projection
    claim(row)
    with projection_operation() as operation:
        repository = state_repository(operation=operation)
        fields, chunks = source_call(row, coordinates=True)
        assert fields == ("id", "doc_id", "chunk_number") and chunks
        source.activate_only("Orion uses MMLU. Changed after original coordinate read.")
        assert operation.remaining_ms() > 100
        result = repository._one("ready", tuple(ready_parameters(row)))
        assert result is not None and result["published"] is False
        assert operation.remaining_ms() > 0
    row.refresh_from_db()
    assert row.state != "ready"
