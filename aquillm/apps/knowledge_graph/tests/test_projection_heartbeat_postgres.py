from __future__ import annotations

import os
from datetime import timedelta
from threading import current_thread
from time import sleep

import pytest
from django.db import connection, transaction
from django.utils import timezone

from apps.collections.models import Collection
from apps.knowledge_graph.projection.state_repository import (
    FunctionProjectionStateRepository,
)
from apps.knowledge_graph.projection.worker import (
    _ProjectionLeaseHeartbeat,
    _ProjectionLeaseLost,
)
from apps.knowledge_graph.tests.test_projection_locking_postgres import (
    _active_artifact,
    _building_projection,
)

pytestmark = [
    pytest.mark.django_db(transaction=True),
    pytest.mark.skipif(
        os.environ.get("KG_REQUIRE_POSTGRES_TESTS") != "1",
        reason="requires disposable PostgreSQL state-role functions",
    ),
]


@pytest.mark.parametrize("takeover", [False, True])
def test_heartbeat_renews_under_state_role_and_closes_thread_connection(
    takeover, monkeypatch
):
    monkeypatch.setenv("KG_PROJECTION_IDENTIFIER_KEY_VERSION", "heartbeat-test-v1")
    assert connection.vendor == "postgresql"
    collection = Collection.objects.create(name="Projection heartbeat fixture")
    artifact = _active_artifact(collection)
    now = timezone.now()
    _membership, row = _building_projection(collection, artifact, now)
    row.lease_expires_at = now + timedelta(seconds=1)
    row.save(update_fields=["lease_expires_at"])
    original_expiry = row.lease_expires_at
    thread_connections = []

    class RestrictedState(FunctionProjectionStateRepository):
        def _one(self, operation, parameters):
            with transaction.atomic():
                with connection.cursor() as cursor:
                    cursor.execute("SET LOCAL ROLE aquillm_projection_state")
                    if current_thread().name.startswith("projection-lease-"):
                        thread_connections.append(connection.connection)
                    return super()._one(operation, parameters)

    repository = RestrictedState(state_using="default", source_using="default")
    lease = repository.claim(
        projection_id=row.pk, owner="race-worker", now=now, lease_seconds=1
    )
    heartbeat = _ProjectionLeaseHeartbeat(repository, lease, lease_seconds=1)
    if takeover:
        with pytest.raises(_ProjectionLeaseLost):
            with heartbeat:
                type(row).objects.filter(pk=row.pk).update(lease_owner="new-owner")
                assert heartbeat._stop.wait(2.0)
                heartbeat.check()
        row.refresh_from_db()
        assert row.lease_owner == "new-owner"
        assert row.state == "building"
    else:
        with heartbeat:
            sleep(1.25)
            assert timezone.now() > original_expiry
            heartbeat.check()
        row.refresh_from_db()
        assert row.lease_owner == "race-worker"
        assert row.lease_expires_at > timezone.now()
        assert row.attempt_count == lease.attempt_count == 1
    assert thread_connections
    assert all(item.closed for item in thread_connections)
