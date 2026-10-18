"""Job cancellation and connection setup never reset projection authority."""

from threading import Event
from types import SimpleNamespace

import pytest

from apps.knowledge_graph.projection import database_deadline as deadlines
from apps.knowledge_graph.projection import runtime
from apps.knowledge_graph.projection.projection_lease import _ProjectionLeaseHeartbeat
from apps.knowledge_graph.tests.test_projection_runtime import _projection_environment
from apps.knowledge_graph.tests.test_projection_worker_heartbeat import (
    run_projection as run_projection,
)


def test_projection_driver_configures_bounded_connection_and_acquisition(monkeypatch):
    captured = []
    monkeypatch.setattr(
        runtime, "Neo4jMemgraphDriver", lambda *a, **k: captured.append(k)
    )
    monkeypatch.setattr(runtime, "MemgraphProjectionRepository", lambda driver: driver)
    runtime.memgraph_projection_repository(
        runtime.load_projection_runtime_settings(
            {**_projection_environment(), "KG_PROJECTION_TIMEOUT_MS": "12000"},
        )
    )
    assert captured[0]["connection_timeout"] <= 5.0
    assert captured[0]["connection_acquisition_timeout"] <= 5.0
    assert captured[0]["max_transaction_retry_time"] == 0.0


def test_projection_driver_passes_shorter_bounds_to_sdk(monkeypatch):
    from neo4j import GraphDatabase

    captured = []
    monkeypatch.setattr(GraphDatabase, "driver", lambda *a, **k: captured.append(k))
    repository = runtime.memgraph_projection_repository(
        runtime.load_projection_runtime_settings(
            {**_projection_environment(), "KG_PROJECTION_TIMEOUT_MS": "100"},
        )
    )
    repository._driver._connection()
    assert captured[0]["connection_timeout"] == 0.1
    assert captured[0]["connection_acquisition_timeout"] == 0.1
    assert captured[0]["max_transaction_retry_time"] == 0.0


def test_heartbeat_rejects_repository_replacement_after_capture():
    from apps.knowledge_graph.projection.projection_lease import _ProjectionLeaseLost

    calls = []
    with pytest.raises(TimeoutError):
        with deadlines.projection_operation(300000) as job:
            repository = SimpleNamespace(
                operation=job, renew=lambda **kw: calls.append(kw)
            )
            heartbeat = _ProjectionLeaseHeartbeat(repository, None, lease_seconds=300)
            repository.operation = deadlines.ProjectionOperation(300000)
            with pytest.raises(_ProjectionLeaseLost):
                heartbeat.pulse()
    assert calls == []


def test_heartbeat_loss_revokes_the_exact_captured_job(run_projection, monkeypatch):
    fixture = run_projection
    held, revoked = [], []
    finished = Event()
    original = _ProjectionLeaseHeartbeat._revoke

    def revoke(heartbeat):
        original(heartbeat)
        finished.set()

    monkeypatch.setattr(_ProjectionLeaseHeartbeat, "_revoke", revoke)

    def stage(**kwargs):
        job = deadlines.current_operation()
        held.append(job)
        fixture.state.loss = "owner"
        fixture.state.renewed.clear()
        assert fixture.state.renewed.wait(1)
        assert finished.wait(1)
        with pytest.raises(TimeoutError):
            job.remaining_ms()
        revoked.append(True)

    fixture.graph.write_staging_generation = stage
    assert not fixture.run().ready
    assert held and revoked == [True] and fixture.state.published == 0


def test_projection_call_scope_does_not_renew_shorter_outer_operation(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(deadlines, "monotonic", lambda: now[0])
    with deadlines.projection_operation(1000) as outer:
        with deadlines.projection_operation(300000, call_timeout_ms=5000) as inner:
            assert outer is inner
            assert inner.end() == 101.0
        now[0] += 0.5
        assert outer.remaining_ms() == 500


def test_heartbeat_captures_job_even_when_repository_is_a_test_adapter():
    with deadlines.projection_operation(300000) as job:
        heartbeat = _ProjectionLeaseHeartbeat(
            SimpleNamespace(), None, lease_seconds=300
        )
        assert heartbeat.operation is job


def test_heartbeat_rejects_repository_operation_replacement():
    other = deadlines.ProjectionOperation(300000)
    with deadlines.projection_operation(300000):
        with pytest.raises(ValueError, match="operation"):
            _ProjectionLeaseHeartbeat(
                SimpleNamespace(operation=other),
                None,
                lease_seconds=300,
            )


def test_owner_exit_revokes_a_retained_heartbeat_before_renewal():
    calls = []
    with deadlines.projection_operation(300000):
        heartbeat = _ProjectionLeaseHeartbeat(
            SimpleNamespace(renew=lambda **kwargs: calls.append(kwargs)),
            SimpleNamespace(projection_id=1, owner="owner"),
            lease_seconds=300,
        )
    with pytest.raises(TimeoutError):
        heartbeat.pulse()
    assert calls == []


def test_expiry_after_successful_ready_cas_never_requests_staging_retry(
    monkeypatch,
    run_projection,
):
    now = [100.0]
    monkeypatch.setattr(deadlines, "monotonic", lambda: now[0])
    run_projection.state.after_ready = lambda: now.__setitem__(0, 401.0)
    outcome = run_projection.run()
    assert outcome.ready is True and outcome.failure_code is None
    assert run_projection.state.status == "ready"
    assert run_projection.state.published == 1
    assert run_projection.state.failed == 0
