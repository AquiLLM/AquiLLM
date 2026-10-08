"""Long valid generation audits must fit without removing the maintenance cap."""

from types import SimpleNamespace

from apps.knowledge_graph.projection import reconciler, tasks
from apps.knowledge_graph.tests.test_projection_bounded_maintenance import (
    RedisBoundary,
    setup_pass,
)


def test_slow_valid_audit_completes_and_pass_stops_at_ninety_seconds(monkeypatch):
    redis = RedisBoundary()
    now = [0.0]
    completed, timeouts = [], []
    rows = {i: SimpleNamespace(state="ready", id=i) for i in range(1, 14)}
    enqueued = setup_pass(monkeypatch, rows, lambda: now[0])

    class Driver:
        def execute_read(self, *args, timeout_seconds, **kwargs):
            timeouts.append(timeout_seconds)
            now[0] += min(3.0, timeout_seconds)
            if timeout_seconds < 3.0:
                raise TimeoutError("transaction_timeout")
            return ()

    monkeypatch.setattr(
        reconciler, "_memgraph_repository", lambda: SimpleNamespace(_driver=Driver())
    )

    def audit(*, row, graph, **kwargs):
        # A valid large generation takes 51 seconds of individually bounded
        # reads. Another generation consumes the remaining 39 seconds.
        for _ in range(17 if row.id == 1 else 13):
            graph._driver.execute_read(
                "synthetic audit read", {}, timeout_seconds=5.0, max_records=1
            )
        completed.append(row.id)
        return SimpleNamespace(replay_reason=None)

    monkeypatch.setattr(reconciler, "_generation_audit", audit)
    monkeypatch.setattr(tasks.maintenance, "broker_client", lambda: redis)
    monkeypatch.setattr(tasks, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        tasks,
        "reconcile_projection_batch",
        lambda **kwargs: reconciler.reconcile_projection_batch(
            **kwargs, clock=lambda: now[0]
        ),
    )
    monkeypatch.setattr(
        tasks,
        "publish_projection_outbox",
        lambda **kwargs: SimpleNamespace(published_count=0, failed_count=0),
    )
    result = tasks.reconcile_knowledge_graph_projections.run()

    assert completed == [1, 2]
    assert (result["examined_count"], result["failure_count"]) == (2, 0)
    assert now[0] == 90.0
    assert timeouts == [5.0] * 29 + [3.0]
    assert enqueued == []
    assert all(row.state == "ready" for row in rows.values())
    for instant in (90, 120, 299):
        redis.now = instant
        assert tasks.maintenance.admit(redis, scope="global", interval=300) is None
    redis.now = 300
    successor = tasks.maintenance.admit(redis, scope="global", interval=300)
    assert successor.cursor == 2


def test_small_interval_admission_outlives_pass_and_fences_expired_owner():
    redis = RedisBoundary()
    owner = tasks.maintenance.admit(redis, scope="global", interval=60)
    assert owner.save_cursor(7)
    for instant in (60, 90, 119):
        redis.now = instant
        assert tasks.maintenance.admit(redis, scope="global", interval=60) is None
        assert owner.save_cursor(8)
    redis.now = 120
    successor = tasks.maintenance.admit(redis, scope="global", interval=60)
    assert successor.cursor == 8
    assert not owner.save_cursor(99)
    assert successor.save_cursor(9)
