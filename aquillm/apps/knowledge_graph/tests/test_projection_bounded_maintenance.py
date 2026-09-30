from datetime import timedelta
from types import SimpleNamespace

import pytest
from django.utils import timezone

from apps.knowledge_graph.projection import reconciler, tasks
from apps.knowledge_graph.projection.memgraph_driver import MemgraphDriverError


class RedisBoundary:
    def __init__(self):
        self.now = 0
        self.values = {}
        self.expires = {}

    def get(self, key):
        if self.expires.get(key, float("inf")) <= self.now:
            self.values.pop(key, None)
        return self.values.get(key)

    def set(self, key, value, nx=False, ex=None):
        if nx and self.get(key) is not None:
            return False
        self.values[key] = str(value)
        if ex:
            self.expires[key] = self.now + ex
        return True

    def eval(self, script, count, lock, cursor, token, value):
        assert count == 2
        if self.get(lock) != token:
            return 0
        self.set(cursor, value)
        return 1


def maintenance_module():
    assert hasattr(tasks, "maintenance"), (
        "scheduled task needs distributed bounded maintenance"
    )
    return tasks.maintenance


def test_admission_skips_old_duplicates_recovers_after_ttl_and_fences_cursor():
    maintenance = maintenance_module()
    redis = RedisBoundary()
    first = maintenance.admit(redis, scope="global", interval=300)
    assert first is not None
    assert maintenance.admit(redis, scope="global", interval=300) is None
    assert first.save_cursor(11)
    assert maintenance.admit(redis, scope="collection:7", interval=300).cursor == 0
    redis.now = 301
    second = maintenance.admit(redis, scope="global", interval=300)
    assert second.cursor == 11
    assert not first.save_cursor(99)
    assert second.save_cursor(12)


def setup_pass(monkeypatch, rows, clock):
    monkeypatch.setattr(
        reconciler,
        "_projection_settings",
        lambda: SimpleNamespace(
            projection_timeout_ms=5000,
            projection_schema_version="v1",
            projection_format_version="v1",
            projection_identifier_key_version="v1",
        ),
    )
    monkeypatch.setattr(reconciler, "_postgres_repository", lambda: object())
    monkeypatch.setattr(
        reconciler, "_memgraph_repository", lambda: SimpleNamespace(_driver=object())
    )
    monkeypatch.setattr(
        reconciler, "projection_identifier_codec", lambda settings: object()
    )
    monkeypatch.setattr(
        reconciler,
        "_active_artifact_page",
        lambda after_id, page_size, collection_id: tuple(
            (k, k)
            for k in rows
            if k > after_id and (collection_id is None or k == collection_id)
        )[:page_size],
    )
    monkeypatch.setattr(
        reconciler,
        "_projection_for_active",
        lambda collection_id, artifact_id: rows[artifact_id],
    )
    enqueued = []
    monkeypatch.setattr(
        reconciler,
        "_replay_projection",
        lambda **kw: enqueued.append(kw["artifact_id"]),
    )
    return enqueued


def test_bounded_pass_advances_wraps_and_recovers_missing_and_expired(monkeypatch):
    assert hasattr(reconciler, "reconcile_projection_batch")
    rows = dict.fromkeys(range(1, 14))
    rows[2] = SimpleNamespace(
        state="building",
        schema_version="v1",
        projection_version="v1",
        identifier_key_version="v1",
        lease_expires_at=timezone.now() - timedelta(seconds=1),
    )
    enqueued = setup_pass(monkeypatch, rows, lambda: 0)
    first = reconciler.reconcile_projection_batch(
        after_id=0, page_size=100, deadline=30, clock=lambda: 0
    )
    assert (first.examined_count, first.next_cursor) == (10, 10)
    second = reconciler.reconcile_projection_batch(
        after_id=10, page_size=100, deadline=30, clock=lambda: 0
    )
    assert (second.examined_count, second.next_cursor) == (3, 0)
    assert enqueued == list(range(1, 14))


def test_audit_failure_preserves_authority_and_progress(monkeypatch):
    assert hasattr(reconciler, "reconcile_projection_batch")
    rows = {1: SimpleNamespace(state="ready"), 2: None}
    enqueued = setup_pass(monkeypatch, rows, lambda: 0)
    real_audit = reconciler._generation_audit

    def audit(**kwargs):
        if kwargs["row"] is not None:
            raise MemgraphDriverError("memgraph_timeout")
        return real_audit(**kwargs)

    monkeypatch.setattr(reconciler, "_generation_audit", audit)
    result = reconciler.reconcile_projection_batch(
        after_id=0, page_size=10, deadline=30, clock=lambda: 0
    )
    assert (result.examined_count, result.failure_count, result.next_cursor) == (
        2,
        1,
        0,
    )
    assert rows[1].state == "ready"
    assert enqueued == [2]


def test_each_transaction_uses_remaining_deadline_and_stops_at_expiry():
    maintenance = maintenance_module()
    now = [0.0]
    timeouts = []

    class Driver:
        def execute_read(self, *args, timeout_seconds, **kwargs):
            timeouts.append(timeout_seconds)
            now[0] += 4
            return ()

        execute_write = execute_read

    driver = maintenance.DeadlineDriver(
        Driver(), deadline=10, timeout_seconds=5, clock=lambda: now[0]
    )
    for _ in range(3):
        driver.execute_read("query", {}, timeout_seconds=100, max_records=10)
    with pytest.raises(TimeoutError):
        driver.execute_read("query", {}, timeout_seconds=100, max_records=10)
    assert timeouts == [5, 5, 2]


def test_pass_stops_before_next_artifact_at_overall_deadline(monkeypatch):
    assert hasattr(reconciler, "reconcile_projection_batch")
    now = [0]
    enqueued = setup_pass(monkeypatch, dict.fromkeys(range(1, 14)), lambda: now[0])
    real_audit = reconciler._generation_audit

    def audit(**kwargs):
        now[0] += 10
        return real_audit(**kwargs)

    monkeypatch.setattr(reconciler, "_generation_audit", audit)
    result = reconciler.reconcile_projection_batch(
        after_id=0, page_size=10, deadline=30, clock=lambda: now[0]
    )
    assert result.examined_count == 3
    assert result.next_cursor == 3
    assert len(enqueued) <= 3


def test_coordination_failure_returns_fixed_result_without_retry_or_outbox(monkeypatch):
    maintenance = maintenance_module()
    monkeypatch.setattr(
        maintenance,
        "broker_client",
        lambda: (_ for _ in ()).throw(ConnectionError("secret")),
    )
    monkeypatch.setattr(
        tasks,
        "publish_projection_outbox",
        lambda **kwargs: pytest.fail("must fail closed"),
    )
    monkeypatch.setattr(
        tasks.reconcile_knowledge_graph_projections,
        "retry",
        lambda **kw: pytest.fail("retry flood"),
    )
    result = tasks.reconcile_knowledge_graph_projections.run()
    assert result["failure_code"] == "maintenance_coordination_unavailable"
    assert result["examined_count"] == 0
    assert "secret" not in repr(result)


def test_old_task_duplicates_skip_and_full_outbox_does_not_reschedule(monkeypatch):
    redis = RedisBoundary()
    monkeypatch.setattr(tasks.maintenance, "broker_client", lambda: redis)
    enqueued = setup_pass(monkeypatch, dict.fromkeys(range(1, 14)), lambda: 0)
    publications = []

    def publish(**kwargs):
        publications.append(kwargs["limit"])
        return SimpleNamespace(published_count=10, attempted_count=10, failed_count=0)

    monkeypatch.setattr(tasks, "publish_projection_outbox", publish)
    monkeypatch.setattr(
        tasks.reconcile_knowledge_graph_projections,
        "apply_async",
        lambda **kwargs: pytest.fail("must not queue another global pass"),
    )
    first = tasks.reconcile_knowledge_graph_projections.run(page_size=100)
    duplicate = tasks.reconcile_knowledge_graph_projections.run(page_size=100)
    scoped = tasks.reconcile_knowledge_graph_projections.run(collection_id=13)
    redis.now = 301
    second = tasks.reconcile_knowledge_graph_projections.run()
    assert first["examined_count"] == 10
    assert duplicate["skipped"] is True
    assert scoped["examined_count"] == 1
    assert second["examined_count"] == 3
    assert enqueued == list(range(1, 11)) + [13, 11, 12, 13]
    assert publications == [10] * 6


def test_real_isolated_redis_admission_contract():
    """Opt-in integration boundary: only point this at a disposable Redis."""
    import os
    from uuid import uuid4

    from redis import Redis

    url = os.environ.get("REDIS_MAINTENANCE_TEST_URL")
    if not url:
        pytest.skip("isolated Redis URL not supplied")
    client = Redis.from_url(url, socket_timeout=1, socket_connect_timeout=1)
    scope = "contract-test:" + uuid4().hex
    first = tasks.maintenance.admit(client, scope=scope, interval=300)
    try:
        assert first is not None
        assert 30 < client.ttl(first.lock_key) <= 300
        assert first.save_cursor(14)
        assert tasks.maintenance.admit(client, scope=scope, interval=300) is None
        # Expire only this uniquely namespaced test key, never flush the broker.
        client.pexpire(first.lock_key, 0)
        second = tasks.maintenance.admit(client, scope=scope, interval=300)
        assert second.cursor == 14
        assert not first.save_cursor(99)
        assert second.save_cursor(15)
        assert int(client.get(second.cursor_key)) == 15
    finally:
        if first is not None:
            client.delete(first.lock_key, first.cursor_key)
