"""Exercise scheduled publication against a disposable, explicitly opted-in Redis."""

import importlib
import json
import multiprocessing
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC
from uuid import uuid4

import pytest
from celery import Celery
from celery.beat import ScheduleEntry, Scheduler
from kombu.exceptions import OperationalError
from redis import Redis
from redis.exceptions import TimeoutError as RedisTimeout

from apps.knowledge_graph.projection import tasks
from aquillm.celery_schedules import knowledge_graph_maintenance_schedule

SCHEDULED = (
    "apps.knowledge_graph.projection.tasks."
    "scheduled_reconcile_knowledge_graph_projections"
)


@pytest.fixture
def broker(monkeypatch):
    url = os.getenv("REDIS_MAINTENANCE_TEST_URL")
    if not url:
        pytest.skip("requires isolated REDIS_MAINTENANCE_TEST_URL")
    monkeypatch.setenv("CELERY_BROKER_URL", url)
    client = Redis.from_url(url, socket_timeout=1, socket_connect_timeout=1)
    client.ping()
    prefix = "scheduled-test-" + uuid4().hex + ":"
    app = Celery("scheduled-test-" + uuid4().hex, broker=url, set_as_current=False)
    app.conf.update(
        broker_transport_options={"global_keyprefix": prefix},
        task_serializer="json",
        task_default_priority=0,
        task_ignore_result=True,
        task_publish_retry=False,
    )
    monkeypatch.setattr(tasks.maintenance, "current_app", app)
    monkeypatch.setattr(tasks.django_settings, "KG_MAINTENANCE_INTERVAL_SECONDS", 300)
    task = app.tasks.get(SCHEDULED)
    # This assertion also gives a useful pre-implementation regression failure.
    assert task is not None, "Beat needs a registered scheduled-only Task"
    from apps.knowledge_graph.projection import publication

    monkeypatch.setattr(publication, "current_app", app)
    queue = task.queue
    with app.connection_for_write() as connection:
        channel = connection.channel()
        keys = [prefix + channel._q_for_pri(queue, p) for p in channel.priority_steps]
    owner = prefix + "aquillm:projection-maintenance-publication:v1:global"
    try:
        yield app, task, client, keys, owner
    finally:
        # Delete only keys from this test's unique namespace; never FLUSHDB.
        owned = list(client.scan_iter(match=prefix + "*"))
        if owned:
            client.delete(*owned)
        app.close()


def depth(client, keys):
    return sum(client.llen(key) for key in keys)


def test_current_beat_ticks_do_not_grow_stopped_worker_backlog(monkeypatch):
    url = os.getenv("REDIS_MAINTENANCE_TEST_URL")
    if not url:
        pytest.skip("requires isolated REDIS_MAINTENANCE_TEST_URL")
    monkeypatch.setenv("CELERY_BROKER_URL", url)
    client = Redis.from_url(url)
    prefix = "stopped-regression-" + uuid4().hex + ":"
    app = Celery("stopped-regression", broker=url, set_as_current=False)
    app.conf.update(
        broker_transport_options={"global_keyprefix": prefix}, task_publish_retry=False
    )
    task = app.tasks[tasks.reconcile_knowledge_graph_projections.name]
    schedule = knowledge_graph_maintenance_schedule(
        enabled=True,
        extraction_queue="extract",
        projection_queue=task.queue,
        interval_seconds=300,
    )["knowledge-graph-projection-reconcile"]
    entry = ScheduleEntry(name="reconcile", app=app, **schedule)
    scheduler = Scheduler(app=app, lazy=True)
    try:
        for _ in range(4):
            scheduler.apply_async(entry, advance=False)
        with app.connection_for_write() as connection:
            channel = connection.channel()
            keys = [
                prefix + channel._q_for_pri(schedule["options"]["queue"], p)
                for p in channel.priority_steps
            ]
        assert depth(client, keys) == 1
    finally:
        owned = list(client.scan_iter(match=prefix + "*"))
        if owned:
            client.delete(*owned)
        app.close()


def test_real_beat_schedule_uses_scheduled_lane():
    entry = knowledge_graph_maintenance_schedule(
        enabled=True,
        extraction_queue="extract",
        projection_queue="project",
        interval_seconds=300,
    )["knowledge-graph-projection-reconcile"]
    assert entry["task"] == SCHEDULED
    assert entry["options"] == {
        "queue": "project-maintenance",
        "priority": 9,
        "expires": 300,
    }


def test_stopped_worker_coalesces_past_repeated_owner_and_envelope_expiry(
    broker, monkeypatch
):
    from datetime import datetime

    app, task, client, keys, owner = broker
    monkeypatch.setattr(tasks.django_settings, "KG_MAINTENANCE_INTERVAL_SECONDS", 1)
    first = task.apply_async()
    raw = client.lindex(keys[-1], 0)
    envelope = json.loads(raw)
    assert envelope["headers"]["id"] == first.id
    assert client.get(owner) == ("queued:" + first.id).encode()
    assert envelope["properties"]["priority"] == 9
    assert envelope["headers"]["expires"]
    assert client.ttl(owner) in (149, 150)
    time.sleep(1.05)
    assert datetime.fromisoformat(envelope["headers"]["expires"]) < datetime.now(UTC)
    for _ in range(3):
        client.pexpire(owner, 1)
        time.sleep(0.02)
        for _ in range(4):
            task.apply_async()
        assert depth(client, keys) == 1
        assert client.lindex(keys[-1], 0) == raw
    task.apply_async()
    assert depth(client, keys) == 1
    assert all(
        key.startswith(app.conf.broker_transport_options["global_keyprefix"])
        for key in keys
    )
    assert client.exists(task.queue) == 0
    assert client.exists("aquillm:projection-maintenance-publication:v1:global") == 0


@pytest.mark.parametrize("occupied_without_owner", [False, True])
def test_concurrent_publishers_have_one_slot(broker, occupied_without_owner):
    app, task, client, keys, owner = broker
    if occupied_without_owner:
        task.apply_async()
        client.pexpire(owner, 1)
        time.sleep(0.02)
    barrier = threading.Barrier(8)

    def publish(_):
        barrier.wait()
        # Independent producer connections, not shared thread-local channels.
        with app.connection_for_write() as connection:
            from kombu import Producer

            return task.apply_async(producer=Producer(connection)).id

    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(publish, range(8)))
    assert depth(client, keys) == 1
    envelope = json.loads(client.lindex(keys[-1], 0))
    if not occupied_without_owner:
        assert envelope["headers"]["id"] in ids
        assert client.get(owner) == ("queued:" + envelope["headers"]["id"]).encode()
    else:
        assert client.get(owner) is None


def test_publication_reply_loss_never_retries_or_deletes_owner(broker, monkeypatch):
    _, task, client, keys, owner = broker
    from apps.knowledge_graph.projection import publication

    calls = []

    class LostReply:
        def eval(self, *args):
            calls.append(args)
            client.eval(*args)
            raise RedisTimeout("reply lost after committed EVAL")

    monkeypatch.setattr(publication, "broker_client", lambda app: LostReply())
    with pytest.raises(OperationalError, match="reply lost"):
        task.apply_async()
    assert len(calls) == 1
    assert depth(client, keys) == 1
    assert client.get(owner).startswith(b"queued:")
    monkeypatch.setattr(publication, "broker_client", lambda app: client)
    task.apply_async()
    assert depth(client, keys) == 1


def test_connection_failure_recovers_on_later_tick(broker, monkeypatch):
    _, task, client, keys, owner = broker
    from apps.knowledge_graph.projection import publication

    def refused(app):
        raise ConnectionError("isolated refused coordinator")

    monkeypatch.setattr(publication, "broker_client", refused)
    with pytest.raises(OperationalError, match="refused coordinator"):
        task.apply_async()
    assert depth(client, keys) == 0
    assert client.get(owner) is None
    monkeypatch.setattr(publication, "broker_client", lambda app: client)
    task.apply_async()
    assert depth(client, keys) == 1


@pytest.mark.parametrize("bad_key", ["owner", "queue"])
def test_wrong_key_types_fail_before_any_write(broker, bad_key):
    _, task, client, keys, owner = broker
    if bad_key == "owner":
        client.lpush(owner, "corrupt")
    else:
        client.set(keys[0], "corrupt")
    with pytest.raises(OperationalError, match="maintenance_key_type|WRONGTYPE"):
        task.apply_async()
    assert client.llen(keys[-1]) == 0
    if bad_key == "queue":
        assert client.get(owner) is None


@pytest.mark.parametrize(
    "options",
    [
        {"queue": "celery"},
        {"priority": 0},
        {"task_id": "chosen"},
        {"countdown": 1},
        {"exchange": "other"},
        {"serializer": "pickle"},
        {"kwargs": {"collection_id": 1}},
    ],
)
def test_scheduled_route_cannot_be_bypassed(broker, options):
    _, task, client, keys, owner = broker
    with pytest.raises(ValueError):
        task.apply_async(**options)
    assert depth(client, keys) == 0
    assert client.get(owner) is None


def test_token_start_completion_and_stale_owner_fencing(broker, monkeypatch):
    _, task, client, keys, owner = broker
    from apps.knowledge_graph.projection import publication

    first = task.apply_async()
    client.rpop(keys[-1])
    assert publication.start(client, first.id)
    assert client.ttl(owner) in (149, 150)
    assert not publication.start(client, first.id)
    assert not publication.start(client, str(uuid4()))
    client.pexpire(owner, 1)
    time.sleep(0.02)
    successor = task.apply_async()
    client.rpop(keys[-1])
    assert publication.start(client, successor.id)
    assert not publication.finish(client, first.id)
    assert client.get(owner) == ("running:" + successor.id).encode()
    assert publication.finish(client, successor.id)
    assert not publication.start(client, first.id)
    invoked = []
    monkeypatch.setattr(
        tasks.reconcile_knowledge_graph_projections, "run", lambda: invoked.append(True)
    )
    next_id = task.apply_async().id
    client.rpop(keys[-1])
    task.push_request(id=next_id)
    try:
        task.run(publication_token=next_id)
        task.run(publication_token=next_id)
    finally:
        task.pop_request()
    assert invoked == [True]
    assert client.get(owner) is None


def test_scheduled_policy_preserves_normal_delivery_and_priority(broker):
    _, task, _, _, _ = broker
    assert (
        task.acks_late,
        task.reject_on_worker_lost,
        task.ignore_result,
        task.max_retries,
    ) == (False, False, True, 0)
    assert (task.soft_time_limit, task.time_limit) == (100, 120)
    for normal in (
        tasks.project_knowledge_graph_projection,
        tasks.prune_knowledge_graph_projection,
    ):
        assert normal.queue != task.queue
        assert normal.priority == 0
        assert normal.acks_late is True
    assert tasks.project_knowledge_graph_projection.reject_on_worker_lost is True
    assert tasks.reconcile_knowledge_graph_projections.queue != task.queue


def test_actual_beat_calls_custom_publisher(broker):
    app, task, client, keys, _ = broker
    scheduler = Scheduler(app=app, lazy=True)
    entry = ScheduleEntry(
        name="reconcile",
        task=task.name,
        schedule=300,
        app=app,
        options={"queue": task.queue, "priority": 9, "expires": 300},
    )
    for _ in range(3):
        scheduler.apply_async(entry, advance=False)
    assert depth(client, keys) == 1


def test_beat_startup_rejects_missing_registration_before_fallback(broker, monkeypatch):
    from celery.beat import Service
    from celery.signals import beat_init

    importlib.import_module("aquillm.celery")  # registers the Beat guard
    app, task, client, keys, _ = broker
    app.conf.beat_schedule = {
        "knowledge-graph-projection-reconcile": {"task": task.name, "schedule": 300},
    }
    monkeypatch.delitem(app.tasks, task.name)
    with pytest.raises(SystemExit, match="registration missing"):
        beat_init.send(sender=Service(app=app))
    assert depth(client, keys) == 0


def test_coordinator_socket_policy_cannot_be_overridden_by_broker_url(
    broker, monkeypatch
):
    app, _, _, _, _ = broker
    from apps.knowledge_graph.projection import publication

    monkeypatch.setenv(
        "CELERY_BROKER_URL",
        app.conf.broker_url
        + "?retry_on_timeout=true&socket_timeout=30&socket_connect_timeout=30",
    )
    client = publication.broker_client(app)
    try:
        connection = client.connection_pool.make_connection()
        assert connection.socket_timeout == 1
        assert connection.socket_connect_timeout == 1
        assert connection.retry._retries == 0
        assert connection.retry_on_timeout is False
    finally:
        client.close()


def _process_publish(app, task_name, barrier):
    from kombu import Producer

    barrier.wait()
    with app.connection_for_write() as connection:
        app.tasks[task_name].apply_async(producer=Producer(connection))


def test_independent_process_publishers_share_atomic_slot(broker):
    if os.name == "nt":
        pytest.skip("Linux process concurrency integration")
    app, task, client, keys, owner = broker
    context = multiprocessing.get_context("fork")
    barrier = context.Barrier(6)
    processes = [
        context.Process(target=_process_publish, args=(app, task.name, barrier))
        for _ in range(6)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(10)
            assert process.exitcode == 0
        assert depth(client, keys) == 1
        envelope = json.loads(client.lindex(keys[-1], 0))
        assert client.get(owner) == ("queued:" + envelope["headers"]["id"]).encode()
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join(5)


def test_unsupported_transport_fails_closed(broker, monkeypatch):
    app, task, client, keys, owner = broker
    monkeypatch.setenv("CELERY_BROKER_URL", "memory://")
    with pytest.raises(ValueError, match="unsupported_maintenance_transport"):
        task.apply_async()
    assert depth(client, keys) == 0
    assert client.get(owner) is None


@pytest.mark.parametrize(
    "broker_option", ["CELERY_BROKER_WRITE_URL", "CELERY_BROKER_READ_URL"]
)
def test_separate_broker_cannot_split_queue_and_coordinator(
    broker, monkeypatch, broker_option
):
    app, task, client, keys, owner = broker
    monkeypatch.setenv(broker_option, app.conf.broker_url.rsplit("/", 1)[0] + "/3")
    with pytest.raises(ValueError, match="maintenance_broker_mismatch"):
        task.apply_async()
    assert depth(client, keys) == 0
    assert client.get(owner) is None
