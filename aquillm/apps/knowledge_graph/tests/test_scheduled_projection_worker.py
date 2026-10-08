"""Real Linux prefork consumption, priority and child termination regressions.

Only task *operations* are probes; actual registered task boundaries, Redis
envelopes, acknowledgements, worker pool and hard limits execute unchanged.
"""

import multiprocessing
import os
import signal
import time
from contextlib import contextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest
import test_scheduled_projection_publication as publication_tests
from billiard.exceptions import SoftTimeLimitExceeded
from redis import Redis

from apps.knowledge_graph.projection import tasks

broker = publication_tests.broker
depth = publication_tests.depth


def wait_for(predicate, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("isolated worker did not produce expected event")


def run_worker(app, url, prefix, queue, ready):
    from celery.signals import worker_ready

    app.set_current()
    app.set_default()
    app.conf.update(
        worker_prefetch_multiplier=1,
        worker_enable_remote_control=False,
        worker_send_task_events=False,
        broker_connection_retry_on_startup=False,
    )
    client = Redis.from_url(url)

    def audit():
        client.rpush(prefix + "events", "audit")
        client.set(prefix + "child", os.getpid())
        if client.get(prefix + "block"):
            # Prove soft-limit swallowing cannot hang the one-child worker.
            while True:
                try:
                    time.sleep(1)
                except SoftTimeLimitExceeded:
                    client.set(prefix + "soft_limit_seen", "yes")
        return {"probe": "completed"}

    tasks.reconcile_knowledge_graph_projections = SimpleNamespace(run=audit)

    def project(**kwargs):
        client.rpush(prefix + "events", "project")
        return SimpleNamespace(ready=True, failure_code=None)

    def prune(**kwargs):
        client.rpush(prefix + "events", "prune")
        return SimpleNamespace(candidate_count=0, deleted_count=0)

    tasks.project_generation = project
    tasks.prune_graph_projection_generations = prune
    worker_ready.connect(lambda **kwargs: ready.set(), weak=False)
    app.Worker(
        pool="prefork",
        concurrency=1,
        queues=[tasks._TASK_QUEUE, queue],
        loglevel="ERROR",
        without_gossip=True,
        without_mingle=True,
        without_heartbeat=True,
    ).start()


@contextmanager
def worker(broker):
    if os.name == "nt":
        pytest.skip("hard-limit contract requires Linux prefork")
    app, task, client, keys, owner = broker
    context = multiprocessing.get_context("fork")
    ready = context.Event()
    prefix = app.conf.broker_transport_options["global_keyprefix"]
    process = context.Process(
        target=run_worker, args=(app, app.conf.broker_url, prefix, task.queue, ready)
    )
    process.start()
    try:
        assert ready.wait(20), "isolated Celery worker did not become ready"
        yield prefix
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        if process.is_alive():
            process.kill()
            process.join(5)
        assert not process.is_alive()


def project(app):
    app.tasks[tasks.project_knowledge_graph_projection.name].apply_async(
        args=[str(uuid4())]
    )


def prune(app):
    app.tasks[tasks.prune_knowledge_graph_projection.name].apply_async()


def test_actual_envelope_consumed_completion_reopens_slot_and_normal_priority_wins(
    broker,
):
    app, scheduled, client, keys, owner = broker
    scheduled.apply_async()
    project(app)
    prune(app)
    with worker(broker) as prefix:
        events = prefix + "events"
        wait_for(lambda: client.llen(events) == 3)
        assert client.lrange(events, 0, -1) == [b"project", b"prune", b"audit"]
        wait_for(lambda: client.get(owner) is None)
        scheduled.apply_async()
        wait_for(lambda: client.llen(events) == 4)
        wait_for(lambda: client.get(owner) is None)
        assert depth(client, keys) == 0


def test_expired_queued_owner_skips_old_delivery_then_next_tick_executes(broker):
    _, scheduled, client, keys, owner = broker
    scheduled.apply_async()
    client.pexpire(owner, 1)
    time.sleep(0.02)
    with worker(broker) as prefix:
        wait_for(lambda: depth(client, keys) == 0)
        time.sleep(0.2)
        assert client.llen(prefix + "events") == 0
        scheduled.apply_async()
        wait_for(lambda: client.llen(prefix + "events") == 1)
        wait_for(lambda: client.get(owner) is None)


def test_hung_audit_is_hard_killed_at_real_120_seconds_and_normal_tasks_continue(
    broker,
):
    app, scheduled, client, keys, owner = broker
    prefix = app.conf.broker_transport_options["global_keyprefix"]
    client.set(prefix + "block", "yes")
    with worker(broker):
        scheduled.apply_async()
        wait_for(lambda: client.get(prefix + "child"))
        began = time.monotonic()
        assert client.get(owner).startswith(b"running:")
        project(app)
        prune(app)
        wait_for(lambda: client.llen(prefix + "events") == 3, timeout=135)
        elapsed = time.monotonic() - began
        assert 115 <= elapsed < 135
        assert client.get(prefix + "soft_limit_seen") == b"yes"
        assert client.lrange(prefix + "events", 0, -1) == [
            b"audit",
            b"project",
            b"prune",
        ]
        # Billiard first SIGTERMs on hard timeout; Python may unwind finally.
        # SIGKILL/process loss can instead leave the running lease to expire.
        assert client.ttl(owner) == -2 or 0 < client.ttl(owner) <= 30
        print(
            "Linux prefork hard-limit recovery: normal tasks resumed after "
            f"{elapsed:.3f}s"
        )
        assert depth(client, keys) == 0
        client.delete(prefix + "block")
        client.pexpire(owner, 1)
        time.sleep(0.02)
        scheduled.apply_async()
        wait_for(lambda: client.llen(prefix + "events") == 4)
        wait_for(lambda: client.get(owner) is None)


def test_killed_accepted_child_is_not_redelivered_and_owner_recovers(broker):
    app, scheduled, client, keys, owner = broker
    prefix = app.conf.broker_transport_options["global_keyprefix"]
    client.set(prefix + "block", "yes")
    with worker(broker):
        scheduled.apply_async()
        child = int(wait_for(lambda: client.get(prefix + "child")))
        assert client.get(owner).startswith(b"running:")
        os.kill(child, signal.SIGKILL)
        client.delete(prefix + "block")
        project(app)
        wait_for(lambda: client.llen(prefix + "events") == 2)
        assert client.lrange(prefix + "events", 0, -1) == [b"audit", b"project"]
        assert depth(client, keys) == 0
        assert client.ttl(owner) > 0
        scheduled.apply_async()
        assert depth(client, keys) == 0
        client.pexpire(owner, 1)
        time.sleep(0.02)
        scheduled.apply_async()
        wait_for(lambda: client.llen(prefix + "events") == 3)
        wait_for(lambda: client.get(owner) is None)


def test_kombu_reserved_restore_can_add_envelope_but_stale_token_cannot_run(broker):
    """Exercise the broker restore path which intentionally bypasses publisher."""
    from kombu import Consumer, Queue

    _, scheduled, client, keys, owner = broker
    app = scheduled.app
    first = scheduled.apply_async()
    with app.connection_for_read() as connection:
        channel = connection.channel()
        deliveries = []
        with Consumer(
            channel,
            queues=[Queue(scheduled.queue)],
            callbacks=[lambda body, message: deliveries.append(message)],
            accept=["json"],
        ):
            connection.drain_events(timeout=5)
        assert depth(client, keys) == 0
        # This reservation is before pool acceptance; no early ACK has happened.
        assert deliveries[0].headers["id"] == first.id
        assert (
            client.hlen(
                app.conf.broker_transport_options["global_keyprefix"] + "unacked"
            )
            == 1
        )
        client.pexpire(owner, 1)
        time.sleep(0.02)
        successor = scheduled.apply_async()
        channel.qos.restore_by_tag(deliveries[0].delivery_tag)
        assert depth(client, keys) == 2
        scheduled.apply_async()
        assert depth(client, keys) == 2
        assert client.get(owner) == ("queued:" + successor.id).encode()
    with worker(broker) as prefix:
        wait_for(lambda: depth(client, keys) == 0)
        wait_for(lambda: client.get(owner) is None)
        assert client.lrange(prefix + "events", 0, -1) == [b"audit"]
