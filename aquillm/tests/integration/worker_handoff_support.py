"""Own subprocesses and Redis namespace without changing production settings."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

from django.db import connection


def wait_until(check, *, process=None, log=None, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return
        if process is not None and process.poll() is not None:
            raise AssertionError(log.read_text() if log.exists() else "worker exited")
        time.sleep(0.1)
    raise AssertionError("handoff timed out: " + (log.read_text() if log else ""))


class WorkerHarness:
    def __init__(self, tmp_path):
        from aquillm.celery import app
        from redis import Redis
        assert os.environ.get("RAG_REMEDIATION_TEST") == "1"
        assert connection.vendor == "postgresql"
        app.loader.import_default_modules()
        self.app = app
        self.prefix = "handoff-" + uuid4().hex + ":"
        self.queue = "handoff-" + uuid4().hex
        self.redis = Redis.from_url(app.conf.broker_url)
        self.directory = tmp_path
        self.receipt = tmp_path / "discovery.json"
        self.ready = tmp_path / "ready"
        self.log = tmp_path / "worker.log"
        self.deletions = tmp_path / "deletions.jsonl"
        self.environment = {
            **os.environ, "HANDOFF_DB_NAME": connection.settings_dict["NAME"],
            "APPLICATION_MAINTENANCE_SCHEDULER_ENABLED": "1",
            "HANDOFF_PREFIX": self.prefix, "HANDOFF_QUEUE": self.queue,
            "HANDOFF_RECEIPT": str(self.receipt), "HANDOFF_READY": str(self.ready),
            "HANDOFF_DELETIONS": str(self.deletions),
        }

    def command(self, mode):
        return [sys.executable, "-m", "tests.integration.worker_handoff_bootstrap", mode]

    def run(self, mode, *, enabled=True):
        result = subprocess.run(self.command(mode), env={**self.environment,
            "APPLICATION_MAINTENANCE_SCHEDULER_ENABLED": "1" if enabled else "0"},
            capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        return json.loads(self.receipt.read_text())

    @contextmanager
    def worker(self):
        with self.log.open("w") as output:
            process = subprocess.Popen(self.command("worker"), env=self.environment,
                                       stdout=output, stderr=subprocess.STDOUT, text=True)
            try:
                wait_until(self.ready.exists, process=process, log=self.log)
                yield process
            finally:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        # Persist synthetic-only runtime evidence in the owned artifacts volume.
        artifacts = Path("/artifacts/worker-handoff") / self.queue
        artifacts.mkdir(parents=True, exist_ok=True)
        for path in (self.log, self.receipt, self.deletions):
            if path.exists():
                (artifacts / path.name).write_bytes(path.read_bytes())
        print(f"worker-handoff runtime logs: {artifacts}")

    def queued(self, queue):
        return [json.loads(value) for value in self.redis.lrange(self.prefix + queue, 0, -1)]

    def deleted(self):
        if not self.deletions.exists():
            return []
        return [json.loads(line)["deleted"] for line in self.deletions.read_text().splitlines()]


@contextmanager
def isolated_broker(harness):
    app = harness.app
    original = dict(app.conf.broker_transport_options)
    original_routes = app.conf.task_routes
    reset_broker_pools(app)
    routes = {**original_routes,
              "documents.build_embedding_generation": {"queue": harness.queue}}
    # Django's namespaced config shadows a lone lowercase routing override.
    # Set its source key and the observer key so AMQP refreshes its route table.
    app.conf.update(broker_transport_options={"global_keyprefix": harness.prefix},
                    CELERY_TASK_ROUTES=routes, task_routes=routes)
    try:
        yield
    finally:
        reset_broker_pools(app)
        app.conf.update(broker_transport_options=original,
                        CELERY_TASK_ROUTES=original_routes, task_routes=original_routes)


def reset_broker_pools(app):
    # Celery.close deregisters the shared app. Close owned connections instead,
    # then let the next namespace receive fresh connection/producer pools.
    for pool in (app._pool, app.amqp._producer_pool):
        if pool is not None:
            pool.force_close_all()
    app._pool = app.amqp._producer_pool = None
