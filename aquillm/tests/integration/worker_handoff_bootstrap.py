"""Test-only normal Celery startup; discovery precedes all provider fixtures."""
import json
import os
from pathlib import Path
import sys

from aquillm.celery import app


def main():
    import django
    from django.conf import settings
    from django.db import connection, connections
    django.setup()
    database = os.environ["HANDOFF_DB_NAME"]
    assert database == "test_rag_test", "worker must use the isolated pytest database"
    settings.DATABASES["default"]["NAME"] = database
    connections.close_all()
    app.loader.import_default_modules()
    required = (
        "documents.build_embedding_generation",
        "aquillm.tasks.recover_profile_memory_reconciliation_jobs",
        "aquillm.tasks.reconcile_profile_memory_task",
    )
    missing = [name for name in required if name not in app.tasks]
    assert not missing, f"normal autodiscovery missing tasks: {missing}"
    assert app.conf.task_always_eager is False
    with connection.cursor() as cursor:
        cursor.execute("SELECT current_database()")
        assert cursor.fetchone() == ("test_rag_test",)
    app.conf.broker_transport_options = {"global_keyprefix": os.environ["HANDOFF_PREFIX"]}
    receipt = Path(os.environ["HANDOFF_RECEIPT"])
    receipt.write_text(json.dumps({"discovered": required, "schedule": app.conf.beat_schedule,
                                  "database": database, "eager": False}))
    if sys.argv[1] == "discover":
        return
    if sys.argv[1] == "schedule":
        publish_due_schedule(app, receipt)
        return
    # No fixture above imports the embedding task. Registry proof is already saved.
    from tests.integration.worker_handoff_provider import install_provider_boundaries
    from celery.signals import worker_ready
    install_provider_boundaries()

    @worker_ready.connect(weak=False)
    def ready(**kwargs):
        Path(os.environ["HANDOFF_READY"]).write_text("ready")

    app.worker_main([
        "worker", "--pool=solo", "--concurrency=1", "--prefetch-multiplier=1",
        "--queues=celery,memory-promotion," + os.environ["HANDOFF_QUEUE"],
        "--hostname=handoff@" + os.environ["HANDOFF_QUEUE"],
        "--without-gossip", "--without-mingle", "--without-heartbeat", "--loglevel=INFO",
    ])


def publish_due_schedule(app, receipt):
    from datetime import timedelta
    from celery.beat import Scheduler
    name = "profile-memory-reconciliation-recovery"
    scheduler = Scheduler(app=app)
    assert name in scheduler.schedule, "configured scheduler missing profile recovery"
    entry = scheduler.schedule[name]
    entry.last_run_at = app.now() - timedelta(seconds=61)
    assert entry.is_due().is_due
    scheduler.tick()
    assert scheduler.schedule[name].total_run_count == 1
    scheduler.tick()
    assert scheduler.schedule[name].total_run_count == 1
    data = json.loads(receipt.read_text())
    data["published"] = 1
    receipt.write_text(json.dumps(data))


if __name__ == "__main__":
    main()
