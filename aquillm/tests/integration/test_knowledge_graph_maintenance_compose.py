import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]


def _compose(name: str):
    return yaml.safe_load((ROOT / "deploy" / "compose" / name).read_text())


def test_base_and_development_define_one_gated_maintenance_scheduler():
    for name in ("base.yml", "development.yml"):
        services = _compose(name)["services"]
        schedulers = [
            service
            for service in services.values()
            if "celery -A aquillm beat" in service.get("command", "")
            and service.get("environment", {}).get("KG_MAINTENANCE_SCHEDULER_ENABLED") != "0"
        ]
        assert len(schedulers) == 1
        scheduler = schedulers[0]
        assert scheduler["profiles"] == ["knowledge-graph"]
        assert scheduler["depends_on"] == {"redis": {"condition": "service_healthy"}}
        environment = scheduler["environment"]
        assert environment["KG_MAINTENANCE_SCHEDULER_ENABLED"].endswith(":-0}")
        assert environment["KG_EXTRACTION_QUEUE"]
        assert environment["KG_PROJECTION_QUEUE"]
        assert environment["GOOGLE_OAUTH2_CLIENT_ID"] == "disabled"
        assert environment["GOOGLE_OAUTH2_CLIENT_SECRET"] == "disabled"
        assert environment["OPENAI_API_KEY"] == "disabled"
        assert environment["ANTHROPIC_API_KEY"] == "disabled"
        assert environment["GEMINI_API_KEY"] == "disabled"
        assert not any(
            key in environment
            for key in (
                "KG_PROJECTION_POSTGRES_SOURCE_DSN",
                "KG_PROJECTION_POSTGRES_STATE_DSN",
                "KG_MEMGRAPH_PROJECTION_USERNAME",
                "KG_MEMGRAPH_PROJECTION_PASSWORD",
            )
        )


@pytest.mark.parametrize("pruning_enabled", ["0", "1"])
def test_maintenance_scheduler_boots_from_its_allowlisted_environment(pruning_enabled):
    scheduler = _compose("development.yml")["services"][
        "scheduler_knowledge_graph_maintenance"
    ]
    declared = scheduler["environment"]
    environment = {
        key: os.environ[key]
        for key in ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP")
        if key in os.environ
    }
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(
                (str(ROOT / "aquillm"), *(path for path in sys.path if path))
            ),
            "SECRET_KEY": "scheduler-test-only",
            "DJANGO_DEBUG": "0",
            "POSTGRES_HOST": "127.0.0.1",
            "POSTGRES_PORT": "1",
            "KG_MAINTENANCE_SCHEDULER_ENABLED": "1",
            "KG_MAINTENANCE_INTERVAL_SECONDS": "300",
            "KG_ARTIFACT_PRUNING_ENABLED": pruning_enabled,
            "KG_ARTIFACT_PRUNING_INTERVAL_SECONDS": "172800",
            "KG_GRAPH_RECOVERY_PAGE_SIZE": "2",
            "KG_EXTRACTION_QUEUE": "test-extraction",
            "KG_PROJECTION_QUEUE": "test-projection",
            "KG_MEMGRAPH_PROJECTION_ENABLED": "0",
            "AWS_EC2_METADATA_DISABLED": "true",
        }
    )
    for key in (
        "GOOGLE_OAUTH2_CLIENT_ID",
        "GOOGLE_OAUTH2_CLIENT_SECRET",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
    ):
        environment[key] = declared[key]
    if pruning_enabled == "0":
        environment.pop("KG_ARTIFACT_PRUNING_ENABLED")
        environment.pop("KG_ARTIFACT_PRUNING_INTERVAL_SECONDS")

    redis_probe = ""
    if os.getenv("REDIS_MAINTENANCE_TEST_URL"):
        environment["CELERY_BROKER_URL"] = os.environ["REDIS_MAINTENANCE_TEST_URL"]
        redis_probe = """
from uuid import uuid4
from redis import Redis
from celery.beat import ScheduleEntry, Scheduler
prefix = 'allowlisted-beat-' + uuid4().hex + ':'
app.conf.broker_transport_options = {'global_keyprefix': prefix}
client = Redis.from_url(app.conf.broker_url)
entry = ScheduleEntry(name='reconcile', app=app,
    **app.conf.beat_schedule['knowledge-graph-projection-reconcile'])
try:
    scheduler = Scheduler(app=app, lazy=True)
    for _ in range(3):
        scheduler.apply_async(entry, advance=False)
    with app.connection_for_write() as connection:
        channel = connection.channel()
        keys = [prefix + channel._q_for_pri('test-projection-maintenance', p)
                for p in channel.priority_steps]
    assert sum(client.llen(key) for key in keys) == 1
finally:
    keys = list(client.scan_iter(match=prefix + '*'))
    if keys:
        client.delete(*keys)
    app.close()
"""

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from aquillm.celery import app; "
                "app.loader.import_default_modules(); "
                "from apps.knowledge_graph.projection.publication import "
                "SCHEDULED_TASK, ScheduledReconcileTask, validate_beat_registration; "
                "assert isinstance(app.tasks[SCHEDULED_TASK], ScheduledReconcileTask); "
                "validate_beat_registration(app); "
                "assert set(app.conf.beat_schedule) == "
                "{'knowledge-graph-build-recovery', "
                "'knowledge-graph-projection-reconcile'}"
                + (" | {'knowledge-graph-artifact-pruning'}; "
                   "assert app.conf.beat_schedule['knowledge-graph-artifact-pruning']"
                   "['schedule'] == 172800" if pruning_enabled == "1" else
                   "; from django.conf import settings; "
                   "assert settings.KG_ARTIFACT_PRUNING_ENABLED is False; "
                   "assert settings.KG_ARTIFACT_PRUNING_INTERVAL_SECONDS == 86400")
                + redis_probe
            ),
        ],
        cwd=ROOT / "aquillm",
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr[-2000:]


def test_development_pruning_is_explicitly_opt_in_and_application_beat_is_isolated():
    services = _compose("development.yml")["services"]
    environment = services["scheduler_knowledge_graph_maintenance"]["environment"]
    assert environment["KG_ARTIFACT_PRUNING_ENABLED"] == "${KG_ARTIFACT_PRUNING_ENABLED:-0}"
    assert environment["KG_ARTIFACT_PRUNING_INTERVAL_SECONDS"] == "${KG_ARTIFACT_PRUNING_INTERVAL_SECONDS:-86400}"
    application = services["scheduler_application_maintenance"]["environment"]
    assert application["KG_MAINTENANCE_SCHEDULER_ENABLED"] == "0"
    assert application["KG_ARTIFACT_PRUNING_ENABLED"] == "0"
    for name, service in services.items():
        if name == "web" or name.startswith("worker"):
            assert "celery -A aquillm beat" not in service.get("command", "")


def test_broker_restarts_after_host_or_container_runtime_restart():
    for name in ("base.yml", "development.yml", "production.yml"):
        redis = _compose(name)["services"]["redis"]
        assert redis["restart"] == "unless-stopped"


def test_extraction_worker_receives_the_maintenance_runtime_gate():
    for name in ("base.yml", "development.yml", "production.yml"):
        worker = _compose(name)["services"]["worker_knowledge_graph"]
        assert worker["environment"]["KG_MAINTENANCE_SCHEDULER_ENABLED"].endswith(
            ":-0}"
        )


@pytest.mark.parametrize(
    "name",
    ("base.yml", "development.yml", "test.yml", "production.yml", "no_gpu_dev.yml"),
)
def test_projection_worker_uses_same_configured_maintenance_interval(name):
    environment = _compose(name)["services"]["worker_knowledge_graph_projection"][
        "environment"
    ]
    assert environment["KG_MAINTENANCE_INTERVAL_SECONDS"] == (
        "${KG_MAINTENANCE_INTERVAL_SECONDS:-300}"
    )


@pytest.mark.parametrize(
    "name", ("base.yml", "development.yml", "test.yml", "production.yml", "no_gpu_dev.yml"),
)
def test_projection_worker_consumes_normal_and_scheduled_queues_with_hard_limit_pool(name):
    command = _compose(name)["services"]["worker_knowledge_graph_projection"]["command"]
    assert '--queues="$${KG_PROJECTION_QUEUE},$${KG_PROJECTION_QUEUE}-maintenance"' in command
    assert "--pool=prefork" in command
    assert "--concurrency=1" in command
    assert "--prefetch-multiplier=1" in command
