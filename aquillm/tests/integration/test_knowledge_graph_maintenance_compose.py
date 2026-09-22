import os
import subprocess
import sys
from pathlib import Path

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


def test_maintenance_scheduler_boots_from_its_allowlisted_environment():
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

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from aquillm.celery import app; "
                "app.loader.import_default_modules(); "
                "assert set(app.conf.beat_schedule) == "
                "{'knowledge-graph-build-recovery', "
                "'knowledge-graph-projection-reconcile'}"
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
