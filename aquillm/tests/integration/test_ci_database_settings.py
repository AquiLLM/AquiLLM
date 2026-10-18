"""The broad CI selection must set up its collected projection alias safely."""

import configparser
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[3]


def test_broad_ci_database_plan_uses_only_default_creation():
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/test-backend-frontend.yml").read_text()
    )
    job = workflow["jobs"]["backend-smoke"]
    step = next(
        item for item in job["steps"]
        if item.get("name") == "Chat, retrieval, document, and deployment regressions"
    )
    config = configparser.ConfigParser()
    config.read(ROOT / "pytest.ini")
    settings_module = config["pytest"]["DJANGO_SETTINGS_MODULE"]
    arguments = shlex.split(step["run"])
    for index, argument in enumerate(arguments):
        if argument.startswith("--ds="):
            settings_module = argument.partition("=")[2]
        elif argument == "--ds":
            settings_module = arguments[index + 1]

    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("POSTGRES_", "KG_", "DJANGO_"))
    }
    environment.update(job.get("env", {}))
    environment.update(step.get("env", {}))
    environment["DJANGO_SETTINGS_MODULE"] = settings_module
    environment["PYTHONPATH"] = str(ROOT / "aquillm")
    # Loading settings and planning aliases requires neither app-ready hooks nor SQL.
    code = """
import json
from django.db.backends.base.base import BaseDatabaseWrapper
from django.test.utils import get_unique_databases_and_mirrors

def refuse_connection(*args, **kwargs):
    raise AssertionError('database planning attempted a connection')

BaseDatabaseWrapper.ensure_connection = refuse_connection
databases, mirrors = get_unique_databases_and_mirrors(
    aliases={'default', 'projection_source'},
)
print(json.dumps({
    'created_aliases': sorted(alias for _, aliases in databases.values() for alias in aliases),
    'source_mirror': mirrors.get('projection_source'),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=environment,
        capture_output=True, text=True, timeout=20, check=True,
    )
    plan = json.loads(completed.stdout)
    assert plan == {"created_aliases": ["default"], "source_mirror": "default"}
