"""Build duration and individual-call allowance reach the worker independently."""
from pathlib import Path

import pytest

from tests.integration.compose_render_test_support import render_compose_with_reviewed_env

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("name", [
    "base.yml", "development.yml", "production.yml", "no_gpu_dev.yml", "test.yml",
])
def test_projection_worker_receives_independent_job_and_call_limits(name):
    rendered = render_compose_with_reviewed_env(
        (ROOT / "deploy" / "compose" / name,),
        profile="knowledge-graph",
        environment_overrides={
            "KG_PROJECTION_JOB_TIMEOUT_MS": "180000",
            "KG_PROJECTION_TIMEOUT_MS": "750",
        },
    )
    environment = rendered["services"]["worker_knowledge_graph_projection"]["environment"]
    assert environment["KG_PROJECTION_JOB_TIMEOUT_MS"] == "180000"
    assert environment["KG_PROJECTION_TIMEOUT_MS"] == "750"
