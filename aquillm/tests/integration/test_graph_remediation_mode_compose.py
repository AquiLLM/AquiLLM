"""Modern document graphs use the same mode in readers and background writers."""

from pathlib import Path

import pytest

from tests.integration.compose_render_test_support import (
    render_compose_with_reviewed_env,
)

ROOT = Path(__file__).resolve().parents[3]
MODE = "KG_GRAPH_REMEDIATION_MODE"
SOURCE_SERVICES = (
    "web",
    "worker",
    "worker_knowledge_graph",
    "worker_knowledge_graph_schema",
    "worker_knowledge_graph_projection",
)


@pytest.mark.parametrize(
    "name", ("base.yml", "development.yml", "production.yml", "no_gpu_dev.yml")
)
@pytest.mark.parametrize(
    "overrides,expected",
    (
        ({}, "legacy"),
        ({MODE: ""}, "legacy"),
        ({MODE: "shadow"}, "shadow"),
        ({MODE: "v2"}, "v2"),
    ),
    ids=("unset", "empty", "shadow", "v2"),
)
def test_source_readers_and_writers_share_graph_remediation_mode(
    name, overrides, expected
):
    rendered = render_compose_with_reviewed_env(
        (ROOT / "deploy" / "compose" / name,),
        profile="knowledge-graph",
        environment_overrides=overrides,
    )
    actual = {
        name: rendered["services"][name]["environment"].get(MODE)
        for name in SOURCE_SERVICES
    }
    assert actual == dict.fromkeys(SOURCE_SERVICES, expected)
