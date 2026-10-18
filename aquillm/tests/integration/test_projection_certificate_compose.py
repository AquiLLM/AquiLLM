"""Operator certificate and packed-write controls reach projection workers."""

from pathlib import Path

import pytest

from tests.integration.compose_render_test_support import (
    render_compose_with_reviewed_env,
)

ROOT = Path(__file__).resolve().parents[3]
FLAGS = (
    "KG_MEMBERSHIP_CERTIFICATE_ENABLED",
    "KG_MEMBERSHIP_CERTIFICATE_ISSUANCE_ENABLED",
    "KG_FRONTIER_PACKED_WRITES_ENABLED",
)


@pytest.mark.parametrize(
    "name",
    [
        "base.yml",
        "development.yml",
        "production.yml",
        "no_gpu_dev.yml",
        "test.yml",
    ],
)
@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({}, ("0", "0", "0")),
        ({"KG_MEMBERSHIP_CERTIFICATE_ENABLED": "1"}, ("1", "0", "0")),
        ({"KG_MEMBERSHIP_CERTIFICATE_ISSUANCE_ENABLED": "1"}, ("0", "1", "0")),
        ({"KG_FRONTIER_PACKED_WRITES_ENABLED": "1"}, ("0", "0", "1")),
    ],
)
def test_projection_worker_controls_are_independent_and_default_off(
    name, overrides, expected
):
    rendered = render_compose_with_reviewed_env(
        (ROOT / "deploy" / "compose" / name,),
        profile="knowledge-graph",
        environment_overrides=overrides,
    )
    environment = rendered["services"]["worker_knowledge_graph_projection"][
        "environment"
    ]
    assert tuple(environment.get(flag) for flag in FLAGS) == expected
