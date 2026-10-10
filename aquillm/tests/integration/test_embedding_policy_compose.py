"""Development application processes explicitly prohibit cross-provider embeddings."""

from pathlib import Path

import pytest

from tests.integration.compose_render_test_support import (
    render_compose_with_reviewed_env,
)


@pytest.mark.parametrize("ambient_policy", ["legacy-cohere", "", "invalid"])
def test_development_app_environments_enforce_local_only(ambient_policy):
    root = Path(__file__).resolve().parents[3]
    config = render_compose_with_reviewed_env(
        (root / "deploy/compose/development.yml",),
        profile="knowledge-graph",
        environment_overrides={"APP_EMBED_FALLBACK_POLICY": ambient_policy},
    )
    for service in (
        "web",
        "worker",
        "worker_memory_promotion",
        "worker_knowledge_graph",
        "worker_knowledge_graph_schema",
        "worker_knowledge_graph_projection",
        "scheduler_application_maintenance",
        "scheduler_knowledge_graph_maintenance",
    ):
        assert (
            config["services"][service]["environment"]["APP_EMBED_FALLBACK_POLICY"]
            == "local-only"
        )
