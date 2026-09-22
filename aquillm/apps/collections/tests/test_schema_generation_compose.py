from __future__ import annotations

from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[4]
COMPOSE_FILES = tuple(ROOT / "deploy" / "compose" / name for name in (
    "base.yml", "development.yml", "no_gpu_dev.yml", "production.yml", "test.yml",
))
DEFAULTS = {
    "KG_SCHEMA_GENERATION_ENABLED": "${KG_SCHEMA_GENERATION_ENABLED:-0}",
    "KG_SCHEMA_GENERATION_MAX_CHUNKS": "${KG_SCHEMA_GENERATION_MAX_CHUNKS:-32}",
    "KG_SCHEMA_GENERATION_MAX_CHARACTERS": "${KG_SCHEMA_GENERATION_MAX_CHARACTERS:-48000}",
    "KG_SCHEMA_GENERATION_TIMEOUT_SECONDS": "${KG_SCHEMA_GENERATION_TIMEOUT_SECONDS:-180}",
    "KG_GLINER2_LOCAL_FILES_ONLY": "${KG_GLINER2_LOCAL_FILES_ONLY:-1}",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "OMP_NUM_THREADS": "${KG_SCHEMA_CPU_THREADS:-1}",
    "MKL_NUM_THREADS": "${KG_SCHEMA_CPU_THREADS:-1}",
}


@pytest.mark.parametrize("path", COMPOSE_FILES, ids=lambda path: path.name)
def test_schema_generation_settings_reach_the_isolated_schema_worker(
    path: Path,
) -> None:
    services = yaml.safe_load(path.read_text(encoding="utf-8"))["services"]
    worker = services["worker_knowledge_graph_schema"]
    environment = worker["environment"]
    for name, expected in DEFAULTS.items():
        assert environment[name] == expected
    assert "--queues=knowledge-graph-schema" in worker["command"]
    assert "--concurrency=1" in worker["command"]
    assert "--prefetch-multiplier=1" in worker["command"]
    assert worker["profiles"] == ["knowledge-graph"]
    assert "knowledge-graph/Dockerfile" in worker["build"]["dockerfile"]
    assert not worker.get("env_file")
    assert not set(environment).intersection({
        "KG_MEMGRAPH_QUERY_PASSWORD", "KG_MEMGRAPH_PROJECTION_PASSWORD",
        "KG_PROJECTION_POSTGRES_SOURCE_DSN", "KG_PROJECTION_POSTGRES_STATE_DSN",
        "KG_PROJECTION_IDENTIFIER_HMAC_KEY", "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    })
    assert environment["OPENAI_API_KEY"] == "disabled"
    assert environment["ANTHROPIC_API_KEY"] == "disabled"
    assert environment["GEMINI_API_KEY"] == "disabled"
    assert environment["VLLM_BASE_URL"] == "http://vllm:8000/v1"
    assert environment["POSTGRES_HOST"] == (
        "db_test" if path.name == "test.yml" else "db"
    )
    if "worker_knowledge_graph" in services:
        assert (
            "knowledge-graph-schema"
            not in services["worker_knowledge_graph"]["command"]
        )


@pytest.mark.parametrize("path", COMPOSE_FILES, ids=lambda path: path.name)
def test_rendered_schema_worker_preserves_queue_and_credentials(path: Path) -> None:
    from tests.integration.compose_render_test_support import (
        render_compose_with_reviewed_env,
    )

    rendered = render_compose_with_reviewed_env((path,), profile="knowledge-graph")
    worker = rendered["services"]["worker_knowledge_graph_schema"]
    assert "--queues=knowledge-graph-schema" in worker["command"]
    assert not worker.get("env_file")
    assert worker["environment"]["OPENAI_API_KEY"] == "disabled"
    assert worker["environment"]["KG_MEMGRAPH_PROJECTION_ENABLED"] == "0"
    assert not set(worker.get("networks", {})).intersection({
        "knowledge_graph_store", "knowledge_graph_control",
    })
    expected_broker = "redis_test" if path.name == "test.yml" else "redis"
    assert f"--broker=redis://{expected_broker}:6379" in worker["command"]


def test_schema_worker_cpu_threads_can_be_tuned_without_changing_concurrency() -> None:
    from tests.integration.compose_render_test_support import (
        render_compose_with_reviewed_env,
    )

    rendered = render_compose_with_reviewed_env(
        (ROOT / "deploy/compose/development.yml",), profile="knowledge-graph",
        environment_overrides={"KG_SCHEMA_CPU_THREADS": "4"},
    )
    worker = rendered["services"]["worker_knowledge_graph_schema"]
    assert worker["environment"]["OMP_NUM_THREADS"] == "4"
    assert worker["environment"]["MKL_NUM_THREADS"] == "4"
    assert "--concurrency=1" in worker["command"]
