from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
COMPOSE = ROOT / "deploy" / "compose"
FILES = tuple(
    COMPOSE / name
    for name in (
        "base.yml",
        "development.yml",
        "production.yml",
        "no_gpu_dev.yml",
        "test.yml",
    )
)
BUILD_HASH = "${KG_QUERY_EXTRACTOR_BUILD_HASH:-}"
SOURCE_DSN = "${KG_PROJECTION_POSTGRES_SOURCE_DSN:-}"
STATE_DSN = "${KG_PROJECTION_POSTGRES_STATE_DSN:-}"


def _compose(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _env(service: dict) -> dict:
    raw = service.get("environment", {})
    if isinstance(raw, dict):
        return raw
    return dict(str(item).partition("=")[::2] for item in raw)


@pytest.mark.parametrize("path", FILES, ids=lambda path: path.name)
def test_dedicated_hybrid_services_are_profiled_and_isolated(path: Path) -> None:
    services = _compose(path)["services"]
    graph = services["memgraph_knowledge_graph"]
    extractor = services["knowledge_graph_query_extractor"]
    worker = services["worker_knowledge_graph_projection"]
    assert graph["profiles"] == ["knowledge-graph"]
    assert extractor["profiles"] == ["knowledge-graph"]
    assert worker["profiles"] == ["knowledge-graph"]
    assert not graph.get("ports")
    assert "healthcheck" in graph and "healthcheck" in extractor
    assert "memgraph_knowledge_graph" in worker["depends_on"]
    assert "redis" in worker["depends_on"] or "redis_test" in worker["depends_on"]
    assert (
        "memgraph_knowledge_graph"
        not in services.get("web", services.get("web_test"))["depends_on"]
    )
    assert graph is not services.get("memgraph")


@pytest.mark.parametrize("path", FILES, ids=lambda path: path.name)
def test_extractor_has_no_database_credentials_and_exact_provenance(path: Path) -> None:
    services = _compose(path)["services"]
    extractor = services["knowledge_graph_query_extractor"]
    environment = _env(extractor)
    assert not extractor.get("env_file")
    assert environment["KG_QUERY_EXTRACTOR_BUILD_HASH"] == BUILD_HASH
    assert environment["KG_QUERY_EXTRACTOR_BEARER_TOKEN"]
    assert environment["KG_QUERY_EXTRACTOR_MODEL_REVISION"]
    assert environment["KG_QUERY_EXTRACTOR_ONTOLOGY_CHECKSUM"]
    assert not any(
        name.startswith("POSTGRES_") or name.endswith("_DSN") for name in environment
    )
    assert "query_extractor.service" in extractor["command"]


@pytest.mark.parametrize("path", FILES, ids=lambda path: path.name)
def test_web_gets_query_only_and_projection_worker_gets_split_credentials(
    path: Path,
) -> None:
    services = _compose(path)["services"]
    web = services.get("web", services.get("web_test"))
    extractor = services["knowledge_graph_query_extractor"]
    worker = services["worker_knowledge_graph_projection"]
    web_environment = _env(web)
    worker_environment = _env(worker)
    assert web_environment["KG_QUERY_EXTRACTOR_BUILD_HASH"] == BUILD_HASH
    assert (
        web_environment["KG_QUERY_EXTRACTOR_URL"]
        == "http://knowledge-graph-query-extractor:8000"
    )
    assert "knowledge-graph-query-extractor" in extractor["networks"][
        "knowledge_graph_api"
    ]["aliases"]
    assert web_environment["KG_TOPOLOGY_GATEWAY_URL"].endswith(":8092")
    assert web_environment["KG_TOPOLOGY_GATEWAY_BEARER_TOKEN"]
    assert web_environment.get("KG_MEMGRAPH_URI") in (None, "")
    assert web_environment.get("KG_MEMGRAPH_QUERY_USERNAME") in (None, "")
    assert web_environment.get("KG_MEMGRAPH_QUERY_PASSWORD") in (None, "")
    assert web_environment.get("KG_MEMGRAPH_PROJECTION_PASSWORD") in (None, "")
    assert web_environment.get("KG_PROJECTION_POSTGRES_SOURCE_DSN") in (None, "")
    assert web_environment.get("KG_PROJECTION_POSTGRES_STATE_DSN") in (None, "")
    assert not worker.get("env_file")
    assert worker_environment["KG_PROJECTION_POSTGRES_SOURCE_DSN"] == SOURCE_DSN
    assert worker_environment["KG_PROJECTION_POSTGRES_STATE_DSN"] == STATE_DSN
    assert worker_environment["KG_MEMGRAPH_PROJECTION_USERNAME"]
    assert worker_environment["KG_MEMGRAPH_PROJECTION_PASSWORD"]
    assert worker_environment["KG_PROJECTION_QUEUE"]
    assert not any(
        name in worker_environment
        for name in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_NAME")
    )
    assert '--queues="$${KG_PROJECTION_QUEUE}"' in worker["command"]


@pytest.mark.parametrize("path", FILES, ids=lambda path: path.name)
def test_dedicated_graph_has_private_network_and_persistent_volume(path: Path) -> None:
    compose = _compose(path)
    graph = compose["services"]["memgraph_knowledge_graph"]
    assert "knowledge_graph_store" in graph["networks"]
    assert compose["networks"]["knowledge_graph_store"]["internal"] is True
    assert any("knowledge_graph_memgraph_data" in str(row) for row in graph["volumes"])
    assert "knowledge_graph_memgraph_data" in compose["volumes"]


def test_eval_override_neutralizes_ambient_files_for_new_services() -> None:
    source = (COMPOSE / "knowledge-graph-eval.yml").read_text(encoding="utf-8")
    for service in (
        "memgraph_knowledge_graph",
        "knowledge_graph_query_gateway",
        "knowledge_graph_query_extractor",
        "worker_knowledge_graph_projection",
    ):
        start = source.index(f"  {service}:")
        end = source.find("\n  ", start + len(service) + 3)
        while end != -1 and source[end + 3 : end + 4] == " ":
            end = source.find("\n  ", end + 3)
        section = source[start:] if end == -1 else source[start:end]
        assert 'restart: "no"' in section
        assert "env_file: !override []" in section


@pytest.mark.container
@pytest.mark.skipif(
    True,
    reason="Task21 bounded container smoke runs only in the reviewed cloud gate",
)
def test_hybrid_services_health_in_isolated_compose() -> None:
    """Cloud gate placeholder; local Task21 execution must not start resources."""
