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
API = "knowledge_graph_api"
STORE = "knowledge_graph_store"
CONTROL = "knowledge_graph_control"
GATEWAY = "knowledge_graph_query_gateway"
BOLT_KEYS = {
    "KG_MEMGRAPH_URI",
    "KG_MEMGRAPH_DATABASE",
    "KG_MEMGRAPH_QUERY_USERNAME",
    "KG_MEMGRAPH_QUERY_PASSWORD",
    "KG_MEMGRAPH_PROJECTION_USERNAME",
    "KG_MEMGRAPH_PROJECTION_PASSWORD",
}
PROJECTION_KEYS = {
    "KG_PROJECTION_POSTGRES_SOURCE_DSN",
    "KG_PROJECTION_POSTGRES_STATE_DSN",
}
PROVIDER_KEYS = {
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_OAUTH2_CLIENT_ID",
    "GOOGLE_OAUTH2_CLIENT_SECRET",
}


def _compose(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _environment(service: dict) -> dict[str, object]:
    raw = service.get("environment", {})
    if isinstance(raw, dict):
        return raw
    return dict(str(item).partition("=")[::2] for item in raw)


def _web(services: dict) -> dict:
    return services.get("web", services.get("web_test"))


@pytest.mark.parametrize("path", FILES, ids=lambda path: path.name)
def test_gateway_is_the_only_web_to_store_path(path: Path) -> None:
    compose = _compose(path)
    services = compose["services"]
    web = _web(services)
    gateway = services[GATEWAY]
    graph = services["memgraph_knowledge_graph"]
    assert set(gateway["networks"]) == {API, STORE}
    assert set(graph["networks"]) == {STORE}
    assert set(web["networks"]) == {"default", API}
    assert STORE not in web["networks"]
    assert not gateway.get("ports") and not graph.get("ports")
    assert GATEWAY not in web.get("depends_on", {})
    assert gateway["profiles"] == graph["profiles"] == ["knowledge-graph"]
    assert "gateway_service:app" in gateway["command"]
    assert "--no-access-log" in gateway["command"]
    assert "healthcheck" in gateway and "healthcheck" in graph
    assert all(
        compose["networks"][name]["internal"] is True for name in (API, STORE, CONTROL)
    )


@pytest.mark.parametrize("path", FILES, ids=lambda path: path.name)
def test_web_owns_gateway_not_bolt_authority(path: Path) -> None:
    environment = _environment(_web(_compose(path)["services"]))
    assert environment["KG_TOPOLOGY_GATEWAY_URL"] == f"http://{GATEWAY}:8092"
    assert environment["KG_TOPOLOGY_GATEWAY_BEARER_TOKEN"]
    assert environment["KG_PROJECTION_IDENTIFIER_HMAC_KEY"]
    for key in BOLT_KEYS | PROJECTION_KEYS:
        assert environment.get(key) in (None, "")


@pytest.mark.parametrize("path", FILES, ids=lambda path: path.name)
def test_gateway_has_only_query_client_authority(path: Path) -> None:
    gateway = _compose(path)["services"][GATEWAY]
    environment = _environment(gateway)
    assert not gateway.get("env_file")
    assert environment["KG_MEMGRAPH_URI"] == "bolt://memgraph_knowledge_graph:7687"
    assert environment["KG_MEMGRAPH_QUERY_USERNAME"]
    assert environment["KG_MEMGRAPH_QUERY_PASSWORD"]
    assert environment["KG_TOPOLOGY_GATEWAY_BEARER_TOKEN"]
    assert not (
        (PROJECTION_KEYS | PROVIDER_KEYS | {"KG_PROJECTION_IDENTIFIER_HMAC_KEY"})
        & environment.keys()
    )


@pytest.mark.parametrize("path", FILES, ids=lambda path: path.name)
def test_projection_worker_owns_control_and_store_authority(path: Path) -> None:
    services = _compose(path)["services"]
    worker = services["worker_knowledge_graph_projection"]
    environment = _environment(worker)
    assert set(worker["networks"]) == {CONTROL, STORE}
    assert environment["KG_MEMGRAPH_PROJECTION_USERNAME"]
    assert environment["KG_MEMGRAPH_PROJECTION_PASSWORD"]
    assert environment["KG_PROJECTION_POSTGRES_SOURCE_DSN"]
    assert environment["KG_PROJECTION_POSTGRES_STATE_DSN"]
    assert environment["KG_PROJECTION_IDENTIFIER_HMAC_KEY"]
    assert all(environment.get(key) in (None, "", "disabled") for key in PROVIDER_KEYS)
    for name in ("worker", "worker_knowledge_graph"):
        if name not in services:
            continue
        nonowner = _environment(services[name])
        for key in BOLT_KEYS | PROJECTION_KEYS | {"KG_PROJECTION_IDENTIFIER_HMAC_KEY"}:
            assert nonowner.get(key) in (None, "")


@pytest.mark.parametrize("path", FILES, ids=lambda path: path.name)
def test_api_and_control_network_membership_is_closed(path: Path) -> None:
    services = _compose(path)["services"]
    assert set(services["knowledge_graph_query_extractor"]["networks"]) == {API}
    redis = services.get("redis", services.get("redis_test"))
    database = services.get("db", services.get("db_test"))
    assert CONTROL in redis["networks"]
    assert CONTROL in database["networks"]
    graph = services["memgraph_knowledge_graph"]
    assert not any(key.startswith("KG_MEMGRAPH_QUERY_") for key in _environment(graph))
    assert any("knowledge_graph_memgraph_data" in str(row) for row in graph["volumes"])
    if "memgraph" in services:
        assert services["memgraph"] is not graph
        assert services["memgraph"].get("volumes") != graph.get("volumes")


def test_eval_override_neutralizes_gateway_ambient_environment() -> None:
    source = (COMPOSE / "knowledge-graph-eval.yml").read_text(encoding="utf-8")
    section = source.split(f"  {GATEWAY}:", 1)[1].split(
        "\n  knowledge_graph_query_extractor:", 1
    )[0]
    assert 'restart: "no"' in section
    assert "env_file: !override []" in section
    assert "MEMGRAPH_ENTERPRISE" not in source
    assert "MEMGRAPH_LICENSE" not in source
