from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.integration.compose_render_test_support import (
    render_compose_with_reviewed_env,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
COMPOSE_FILES = tuple(
    REPOSITORY_ROOT / "deploy" / "compose" / name
    for name in (
        "base.yml",
        "development.yml",
        "production.yml",
        "no_gpu_dev.yml",
    )
)
MEMGRAPH_HEALTHCHECK_COMPOSE_FILES = COMPOSE_FILES + (
    REPOSITORY_ROOT / "deploy" / "compose" / "test.yml",
)
GRAPH_WORKER = "worker_knowledge_graph"
GRAPH_QUEUE = "knowledge-graph-extraction"
GRAPH_QUEUE_ENVIRONMENT = "${KG_EXTRACTION_QUEUE-knowledge-graph-extraction}"
GRAPH_DOCKERFILE = "deploy/docker/knowledge-graph/Dockerfile"
GRAPH_WORKER_RUNTIME_COMMAND = (
    '/opt/venv/bin/python -c "from lib.knowledge_graph.config import '
    'run_extraction_worker; run_extraction_worker()"'
)
CACHE_DIR_ENVIRONMENT = "${KG_GLINER2_CACHE_DIR:-/root/.cache/huggingface}"
GRAPH_FLAGS = {
    "DJANGO_DEBUG": "${DJANGO_DEBUG:-0}",
    "KG_BUILD_ENABLED": "${KG_BUILD_ENABLED:-0}",
    "KG_OVERLAY_ENABLED": "${KG_OVERLAY_ENABLED:-0}",
    "KG_EVAL_BYPASS_ALLOWED": "${KG_EVAL_BYPASS_ALLOWED:-0}",
}
PROJECT_ENVIRONMENT = "/opt/venv"
CONTAINER_PATH = (
    "/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)
HF_CACHE_VOLUME = (
    "${KG_HF_CACHE_HOST_PATH:-../../data/hf_knowledge_graph}:"
    "${KG_GLINER2_CACHE_DIR:-/root/.cache/huggingface}"
)
ARTIFACT_VOLUME = "../../artifacts:/app/artifacts"
PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def _compose(path: Path) -> dict[str, object]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _environment_map(raw: object) -> dict[str, object]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        values: dict[str, object] = {}
        for item in raw:
            key, separator, value = str(item).partition("=")
            assert separator, f"environment entry must assign a value: {item!r}"
            values[key] = value
        return values
    raise AssertionError("service environment must be a mapping or assignment list")


def _resolved_compose(
    override: Path,
    *,
    include_base: bool = True,
    environment_overrides: dict[str, str] | None = None,
) -> dict[str, object]:
    compose_files = (COMPOSE_FILES[0], override) if include_base else (override,)
    return render_compose_with_reviewed_env(
        compose_files,
        profile="knowledge-graph",
        environment_overrides=environment_overrides,
    )


@pytest.mark.parametrize(
    "compose_file", MEMGRAPH_HEALTHCHECK_COMPOSE_FILES, ids=lambda path: path.name
)
def test_memgraph_healthcheck_uses_supported_non_interactive_input(
    compose_file: Path,
) -> None:
    healthcheck = _compose(compose_file)["services"]["memgraph_knowledge_graph"][
        "healthcheck"
    ]["test"]
    command = healthcheck[1]

    assert healthcheck[0] == "CMD-SHELL"
    assert "echo 'RETURN 1;' | mgconsole" in command
    assert '--username "$$MEMGRAPH_USER"' in command
    assert '--password "$$MEMGRAPH_PASSWORD"' in command
    assert "--execute" not in command


@pytest.mark.parametrize(
    "compose_file", MEMGRAPH_HEALTHCHECK_COMPOSE_FILES, ids=lambda path: path.name
)
def test_memgraph_uses_a_canonical_internal_dns_alias(compose_file: Path) -> None:
    services = _compose(compose_file)["services"]
    memgraph_network = services["memgraph_knowledge_graph"]["networks"][
        "knowledge_graph_store"
    ]

    assert "memgraph-knowledge-graph" in memgraph_network["aliases"]
    for service_name in (
        "knowledge_graph_query_gateway",
        "worker_knowledge_graph_projection",
    ):
        environment = _environment_map(services[service_name]["environment"])
        assert environment["KG_MEMGRAPH_URI"] == "bolt://memgraph-knowledge-graph:7687"


@pytest.mark.parametrize("compose_file", COMPOSE_FILES, ids=lambda path: path.name)
def test_graph_worker_is_an_optional_isolated_cpu_worker(compose_file: Path) -> None:
    services = _compose(compose_file)["services"]
    worker = services[GRAPH_WORKER]

    assert worker["profiles"] == ["knowledge-graph"]
    assert worker["build"]["dockerfile"] == GRAPH_DOCKERFILE
    assert "nvidia" not in repr(worker).lower()
    assert "gpu" not in repr(worker).lower()


@pytest.mark.parametrize("compose_file", COMPOSE_FILES, ids=lambda path: path.name)
def test_graph_worker_launches_the_cpu_scaled_runtime(compose_file: Path) -> None:
    worker = _compose(compose_file)["services"][GRAPH_WORKER]
    environment = _environment_map(worker["environment"])

    assert worker["command"] == GRAPH_WORKER_RUNTIME_COMMAND
    assert environment["KG_EXTRACTION_WORKER_CONCURRENCY"] == (
        "${KG_EXTRACTION_WORKER_CONCURRENCY:-auto}"
    )
    assert environment["KG_GLINER2_CPU_THREADS"] == ("${KG_GLINER2_CPU_THREADS:-auto}")


@pytest.mark.parametrize("compose_file", COMPOSE_FILES, ids=lambda path: path.name)
def test_graph_worker_flags_and_dependencies_fail_closed(compose_file: Path) -> None:
    services = _compose(compose_file)["services"]
    worker = services[GRAPH_WORKER]
    environment = _environment_map(worker["environment"])

    for variable, default in GRAPH_FLAGS.items():
        assert environment[variable] == default
    assert environment["KG_EXTRACTION_QUEUE"] == GRAPH_QUEUE_ENVIRONMENT
    assert environment["C_FORCE_ROOT"] == 1
    assert environment["RUN_CELERY_IN_WEB"] == 0
    assert worker["depends_on"]["db"]["condition"] == "service_healthy"
    assert worker["depends_on"]["redis"]["condition"] == "service_healthy"
    assert "web" not in worker["depends_on"]
    assert GRAPH_WORKER not in services["web"].get("depends_on", {})


@pytest.mark.parametrize("compose_file", COMPOSE_FILES, ids=lambda path: path.name)
def test_graph_worker_has_a_persistent_configurable_hf_cache(
    compose_file: Path,
) -> None:
    worker = _compose(compose_file)["services"][GRAPH_WORKER]
    environment = _environment_map(worker["environment"])

    assert environment["HF_HOME"] == CACHE_DIR_ENVIRONMENT
    assert environment["KG_GLINER2_CACHE_DIR"] == environment["HF_HOME"]
    assert environment["KG_GLINER2_DEVICE"] == "cpu"
    assert HF_CACHE_VOLUME in worker["volumes"]


@pytest.mark.parametrize("compose_file", COMPOSE_FILES, ids=lambda path: path.name)
def test_graph_worker_uses_the_image_environment(
    compose_file: Path,
) -> None:
    worker = _compose(compose_file)["services"][GRAPH_WORKER]
    environment = _environment_map(worker["environment"])

    assert environment["UV_PROJECT_ENVIRONMENT"] == PROJECT_ENVIRONMENT
    assert environment["VIRTUAL_ENV"] == PROJECT_ENVIRONMENT
    assert environment["PATH"] == CONTAINER_PATH


@pytest.mark.parametrize(
    "compose_file",
    (COMPOSE_FILES[1], COMPOSE_FILES[3]),
    ids=lambda path: path.name,
)
def test_development_graph_worker_mounts_live_source_without_host_venv(
    compose_file: Path,
) -> None:
    worker = _compose(compose_file)["services"][GRAPH_WORKER]

    assert "../..:/app" in worker["volumes"]
    assert "/app/.venv" in worker["volumes"]


@pytest.mark.parametrize(
    "compose_file",
    (COMPOSE_FILES[0], COMPOSE_FILES[2]),
    ids=lambda path: path.name,
)
def test_immutable_graph_worker_mounts_only_artifacts_and_hf_cache(
    compose_file: Path,
) -> None:
    worker = _compose(compose_file)["services"][GRAPH_WORKER]

    assert set(worker["volumes"]) == {ARTIFACT_VOLUME, HF_CACHE_VOLUME}


@pytest.mark.parametrize(
    "compose_file",
    COMPOSE_FILES[1:3],
    ids=lambda path: path.name,
)
def test_graph_worker_can_reach_host_embedding_endpoints(compose_file: Path) -> None:
    worker = _compose(compose_file)["services"][GRAPH_WORKER]

    assert "host.docker.internal:host-gateway" in worker["extra_hosts"]


@pytest.mark.parametrize(
    "override",
    COMPOSE_FILES[1:3],
    ids=lambda path: f"base+{path.name}",
)
def test_graph_worker_survives_combined_compose_resolution(override: Path) -> None:
    worker = _resolved_compose(override)["services"][GRAPH_WORKER]
    environment = _environment_map(worker["environment"])

    assert environment["UV_PROJECT_ENVIRONMENT"] == PROJECT_ENVIRONMENT
    assert environment["VIRTUAL_ENV"] == PROJECT_ENVIRONMENT
    assert environment["PATH"] == CONTAINER_PATH
    assert "host.docker.internal=host-gateway" in worker["extra_hosts"]
    assert worker["depends_on"]["db"]["condition"] == "service_healthy"
    assert worker["depends_on"]["redis"]["condition"] == "service_healthy"


def test_base_development_overlay_retains_live_source_mount() -> None:
    worker = _resolved_compose(COMPOSE_FILES[1])["services"][GRAPH_WORKER]
    volumes = {volume["target"]: volume for volume in worker["volumes"]}

    assert set(("/app", "/app/.venv")).issubset(volumes)


def test_base_production_overlay_runs_image_source_and_persists_artifacts() -> None:
    worker = _resolved_compose(COMPOSE_FILES[2])["services"][GRAPH_WORKER]
    volumes = {volume["target"]: volume for volume in worker["volumes"]}

    assert "/app" not in volumes
    assert "/app/.venv" not in volumes
    assert "/app/artifacts" in volumes


@pytest.mark.parametrize("compose_file", COMPOSE_FILES, ids=lambda path: path.name)
def test_ordinary_worker_is_pinned_to_the_default_queue(compose_file: Path) -> None:
    worker = _compose(compose_file)["services"]["worker"]

    assert "--queues=celery" in worker["command"]


@pytest.mark.parametrize("compose_file", COMPOSE_FILES, ids=lambda path: path.name)
def test_every_django_producer_uses_the_same_resolved_graph_queue(
    compose_file: Path,
) -> None:
    services = _compose(compose_file)["services"]

    for service_name in ("web", "worker", "worker_memory_promotion"):
        if service_name in services:
            environment = _environment_map(services[service_name]["environment"])
            assert environment["KG_EXTRACTION_QUEUE"] == GRAPH_QUEUE_ENVIRONMENT


def test_development_projection_enablement_is_scoped_to_projection_worker() -> None:
    services = _compose(COMPOSE_FILES[1])["services"]

    for service_name in (
        "web",
        "worker",
        GRAPH_WORKER,
        "worker_memory_promotion",
    ):
        environment = _environment_map(services[service_name]["environment"])
        assert environment["KG_MEMGRAPH_PROJECTION_ENABLED"] == "0"

    projection_environment = _environment_map(
        services["worker_knowledge_graph_projection"]["environment"]
    )
    assert (
        projection_environment["KG_MEMGRAPH_PROJECTION_ENABLED"]
        == "${KG_MEMGRAPH_PROJECTION_WORKER_ENABLED:-0}"
    )


def test_production_projection_hook_and_worker_have_independent_flags() -> None:
    services = _compose(COMPOSE_FILES[2])["services"]
    web_environment = _environment_map(services["web"]["environment"])
    projection_environment = _environment_map(
        services["worker_knowledge_graph_projection"]["environment"]
    )

    assert web_environment["KG_MEMGRAPH_PROJECTION_ENABLED"] == "0"
    assert (
        web_environment["KG_MEMGRAPH_PROJECTION_HOOK_ENABLED"]
        == "${KG_MEMGRAPH_PROJECTION_HOOK_ENABLED:-0}"
    )
    assert (
        projection_environment["KG_MEMGRAPH_PROJECTION_ENABLED"]
        == "${KG_MEMGRAPH_PROJECTION_WORKER_ENABLED:-0}"
    )


@pytest.mark.parametrize("compose_file", COMPOSE_FILES, ids=lambda path: path.name)
def test_rendered_compose_preserves_one_custom_queue_and_cache_contract(
    compose_file: Path,
) -> None:
    queue_name = "isolated-graph.release_1"
    cache_dir = "/var/cache/aquillm-graph-test"
    services = _resolved_compose(
        compose_file,
        include_base=False,
        environment_overrides={
            "KG_EXTRACTION_QUEUE": queue_name,
            "KG_GLINER2_CACHE_DIR": cache_dir,
        },
    )["services"]

    for service_name in ("web", "worker", "worker_memory_promotion"):
        if service_name in services:
            environment = _environment_map(services[service_name]["environment"])
            assert environment["KG_EXTRACTION_QUEUE"] == queue_name
    graph_worker = services[GRAPH_WORKER]
    graph_environment = _environment_map(graph_worker["environment"])
    assert graph_environment["KG_EXTRACTION_QUEUE"] == queue_name
    assert graph_environment["KG_GLINER2_CACHE_DIR"] == cache_dir
    assert graph_environment["HF_HOME"] == cache_dir
    assert cache_dir in {
        volume["target"]
        for volume in graph_worker["volumes"]
        if isinstance(volume, dict)
    }


@pytest.mark.parametrize("compose_file", COMPOSE_FILES, ids=lambda path: path.name)
def test_rendered_compose_preserves_an_explicitly_empty_queue_for_validation(
    compose_file: Path,
) -> None:
    services = _resolved_compose(
        compose_file,
        include_base=False,
        environment_overrides={"KG_EXTRACTION_QUEUE": ""},
    )["services"]

    for service_name in (
        "web",
        "worker",
        "worker_memory_promotion",
        GRAPH_WORKER,
    ):
        if service_name in services:
            environment = _environment_map(services[service_name]["environment"])
            assert environment["KG_EXTRACTION_QUEUE"] == ""
