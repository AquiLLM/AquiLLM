from __future__ import annotations

import json
import os
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml

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
GRAPH_WORKER = "worker_knowledge_graph"
GRAPH_QUEUE = "knowledge-graph-extraction"
GRAPH_DOCKERFILE = "deploy/docker/knowledge-graph/Dockerfile"
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
    "${KG_HF_HOME:-/root/.cache/huggingface}"
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


def _resolved_compose(override: Path) -> dict[str, object]:
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker Compose is unavailable")
    compose_version = subprocess.run(
        [docker, "compose", "version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if compose_version.returncode != 0:
        pytest.skip("Docker Compose is unavailable")
    environment = os.environ.copy()
    environment.update(
        {
            "POSTGRES_NAME": "compose-test",
            "POSTGRES_USER": "compose-test",
            "POSTGRES_PASSWORD": "compose-test",
            "STORAGE_ACCESS_KEY": "compose-test",
            "STORAGE_SECRET_KEY": "compose-test",
        }
    )
    result = subprocess.run(
        [
            docker,
            "compose",
            "-f",
            str(COMPOSE_FILES[0]),
            "-f",
            str(override),
            "--profile",
            "knowledge-graph",
            "config",
            "--format",
            "json",
            "--no-env-resolution",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("compose_file", COMPOSE_FILES, ids=lambda path: path.name)
def test_graph_worker_is_an_optional_isolated_cpu_worker(compose_file: Path) -> None:
    services = _compose(compose_file)["services"]
    worker = services[GRAPH_WORKER]
    command = worker["command"]

    assert worker["profiles"] == ["knowledge-graph"]
    assert worker["build"]["dockerfile"] == GRAPH_DOCKERFILE
    assert f"--queues={GRAPH_QUEUE} --concurrency=1 --prefetch-multiplier=1" in command
    assert command.startswith("/opt/venv/bin/celery ")
    assert "nvidia" not in repr(worker).lower()
    assert "gpu" not in repr(worker).lower()


@pytest.mark.parametrize("compose_file", COMPOSE_FILES, ids=lambda path: path.name)
def test_graph_worker_flags_and_dependencies_fail_closed(compose_file: Path) -> None:
    services = _compose(compose_file)["services"]
    worker = services[GRAPH_WORKER]
    environment = _environment_map(worker["environment"])

    for variable, default in GRAPH_FLAGS.items():
        assert environment[variable] == default
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

    assert environment["HF_HOME"] == "${KG_HF_HOME:-/root/.cache/huggingface}"
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


def test_only_graph_worker_image_installs_the_optional_ml_extra() -> None:
    graph_dockerfile = REPOSITORY_ROOT / GRAPH_DOCKERFILE
    graph_contents = graph_dockerfile.read_text(encoding="utf-8")

    assert "uv sync --frozen --no-dev --extra knowledge-graph-local" in graph_contents
    assert "ENV UV_PROJECT_ENVIRONMENT=/opt/venv" in graph_contents
    assert "ENV VIRTUAL_ENV=/opt/venv" in graph_contents
    assert "ENV PATH=/opt/venv/bin:$PATH" in graph_contents
    assert "WORKDIR /app/aquillm" in graph_contents
    assert "torch.version.cuda is None" in graph_contents
    assert graph_contents.index("torch.version.cuda is None") > graph_contents.index(
        "uv sync --frozen --no-dev --extra knowledge-graph-local"
    )

    for relative_path in (
        "deploy/docker/web/Dockerfile",
        "deploy/docker/web/Dockerfile.prod",
    ):
        contents = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
        assert "--extra knowledge-graph-local" not in contents


def test_graph_worker_docker_context_excludes_environment_secret_files() -> None:
    dockerignore = (REPOSITORY_ROOT / ".dockerignore").read_text(encoding="utf-8")
    patterns = {
        line.strip()
        for line in dockerignore.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert ".env" in patterns
    assert ".env.*" in patterns
    assert all(not pattern.startswith("!.env") for pattern in patterns)


def test_graph_extra_pins_torch_to_the_linux_cpu_index() -> None:
    project = tomllib.loads(
        (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    graph_dependencies = project["project"]["optional-dependencies"][
        "knowledge-graph-local"
    ]
    assert "torch==2.11.0" in graph_dependencies
    assert project["tool"]["uv"]["sources"]["torch"] == {
        "index": "pytorch-cpu",
        "marker": "sys_platform == 'linux'",
    }
    cpu_indexes = [
        index
        for index in project["tool"]["uv"]["index"]
        if index["name"] == "pytorch-cpu"
    ]
    assert cpu_indexes == [
        {
            "name": "pytorch-cpu",
            "url": PYTORCH_CPU_INDEX,
            "explicit": True,
        }
    ]


def test_lock_selects_cpu_torch_only_on_linux_without_accelerator_packages() -> None:
    lock = tomllib.loads((REPOSITORY_ROOT / "uv.lock").read_text(encoding="utf-8"))
    packages = lock["package"]
    torch_packages = [package for package in packages if package["name"] == "torch"]

    assert len(torch_packages) == 2
    cpu_torch = next(
        package
        for package in torch_packages
        if package["source"]["registry"] == PYTORCH_CPU_INDEX
    )
    pypi_torch = next(
        package
        for package in torch_packages
        if package["source"]["registry"] == "https://pypi.org/simple"
    )
    assert cpu_torch["version"] == "2.11.0+cpu"
    assert cpu_torch["resolution-markers"]
    assert all(
        "sys_platform == 'linux'" in marker
        for marker in cpu_torch["resolution-markers"]
    )
    assert all("pytorch.org/whl/cpu/" in wheel["url"] for wheel in cpu_torch["wheels"])
    assert pypi_torch["version"] == "2.11.0"
    assert pypi_torch["resolution-markers"]
    assert all(
        "sys_platform != 'linux'" in marker
        for marker in pypi_torch["resolution-markers"]
    )
    pypi_wheel_urls = [wheel["url"] for wheel in pypi_torch["wheels"]]
    assert any("macosx" in url for url in pypi_wheel_urls)
    assert any("win_" in url for url in pypi_wheel_urls)

    forbidden = sorted(
        package["name"]
        for package in packages
        if package["name"].startswith(("cuda-", "nvidia-"))
        or "triton" in package["name"]
    )
    assert forbidden == []
