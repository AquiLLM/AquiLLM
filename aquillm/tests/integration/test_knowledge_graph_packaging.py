"""Knowledge graph image, packaging, and deployment documentation contracts."""
import shutil
import subprocess
import tomllib
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
GRAPH_DOCKERFILE = "deploy/docker/knowledge-graph/Dockerfile"
GRAPH_QUEUE = "knowledge-graph-extraction"


def test_only_graph_worker_image_installs_the_optional_ml_extra() -> None:
    graph_dockerfile = REPOSITORY_ROOT / GRAPH_DOCKERFILE
    graph_contents = graph_dockerfile.read_text(encoding="utf-8")

    assert "uv sync --frozen --no-dev --extra knowledge-graph-local" in graph_contents
    assert "ENV UV_PROJECT_ENVIRONMENT=/opt/venv" in graph_contents
    assert "ENV VIRTUAL_ENV=/opt/venv" in graph_contents
    assert "ENV PATH=/opt/venv/bin:$PATH" in graph_contents
    assert "WORKDIR /app/aquillm" in graph_contents
    assert f"ENV KG_EXTRACTION_QUEUE={GRAPH_QUEUE}" in graph_contents
    assert "run_extraction_worker; run_extraction_worker()" in graph_contents
    assert "--concurrency=1" not in graph_contents
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
    assert "/artifacts/" in patterns
    assert all(not pattern.startswith("!.env") for pattern in patterns)

    git = shutil.which("git")
    assert git is not None
    ignored_report = subprocess.run(
        [git, "check-ignore", "--verbose", "artifacts/kg-eval-comparison.json"],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert ignored_report.returncode == 0, ignored_report.stderr
    assert "/artifacts/" in ignored_report.stdout


def test_runbook_uses_compose_network_and_an_isolated_eval_worker() -> None:
    runbook = (
        REPOSITORY_ROOT
        / "docs"
        / "documents"
        / "operations"
        / "knowledge-graph-overlay-runbook.md"
    ).read_text(encoding="utf-8")
    normal_prefix = (
        "docker compose --env-file .env "
        "-f deploy/compose/development.yml exec web "
        "/opt/venv/bin/python manage.py"
    )

    host_python_commands = (
        "python aquillm/manage.py",
        "python manage.py",
    )
    assert not any(
        line.strip().startswith(host_python_commands) for line in runbook.splitlines()
    )
    assert f"{normal_prefix} rebuild_knowledge_graph" in runbook
    assert f"{normal_prefix} inspect_knowledge_graph" in runbook
    assert f"{normal_prefix} prune_knowledge_graph" in runbook
    assert 'KG_EVAL_RUN_ID="$(python -c' in runbook
    assert 'KG_EVAL_PROJECT="aquillm-kg-eval-$KG_EVAL_RUN_ID"' in runbook
    assert 'KG_EVAL_QUEUE="knowledge-graph-eval-$KG_EVAL_RUN_ID"' in runbook
    assert "env -i" in runbook
    assert "trap - EXIT INT TERM" in runbook
    assert "trap 'status=$?; cleanup_kg_eval; exit \"$status\"' EXIT" in runbook
    assert "trap 'cleanup_kg_eval; exit 130' INT" in runbook
    assert "trap 'cleanup_kg_eval; exit 143' TERM" in runbook
    assert "trap cleanup_kg_eval EXIT INT TERM" not in runbook
    assert 'docker rm -fv "$worker_container"' in runbook
    assert '"KG_EXTRACTION_QUEUE=$KG_EVAL_QUEUE"' in runbook
    assert "DJANGO_DEBUG=1 KG_EVAL_BYPASS_ALLOWED=1" in runbook
    assert "KG_BUILD_ENABLED=0 KG_OVERLAY_ENABLED=0" in runbook
    cleanup = runbook.split("stop_eval_worker() {", 1)[1].split("}\n", 1)[0]
    assert cleanup.index("docker inspect") < cleanup.index("docker rm -fv")
    assert '--user "$(id -u):$(id -g)"' in runbook
    assert "kg_eval_python manage.py rebuild_knowledge_graph" in runbook
    assert "kg_eval_python manage.py inspect_knowledge_graph" in runbook
    assert "kg_eval_python -m apps.knowledge_graph.evals.run_kg_eval" in runbook
    assert "/app/artifacts/kg-eval-comparison-$KG_EVAL_RUN_ID.json" in runbook
    assert "Never enable the evaluation bypass on the deployed graph worker" in runbook
    assert "restricted operator-only output" in runbook
    for gate_name in (
        "Permission isolation",
        "Fail-open parity",
        "Identity precision",
        "Retrieval quality",
        "Multi-hop value",
        "Latency",
        "Determinism",
        "Citations",
    ):
        assert runbook.count(f"| {gate_name} |") == 1


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
