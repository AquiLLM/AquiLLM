"""Run RAG checks in a labeled, internal-only disposable Compose project.

This module deliberately imports no Django or provider packages. Its pure
validators also run with host stdlib, before starting any application process.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
import uuid
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
OWNER_LABEL = "io.aquillm.rag-remediation-test"
SERVICES = {"app", "db", "redis", "memgraph", "fake-models", "frontend"}


class IsolationError(ValueError):
    pass


def validate_project(project):
    if not re.fullmatch(r"aquillm-rag-[a-z0-9][a-z0-9-]{2,48}", project):
        raise IsolationError("project must be a unique aquillm-rag-* test name")


def test_environment():
    return {
        "RAG_REMEDIATION_TEST": "1", "DJANGO_SETTINGS_MODULE": "aquillm.settings_rag_remediation_test",
        "DJANGO_DEBUG": "1", "DJANGO_TESTING": "1", "SECRET_KEY": "rag-synthetic-secret",
        "POSTGRES_HOST": "db", "POSTGRES_PORT": "5432", "POSTGRES_NAME": "rag_test",
        "POSTGRES_USER": "rag_test", "POSTGRES_PASSWORD": "rag-synthetic-password",
        "DJANGO_CACHE_REDIS_URL": "redis://redis:6379/1",
        "CELERY_BROKER_URL": "redis://redis:6379/2", "CELERY_RESULT_BACKEND": "redis://redis:6379/3",
        "OPENAI_BASE_URL": "http://fake-models:8000/v1", "ANTHROPIC_BASE_URL": "http://fake-models:8000",
        "APP_EMBED_BASE_URL": "http://fake-models:8000/v1", "APP_RERANK_BASE_URL": "http://fake-models:8000/v1",
        "VLLM_BASE_URL": "http://fake-models:8000/v1", "MEM0_LLM_BASE_URL": "http://fake-models:8000/v1",
        "MEM0_EMBED_BASE_URL": "http://fake-models:8000/v1", "INGEST_TRANSCRIBE_OPENAI_BASE_URL": "http://fake-models:8000/v1",
        "APP_OCR_QWEN_BASE_URL": "http://fake-models:8000/v1", "KG_QUERY_EXTRACTOR_URL": "http://fake-models:8000",
        "KG_TOPOLOGY_GATEWAY_URL": "http://fake-models.internal:8000", "KG_TOPOLOGY_GATEWAY_BEARER_TOKEN": "rag-synthetic-key", "KG_MEMGRAPH_URI": "bolt://memgraph:7687",
        "KG_PROJECTION_POSTGRES_SOURCE_DSN": "postgresql://aquillm_projection_source:rag-synthetic-password@db:5432/rag_test",
        "KG_PROJECTION_POSTGRES_STATE_DSN": "postgresql://aquillm_projection_state:rag-synthetic-password@db:5432/rag_test",
        "OPENAI_API_KEY": "rag-synthetic-key", "GEMINI_API_KEY": "rag-synthetic-key",
        "ANTHROPIC_API_KEY": "rag-synthetic-key", "COHERE_API_KEY": "rag-synthetic-key",
        "APP_EMBED_API_KEY": "rag-synthetic-key", "APP_RERANK_API_KEY": "rag-synthetic-key",
        "LLM_CHOICE": "OPENAI", "MEM0_ENABLED": "0", "KG_REQUIRE_POSTGRES_TESTS": "1",
        "DJANGO_ALLOWED_HOSTS": "app", "DJANGO_CSRF_TRUSTED_ORIGINS": "http://app:8000",
        "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    }


def validate_environment(environment):
    expected = test_environment()
    for key, value in expected.items():
        if environment.get(key) != value:
            raise IsolationError(f"unsafe or missing test setting: {key}")
    for key, value in environment.items():
        if key in expected or not value:
            continue
        # Unknown connection settings must not create alternate escape routes.
        if re.search(r"(?:URL|URI|DSN|HOST|ENDPOINT|PROXY)$", key) or "://" in value:
            raise IsolationError(f"unexpected connection setting: {key}")


def validate_browser(base_url, storage_state):
    if base_url != "http://app:8000":
        raise IsolationError("browser URL must belong to the owned app")
    path = PurePosixPath(storage_state)
    if ".." in path.parts or not re.fullmatch(r"/artifacts/browser/run-[a-f0-9-]+/state\.json", storage_state):
        raise IsolationError("browser storage must be fresh task-owned state")


def resolve_spec(spec):
    candidate = (ROOT / spec).resolve()
    if not candidate.is_relative_to((ROOT / "react").resolve()) or not candidate.is_file() or not candidate.name.endswith((".spec.ts", ".spec.js")):
        raise IsolationError("spec must resolve beneath react/ and be a Playwright spec")
    if re.search(r"page\.goto\(\s*['\"]https?://", candidate.read_text(encoding="utf-8")):
        raise IsolationError("spec page.goto URLs must be relative")
    return candidate.relative_to(ROOT / "react").as_posix()


def _owned(labels, project):
    if labels.get(OWNER_LABEL) != "v1" or labels.get("com.docker.compose.project") != project:
        raise IsolationError("resource lacks the owned test-project labels")


def validate_resources(project, containers, networks, volumes):
    validate_project(project)
    network_name = project + "_isolated"
    for network in networks:
        _owned(network.get("Labels") or {}, project)
        if network.get("Name") != network_name or network.get("Internal") is not True:
            raise IsolationError("unexpected or externally connected network")
    volume_names = set()
    for volume in volumes:
        _owned(volume.get("Labels") or {}, project)
        if volume.get("Driver") != "local" or volume.get("Options") or volume["Name"] not in {project + "_" + n for n in ("postgres", "artifacts", "frontend-build", "memgraph", "redis")}:
            raise IsolationError("unexpected volume or host-backed volume")
        volume_names.add(volume["Name"])
    for container in containers:
        config = container.get("Config") or {}
        labels = config.get("Labels") or {}
        _owned(labels, project)
        service = labels.get("com.docker.compose.service")
        if service not in SERVICES:
            raise IsolationError("unexpected test service")
        host = container.get("HostConfig") or {}
        if host.get("Privileged") or host.get("PortBindings") or host.get("NetworkMode") != network_name:
            raise IsolationError("host access or published ports forbidden")
        if set(container.get("NetworkSettings", {}).get("Networks", {})) != {network_name}:
            raise IsolationError("container is attached to an unexpected network")
        if any(m.get("Type") != "volume" or m.get("Name") not in volume_names for m in container.get("Mounts", [])):
            raise IsolationError("host or foreign volumes forbidden")
        if service == "app":
            raw = config.get("Env", [])
            env = dict(item.split("=", 1) for item in raw)
            if len(env) != len(raw):
                raise IsolationError("duplicate container environment keys")
            validate_environment(env)
    if containers and not networks:
        raise IsolationError("owned network missing")


def docker(*args, capture=False):
    result = subprocess.run(["docker", *args], cwd=ROOT, text=True, capture_output=capture)
    if result.returncode:
        # Never echo inspect/config output, which can contain secret values.
        raise IsolationError("Docker operation failed (details suppressed)")
    return result.stdout if capture else ""


def inspect_project(project):
    snapshots = []
    for kind, listing in (("container", ("ps", "-aq")), ("network", ("network", "ls", "-q")), ("volume", ("volume", "ls", "-q"))):
        identities = docker(*listing, "--filter", "label=com.docker.compose.project=" + project, capture=True).split()
        snapshots.append(json.loads(docker(kind, "inspect", *identities, capture=True)) if identities else [])
    validate_resources(project, *snapshots)
    return snapshots


def validate_compose(config, project):
    if set(config["services"]) != SERVICES or set(config["networks"]) != {"isolated"} or config["networks"]["isolated"].get("internal") is not True:
        raise IsolationError("unexpected Compose services or networks")
    for name, service in config["services"].items():
        if service.get("labels", {}).get(OWNER_LABEL) != "v1" or set(service.get("networks", {})) != {"isolated"}:
            raise IsolationError("Compose ownership or network mismatch")
        if any(service.get(key) for key in ("ports", "env_file", "privileged", "network_mode", "devices", "extra_hosts")):
            raise IsolationError("unsafe Compose service setting")
        if any(v.get("type") != "volume" or v.get("source") not in config["volumes"] for v in service.get("volumes", [])):
            raise IsolationError("Compose host or foreign volume")
    validate_environment(config["services"]["app"]["environment"])
    for volume in config["volumes"].values():
        if volume.get("external") or volume.get("driver_opts") or volume.get("labels", {}).get(OWNER_LABEL) != "v1" or not volume["name"].startswith(project + "_"):
            raise IsolationError("unsafe Compose volume")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--playwright")
    parser.add_argument("--base-url", default="http://app:8000")
    parser.add_argument("--storage-state")
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--manage", nargs=argparse.REMAINDER, help="run manage.py after the same resource validation")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        validate_project(args.project)
        # Reject inherited connection settings before Compose or Django startup.
        expected = test_environment()
        for key, value in os.environ.items():
            if key in expected and value != expected[key]:
                raise IsolationError(f"unsafe inherited test setting: {key}")
            if re.search(r"(?:DATABASE_URL|MODEL_ENDPOINT|BASE_URL|REDIS_URL|MEMGRAPH_URI)$", key) and key not in expected and value:
                raise IsolationError(f"unexpected inherited connection setting: {key}")
        state = args.storage_state or f"/artifacts/browser/run-{uuid.uuid4()}/state.json"
        validate_browser(args.base_url, state)
        if args.storage_state:
            raise IsolationError("externally supplied storage state is forbidden; wrapper creates fresh state")
        spec = resolve_spec(args.playwright) if args.playwright else None
        snapshots = inspect_project(args.project)
        if args.cleanup:
            containers, networks, volumes = snapshots
            for container in containers:
                docker("rm", "-f", container["Id"])
            for network in networks:
                docker("network", "rm", network["Id"])
            for volume in volumes:
                if not volume["Name"].endswith("_artifacts"):
                    docker("volume", "rm", volume["Name"])
            return 0
        with tempfile.TemporaryDirectory(prefix="rag-harness-") as temp:
            empty_env = Path(temp) / "empty.env"
            empty_env.write_text("", encoding="utf-8")
            compose = ["compose", "--env-file", str(empty_env), "-f", str(ROOT / "compose.rag-remediation-test.yml"), "-p", args.project]
            config = json.loads(docker(*compose, "--profile", "browser", "config", "--format", "json", capture=True))
            validate_compose(config, args.project)
            docker(*compose, "up", "--build", "-d", "--wait", "app")
            containers, networks, volumes = inspect_project(args.project)
            print("Owned resources: " + ", ".join(c["Name"].lstrip("/") for c in containers), flush=True)
            docker(*compose, "exec", "-T", "app", "python", "scripts/rag_remediation_provision.py")
            if spec:
                docker(*compose, "--profile", "browser", "up", "--build", "-d", "frontend")
                inspect_project(args.project)
                docker(*compose, "exec", "-T", "app", "python", "scripts/rag_remediation_seed_browser.py", state)
                command = ["docker", *compose, "exec", "-T", "-e", "RAG_PLAYWRIGHT_BASE_URL=" + args.base_url, "-e", "RAG_PLAYWRIGHT_STORAGE_STATE=" + state, "frontend", "npm", "run", "test:e2e", "--", "--config=playwright.rag-remediation.config.ts", spec]
            elif args.manage:
                command = ["docker", *compose, "exec", "-T", "app", "python", "aquillm/manage.py", *args.manage]
            else:
                tests = args.pytest_args
                if tests[:1] == ["--"]:
                    tests = tests[1:]
                command = ["docker", *compose, "exec", "-T", "app", "python", "-m", "pytest", *tests]
            return subprocess.run(command, cwd=ROOT).returncode
    except IsolationError as error:
        print(f"Isolation refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
