"""Pure resource and configuration validation for the isolated RAG harness."""
from pathlib import Path
import re

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


def _owned(labels, project):
    if labels.get(OWNER_LABEL) != "v1" or labels.get("com.docker.compose.project") != project:
        raise IsolationError("resource lacks the owned test-project labels")


def validate_resources(project, containers, networks, volumes, *, require_complete=True):
    validate_project(project)
    network_name = project + "_isolated"
    service_names = [c.get("Config", {}).get("Labels", {}).get("com.docker.compose.service") for c in containers]
    if containers and (len(service_names) != len(set(service_names)) or (require_complete and not SERVICES - {"frontend"} <= set(service_names))):
        raise IsolationError("incomplete or duplicate owned service inventory")
    identities = {c["Id"] for c in containers}
    for network in networks:
        _owned(network.get("Labels") or {}, project)
        if network.get("Name") != network_name or network.get("Internal") is not True or network.get("Driver") != "bridge":
            raise IsolationError("unexpected or externally connected network")
        if not set(network.get("Containers") or {}) <= identities:
            raise IsolationError("uninspected peers attached to owned network")
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
        if any(host.get(key) for key in ("Privileged", "PortBindings", "CapAdd", "Devices", "ExtraHosts")) or host.get("PidMode") == "host" or host.get("IpcMode") == "host" or host.get("NetworkMode") != network_name:
            raise IsolationError("host access or published ports forbidden")
        if set(container.get("NetworkSettings", {}).get("Networks", {})) != {network_name}:
            raise IsolationError("container is attached to an unexpected network")
        aliases = set(container["NetworkSettings"]["Networks"][network_name].get("Aliases") or [])
        if service not in aliases or (aliases & SERVICES) != {service}:
            raise IsolationError("service endpoint aliases do not match owned identity")
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


def validate_compose(config, project, *, root):
    if set(config["services"]) != SERVICES or set(config["networks"]) != {"isolated"} or config["networks"]["isolated"].get("internal") is not True:
        raise IsolationError("unexpected Compose services or networks")
    network = config["networks"]["isolated"]
    if network.get("external") or network.get("driver", "bridge") != "bridge" or network.get("driver_opts") or network.get("name") != project + "_isolated" or network.get("labels", {}).get(OWNER_LABEL) != "v1":
        raise IsolationError("Compose network is not owned")
    for name, service in config["services"].items():
        if name in {"app", "fake-models", "frontend"}:
            build = service.get("build", {})
            expected_file = "deploy/docker/test/Dockerfile.rag-remediation-browser" if name == "frontend" else "deploy/docker/test/Dockerfile.rag-remediation"
            if Path(build.get("context", "")).resolve() != root or build.get("dockerfile") != expected_file or any(build.get(key) for key in ("additional_contexts", "ssh", "secrets", "args", "dockerfile_inline", "network", "privileged")):
                raise IsolationError("unexpected build source or Dockerfile")
        elif service.get("image") != {"db": "pgvector/pgvector:pg16", "redis": "redis:7.4-bookworm", "memgraph": "memgraph/memgraph-mage:3.8.1"}[name]:
            raise IsolationError("unexpected dependency image")
        if service.get("labels", {}).get(OWNER_LABEL) != "v1" or set(service.get("networks", {})) != {"isolated"}:
            raise IsolationError("Compose ownership or network mismatch")
        if any(service.get(key) for key in ("ports", "env_file", "privileged", "network_mode", "devices", "extra_hosts", "cap_add", "pid", "ipc")):
            raise IsolationError("unsafe Compose service setting")
        if any(v.get("type") != "volume" or v.get("source") not in config["volumes"] for v in service.get("volumes", [])):
            raise IsolationError("Compose host or foreign volume")
    validate_environment(config["services"]["app"]["environment"])
    for volume in config["volumes"].values():
        if volume.get("external") or volume.get("driver_opts") or volume.get("labels", {}).get(OWNER_LABEL) != "v1" or not volume["name"].startswith(project + "_"):
            raise IsolationError("unsafe Compose volume")
