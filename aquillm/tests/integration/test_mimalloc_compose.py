"""Allocator controls must reach services even without a shared env_file."""

import json
from pathlib import Path

import pytest

from tests.integration.compose_render_test_support import (
    render_compose_with_reviewed_env,
)

ROOT = Path(__file__).resolve().parents[3]
COMPOSE_FILES = (
    "base.yml",
    "development.yml",
    "production.yml",
    "no_gpu_dev.yml",
    "test.yml",
)


@pytest.mark.parametrize("compose_name", COMPOSE_FILES)
@pytest.mark.parametrize("allocator", [None, "mimalloc", "system"])
def test_allocator_global_defaults_and_rollback_reach_every_python_service(
    compose_name, allocator
):
    config = render_compose_with_reviewed_env(
        (ROOT / "deploy/compose" / compose_name,),
        profile="*",
        environment_overrides={"AQUILLM_ALLOCATOR": allocator} if allocator else {},
    )
    checked = 0
    for service in config["services"].values():
        dockerfile = service.get("build", {}).get("dockerfile", "")
        if dockerfile.startswith(
            (
                "deploy/docker/web/",
                "deploy/docker/knowledge-graph/",
                "deploy/docker/vllm/",
                "deploy/docker/test/",
            )
        ):
            checked += 1
            assert service["environment"]["AQUILLM_ALLOCATOR"] == (
                allocator or "mimalloc"
            )
            assert service["environment"]["PYTHONMALLOC"] == "default"
        else:
            # Shared env_file entries can carry this inert selector into other
            # services. Activation requires a wrapped image/entrypoint or preload.
            if dockerfile:
                assert dockerfile == "deploy/docker/certbot/Dockerfile"
            else:
                assert service["image"].split(":", 1)[0] in {
                    "minio/mc", "minio/minio", "pgvector/pgvector",
                    "memgraph/memgraph-mage", "redis", "nginx",
                    "qdrant/qdrant", "localstack/localstack",
                }
            assert not service.get("environment", {}).get("LD_PRELOAD")
            wiring = json.dumps({
                key: service.get(key)
                for key in ("entrypoint", "command", "volumes")
            })
            for marker in (
                "aquillm-allocator", "with_mimalloc", "libmimalloc", "/opt/mimalloc"
            ):
                assert marker not in wiring
    assert checked >= 4


@pytest.mark.parametrize(
    "compose_name", ["base.yml", "development.yml", "production.yml"]
)
def test_one_service_can_trial_all_python_allocations_without_changing_others(
    compose_name,
):
    config = render_compose_with_reviewed_env(
        (
            ROOT / "deploy/compose" / compose_name,
            ROOT / "deploy/compose/nemotron-asr.yml",
        ),
        profile="*",
        environment_overrides={
            "AQUILLM_ALLOCATOR": "system",
            "VLLM_RERANK_ALLOCATOR": "mimalloc",
            "VLLM_RERANK_PYTHONMALLOC": "malloc",
        },
    )
    services = config["services"]
    assert services["vllm_rerank"]["environment"]["AQUILLM_ALLOCATOR"] == "mimalloc"
    assert services["vllm_rerank"]["environment"]["PYTHONMALLOC"] == "malloc"
    for name in (
        "web",
        "worker",
        "vllm",
        "vllm_transcribe",
        "knowledge_graph_query_gateway",
    ):
        assert services[name]["environment"]["AQUILLM_ALLOCATOR"] == "system"
        assert services[name]["environment"]["PYTHONMALLOC"] == "default"
