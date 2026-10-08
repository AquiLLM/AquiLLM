"""Qdrant connection configuration, including the SDK's keyed HTTP behavior."""

import warnings

import httpx
import pytest
from mem0.configs.vector_stores.qdrant import QdrantConfig
from qdrant_client import QdrantClient

from lib.memory.mem0.config_builder import build_mem0_oss_config_dict


@pytest.fixture(autouse=True)
def qdrant_environment(monkeypatch):
    for name in ("MEM0_QDRANT_API_KEY", "MEM0_QDRANT_URL", "MEM0_EMBED_DIMS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MEM0_QDRANT_HOST", "qdrant")
    monkeypatch.setenv("MEM0_QDRANT_PORT", "6333")
    monkeypatch.setenv("MEM0_COLLECTION_NAME", "existing-memory")


def vector_config():
    return build_mem0_oss_config_dict(graph_enabled_override=False)[0]["vector_store"][
        "config"
    ]


@pytest.mark.parametrize("key", [None, "", "   "])
def test_anonymous_qdrant_preserves_host_port_without_api_key(monkeypatch, key):
    if key is not None:
        monkeypatch.setenv("MEM0_QDRANT_API_KEY", key)
    assert vector_config() == {
        "host": "qdrant",
        "port": 6333,
        "collection_name": "existing-memory",
    }


def test_keyed_docker_qdrant_uses_explicit_http_url(monkeypatch):
    monkeypatch.setenv("MEM0_QDRANT_API_KEY", "test-only-key")
    monkeypatch.setenv("MEM0_QDRANT_HOST", "memory-store")
    monkeypatch.setenv("MEM0_QDRANT_PORT", "6334")
    assert vector_config() == {
        "url": "http://memory-store:6334",
        "api_key": "test-only-key",
        "collection_name": "existing-memory",
    }


@pytest.mark.parametrize("key", [None, "test-only-key"])
def test_explicit_qdrant_url_overrides_host_port(monkeypatch, key):
    monkeypatch.setenv("MEM0_QDRANT_URL", " https://memory.example:7443 ")
    if key:
        monkeypatch.setenv("MEM0_QDRANT_API_KEY", key)
    expected = {
        "url": "https://memory.example:7443",
        "collection_name": "existing-memory",
    }
    if key:
        expected["api_key"] = key
    assert vector_config() == expected


def test_actual_mem0_schema_and_qdrant_sdk_keep_keyed_docker_requests_http(monkeypatch):
    monkeypatch.setenv("MEM0_QDRANT_API_KEY", "test-only-key")
    config = QdrantConfig(**vector_config())
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200, json={"result": {"collections": []}, "status": "ok", "time": 0}
        )

    # Only HTTP transport is replaced: the actual SDK builds URLs and auth headers.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Api key is used with an insecure connection.*"
        )
        client = QdrantClient(
            url=config.url,
            host=config.host,
            port=config.port,
            api_key=config.api_key,
            check_compatibility=False,
            transport=httpx.MockTransport(respond),
        )
        try:
            client.get_collections()
        finally:
            client.close()
    assert len(requests) == 1
    assert str(requests[0].url) == "http://qdrant:6333/collections"
    assert requests[0].headers["api-key"] == "test-only-key"
