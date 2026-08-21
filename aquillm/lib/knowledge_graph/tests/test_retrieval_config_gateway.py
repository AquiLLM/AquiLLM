from __future__ import annotations

import dataclasses

import pytest

from lib.knowledge_graph import retrieval_config as config
from lib.knowledge_graph.topology_gateway_config import GATEWAY_SETTING_KEYS

GATEWAY = {
    "KG_TOPOLOGY_GATEWAY_URL": "http://knowledge_graph_query_gateway:8092",
    "KG_TOPOLOGY_GATEWAY_BEARER_TOKEN": "gateway-secret",
    "KG_TOPOLOGY_GATEWAY_TIMEOUT_MS": "300",
    "KG_TOPOLOGY_GATEWAY_MAX_REQUEST_BYTES": "16384",
    "KG_TOPOLOGY_GATEWAY_MAX_RESPONSE_BYTES": "1048576",
}


def test_traversal_requires_gateway_not_web_bolt_credentials() -> None:
    old = {
        "KG_MEMGRAPH_TRAVERSAL_ENABLED": "1",
        "KG_MEMGRAPH_URI": "bolt://memgraph_knowledge_graph:7687",
        "KG_MEMGRAPH_QUERY_USERNAME": "reader",
        "KG_MEMGRAPH_QUERY_PASSWORD": "bolt-secret",
    }
    with pytest.raises(config.HybridRetrievalConfigError):
        config.load_hybrid_retrieval_settings(old)

    settings = config.load_hybrid_retrieval_settings(
        {"KG_MEMGRAPH_TRAVERSAL_ENABLED": "1", **GATEWAY}
    )
    assert settings.memgraph_traversal_enabled
    assert not hasattr(settings, "memgraph_query_username")
    assert not hasattr(settings, "memgraph_query_password")
    assert settings.memgraph_uri == ""


@pytest.mark.parametrize("key", tuple(GATEWAY))
def test_enabled_traversal_requires_every_gateway_field(key: str) -> None:
    with pytest.raises(config.HybridRetrievalConfigError, match=key):
        config.load_hybrid_retrieval_settings(
            {"KG_MEMGRAPH_TRAVERSAL_ENABLED": "1", **GATEWAY, key: ""}
        )


def test_django_loader_merges_pure_gateway_values_without_ambient_input() -> None:
    exposed = config.load_django_hybrid_retrieval_settings(
        {**GATEWAY, "KG_TOPOLOGY_GATEWAY_URL_TYPO": object(), "UNRELATED": object()}
    )
    core = {
        f"KG_{field.name.upper()}"
        for field in dataclasses.fields(config.HybridRetrievalSettings)
    }
    assert set(exposed) == core | GATEWAY_SETTING_KEYS
    assert exposed["KG_TOPOLOGY_GATEWAY_URL"] == GATEWAY["KG_TOPOLOGY_GATEWAY_URL"]
    assert "gateway-secret" not in repr(exposed)
