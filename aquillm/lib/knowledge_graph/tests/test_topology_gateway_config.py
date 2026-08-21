from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from lib.knowledge_graph.topology_gateway_config import (
    GatewayClientConfigError,
    TopologyGatewayClientSettings,
    django_topology_gateway_client_values,
    load_topology_gateway_client_settings,
)

VALID = {
    "KG_TOPOLOGY_GATEWAY_URL": "http://knowledge_graph_query_gateway:8092",
    "KG_TOPOLOGY_GATEWAY_BEARER_TOKEN": "gateway-secret",
    "KG_TOPOLOGY_GATEWAY_TIMEOUT_MS": "300",
    "KG_TOPOLOGY_GATEWAY_MAX_REQUEST_BYTES": "16384",
    "KG_TOPOLOGY_GATEWAY_MAX_RESPONSE_BYTES": "1048576",
}


def test_gateway_client_settings_default_off_and_redacted() -> None:
    defaults = load_topology_gateway_client_settings({})
    assert defaults.url == ""
    assert not defaults.bearer_token
    assert defaults.timeout_ms == 300
    settings = load_topology_gateway_client_settings(VALID, required=True)
    assert type(settings) is TopologyGatewayClientSettings
    assert settings.bearer_token.get_secret_value() == "gateway-secret"
    assert "gateway-secret" not in repr(settings)
    assert "knowledge_graph_query_gateway" not in repr(settings)
    with pytest.raises(FrozenInstanceError):
        settings.timeout_ms = 10  # type: ignore[misc]
    assert replace(settings) == settings


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("KG_TOPOLOGY_GATEWAY_URL", "https://public.example:8092"),
        ("KG_TOPOLOGY_GATEWAY_URL", "http://user:secret@localhost:8092"),
        ("KG_TOPOLOGY_GATEWAY_URL", "http://localhost:8092/path"),
        ("KG_TOPOLOGY_GATEWAY_BEARER_TOKEN", "secret\nvalue"),
        ("KG_TOPOLOGY_GATEWAY_TIMEOUT_MS", "0300"),
        ("KG_TOPOLOGY_GATEWAY_MAX_REQUEST_BYTES", "16383"),
        ("KG_TOPOLOGY_GATEWAY_MAX_RESPONSE_BYTES", "1048577"),
    ),
)
def test_gateway_client_settings_reject_unsafe_or_drifting_values(
    key: str, value: str
) -> None:
    with pytest.raises(GatewayClientConfigError, match=key):
        load_topology_gateway_client_settings({**VALID, key: value}, required=True)


def test_required_gateway_ignores_ambient_bolt_aliases_and_partial_values() -> None:
    with pytest.raises(GatewayClientConfigError, match="KG_TOPOLOGY_GATEWAY_URL"):
        load_topology_gateway_client_settings(
            {
                "KG_MEMGRAPH_URI": "bolt://memgraph_knowledge_graph:7687",
                "KG_MEMGRAPH_QUERY_PASSWORD": "not-a-gateway-token",
            },
            required=True,
        )
    with pytest.raises(GatewayClientConfigError):
        load_topology_gateway_client_settings(
            {"KG_TOPOLOGY_GATEWAY_URL": VALID["KG_TOPOLOGY_GATEWAY_URL"]}
        )


def test_django_export_contains_only_exact_gateway_values() -> None:
    exported = django_topology_gateway_client_values(
        {**VALID, "KG_TOPOLOGY_GATEWAY_URL_TYPO": object(), "UNRELATED": object()}
    )
    assert set(exported) == set(VALID)
    assert exported["KG_TOPOLOGY_GATEWAY_URL"] == VALID["KG_TOPOLOGY_GATEWAY_URL"]
    assert "gateway-secret" not in repr(exported)
