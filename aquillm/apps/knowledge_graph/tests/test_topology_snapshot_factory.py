"""Pure factory wiring checks without requiring a configured Django application."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from apps.documents.services import hybrid_graph_dependencies as factory
from lib.knowledge_graph.topology_gateway_config import (
    django_topology_gateway_client_values,
    load_topology_gateway_client_settings,
)


@pytest.mark.parametrize("enabled", [True, False])
def test_django_mapping_factory_and_cached_client_share_exact_opt_in(
    monkeypatch, enabled
):
    env = {
        "KG_TOPOLOGY_GATEWAY_URL": "http://gateway.internal",
        "KG_TOPOLOGY_GATEWAY_BEARER_TOKEN": "test-secret",
        "KG_TOPOLOGY_GATEWAY_SNAPSHOT_ENABLED": "true" if enabled else "false",
    }
    values = django_topology_gateway_client_values(env)
    monkeypatch.setattr("django.conf.settings", SimpleNamespace(**values))
    settings = factory.django_topology_gateway_client_settings()
    assert settings == load_topology_gateway_client_settings(env)
    assert factory._topology_loader(settings).driver.snapshot_enabled is enabled
    opposite = factory._topology_loader(replace(settings, snapshot_enabled=not enabled))
    assert opposite.driver.snapshot_enabled is not enabled
