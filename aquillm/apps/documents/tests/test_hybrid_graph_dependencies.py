"""Production assembly contracts for the projected hybrid graph overlay."""

from __future__ import annotations

from types import SimpleNamespace

from apps.documents.services.hybrid_graph_dependencies import (
    _topology_loader,
    build_hybrid_graph_dependencies,
)
from apps.documents.tests.hybrid_graph_test_support import Policy, authorization
from apps.knowledge_graph.retrieval.topology.gateway_client import (
    TopologyGatewayClient,
)
from apps.knowledge_graph.retrieval.topology.memgraph import (
    MemgraphProjectedTopologyLoader,
)
from lib.knowledge_graph.topology_gateway_config import (
    load_topology_gateway_client_settings,
)


def _settings(**overrides: object):
    values = {
        "memgraph_traversal_enabled": True,
        "graph_direct_enabled": True,
        "graph_extended_enabled": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_factory_is_default_off_and_never_constructs_runtime_without_capability():
    observed: list[str] = []

    def runtime_factory(**_kwargs):
        observed.append("runtime")
        raise AssertionError("disabled path constructed a runtime")

    assert (
        build_hybrid_graph_dependencies(
            authorization=None,
            settings=_settings(),
            runtime_factory=runtime_factory,
        )
        is None
    )
    assert (
        build_hybrid_graph_dependencies(
            authorization=object(),
            settings=_settings(memgraph_traversal_enabled=False),
            runtime_factory=runtime_factory,
        )
        is None
    )
    assert observed == []


def test_factory_binds_one_request_runtime_and_its_attested_materializer():
    request_authorization = authorization(Policy())
    runtime = SimpleNamespace(
        prepare_shared=lambda **_kwargs: None,
        run_direct=lambda **_kwargs: None,
        prepare_extended=lambda **_kwargs: None,
        run_extended=lambda **_kwargs: None,
        materialize=lambda **_kwargs: ("attested",),
    )
    observed: list[dict[str, object]] = []

    def runtime_factory(**kwargs):
        observed.append(kwargs)
        return runtime

    dependencies = build_hybrid_graph_dependencies(
        authorization=request_authorization,
        settings=_settings(),
        runtime_factory=runtime_factory,
    )

    assert dependencies is not None
    assert dependencies.runtime is runtime
    assert dependencies.materialize(
        chunk_keys=("opaque",), authorization=request_authorization, outcome=object()
    ) == ("attested",)
    assert observed == [
        {
            "authorization": request_authorization,
            "settings": dependencies.settings,
        }
    ]


def test_factory_reauthorizes_before_constructing_provider_runtime():
    policy = Policy()
    request_authorization = authorization(policy)
    policy.rows = ()
    observed: list[str] = []

    def runtime_factory(**_kwargs):
        observed.append("runtime")
        raise AssertionError("revoked scope constructed a runtime")

    assert (
        build_hybrid_graph_dependencies(
            authorization=request_authorization,
            settings=_settings(),
            runtime_factory=runtime_factory,
        )
        is None
    )
    assert observed == []


def test_shipping_topology_loader_is_http_only_and_redacted() -> None:
    gateway = load_topology_gateway_client_settings(
        {
            "KG_TOPOLOGY_GATEWAY_URL": "http://knowledge_graph_query_gateway:8092",
            "KG_TOPOLOGY_GATEWAY_BEARER_TOKEN": "gateway-secret",
            "KG_TOPOLOGY_GATEWAY_TIMEOUT_MS": "300",
            "KG_TOPOLOGY_GATEWAY_MAX_REQUEST_BYTES": "16384",
            "KG_TOPOLOGY_GATEWAY_MAX_RESPONSE_BYTES": "1048576",
        },
        required=True,
    )
    loader = _topology_loader(gateway)
    assert type(loader) is MemgraphProjectedTopologyLoader
    assert type(loader.driver) is TopologyGatewayClient
    assert "gateway-secret" not in repr(loader.driver)
    source = __import__(
        "apps.documents.services.hybrid_graph_dependencies", fromlist=["x"]
    ).__file__
    assert source is not None
    text = open(source, encoding="utf-8").read()
    assert "Neo4jMemgraphDriver" not in text
    assert "Neo4jProjectedTopologyQueryAdapter" not in text
