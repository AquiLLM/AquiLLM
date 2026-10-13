"""Fail-closed topology loader selection with no PostgreSQL fallback."""

from __future__ import annotations

from .memgraph import MemgraphProjectedTopologyLoader
from .postgres import PostgresParityCapability, PostgresProjectedTopologyLoader


def create_canonical_frontier_loader_v2(*, selection, codec, gateway_client):
    """Explicit private V2 entry point; legacy factory remains unchanged."""
    from apps.knowledge_graph.projection.frontier_adapter_v2 import (
        Neo4jCanonicalFrontierAdapterV2,
    )

    from .frontier_gateway_client_v2 import FrontierGatewayClientV2
    from .frontier_loader_v2 import canonical_frontier_loader_v2

    return canonical_frontier_loader_v2(
        selection=selection,
        codec=codec,
        adapter=Neo4jCanonicalFrontierAdapterV2(
            FrontierGatewayClientV2(gateway_client)
        ),
    )


def create_projected_topology_loader(*, backend: str, driver: object):
    if backend != "memgraph":
        raise ValueError("production topology backend must be memgraph")
    return MemgraphProjectedTopologyLoader(driver)  # type: ignore[arg-type]


def create_evaluation_projected_topology_loader(
    *, backend: str, source: object, capability: PostgresParityCapability
):
    if backend != "postgres" or type(capability) is not PostgresParityCapability:
        raise ValueError("postgres evaluation requires the exact private capability")
    return PostgresProjectedTopologyLoader(source, capability)


__all__ = [
    "create_evaluation_projected_topology_loader",
    "create_projected_topology_loader",
]
