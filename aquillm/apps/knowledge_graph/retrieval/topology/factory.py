"""Fail-closed topology loader selection with no PostgreSQL fallback."""

from __future__ import annotations

from ..frontier_ranking_validation_v2 import POLICY, frontier_tags
from .memgraph import MemgraphProjectedTopologyLoader
from .postgres import PostgresParityCapability, PostgresProjectedTopologyLoader


def create_canonical_frontier_loader_v2(
    *, selection, codec, gateway_client, acquisition_policy=POLICY
):
    """Explicit private V2 entry point; legacy factory remains unchanged."""
    from apps.knowledge_graph.projection.frontier_adapter_v2 import (
        Neo4jCanonicalFrontierAdapterV2,
    )

    from .frontier_gateway_client_v2 import FrontierGatewayClientV2
    from .frontier_loader_v2 import canonical_frontier_loader_v2
    frontier_tags(acquisition_policy)

    return canonical_frontier_loader_v2(
        selection=selection,
        codec=codec,
        acquisition_policy=acquisition_policy,
        adapter=Neo4jCanonicalFrontierAdapterV2(
            FrontierGatewayClientV2(gateway_client), acquisition_policy=acquisition_policy
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
