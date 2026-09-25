"""Provider-neutral parity through the fixed topology gateway wire contract."""

from __future__ import annotations

import pytest

from apps.knowledge_graph.projection.memgraph_driver import MemgraphDriverError
from apps.knowledge_graph.projection.topology_adapter import (
    Neo4jProjectedTopologyQueryAdapter,
)
from apps.knowledge_graph.retrieval.topology.contracts import (
    ProjectedSeedV1,
    TopologyFailureReason,
    TopologyQueryName,
)
from apps.knowledge_graph.retrieval.topology.failures import TopologyLoadError
from apps.knowledge_graph.retrieval.topology.gateway_contracts import (
    TopologyGatewayRequestV1,
    TopologyGatewaySuccessV1,
    decode_request,
    decode_response,
    encode_request,
    encode_response,
)
from apps.knowledge_graph.retrieval.topology.memgraph import (
    MemgraphProjectedTopologyLoader,
)
from apps.knowledge_graph.tests.test_projected_topology_adapter import (
    ProjectionDriver,
    _caps,
    _ready,
)
from apps.knowledge_graph.tests.test_projection_records import _bundle


class CanonicalGatewayRoundTrip:
    def __init__(self, adapter):
        self.adapter = adapter

    def execute_read(self, *, query, parameters, deadline, max_records):
        request = decode_request(
            encode_request(
                TopologyGatewayRequestV1(query, parameters, deadline, max_records)
            )
        )
        rows = self.adapter.execute_read(
            query=request.query,
            parameters=request.parameters,
            deadline=request.deadline,
            max_records=request.max_records,
        )
        response = decode_response(encode_response(TopologyGatewaySuccessV1(rows)))
        assert type(response) is TopologyGatewaySuccessV1
        return response.rows


class ResultCapProjectionDriver(ProjectionDriver):
    def execute_read(self, cypher, parameters, *, timeout_seconds, max_records):
        if "AS collection_key" in cypher:
            return super().execute_read(
                cypher,
                parameters,
                timeout_seconds=timeout_seconds,
                max_records=max_records,
            )
        raise MemgraphDriverError("memgraph_result_limit")


def _loader(driver):
    return MemgraphProjectedTopologyLoader(
        Neo4jProjectedTopologyQueryAdapter(driver, clock=lambda: 40.0)
    )


def test_gateway_wire_snapshot_is_exactly_direct_adapter_snapshot() -> None:
    bundle = _bundle()
    ready = _ready(bundle)
    seeds = (ProjectedSeedV1(bundle.entities[0].entity_key, 1.0),)
    direct = _loader(ProjectionDriver(bundle)).load(
        ready=ready, seeds=seeds, caps=_caps(), deadline=42.5
    )
    gateway = MemgraphProjectedTopologyLoader(
        CanonicalGatewayRoundTrip(
            Neo4jProjectedTopologyQueryAdapter(
                ProjectionDriver(bundle), clock=lambda: 40.0
            )
        )
    ).load(ready=ready, seeds=seeds, caps=_caps(), deadline=42.5)
    assert gateway == direct


def test_gateway_wire_rejects_arbitrary_operations() -> None:
    with pytest.raises(TypeError):
        TopologyGatewayRequestV1(
            "MATCH (n) RETURN n",
            {},
            42.5,
            1,  # type: ignore[arg-type]
        )
    assert tuple(TopologyQueryName) == (
        TopologyQueryName.GENERATION_MANIFESTS,
        TopologyQueryName.AUTOMATIC_MEMBERSHIPS,
        TopologyQueryName.RELATION_TOPOLOGY,
        TopologyQueryName.EVIDENCE_MENTIONS,
    )


def test_adapter_maps_bounded_result_cap_to_branch_local_invalid() -> None:
    bundle = _bundle()
    with pytest.raises(TopologyLoadError) as captured:
        _loader(ResultCapProjectionDriver(bundle)).load(
            ready=_ready(bundle),
            seeds=(ProjectedSeedV1(bundle.entities[0].entity_key, 1.0),),
            caps=_caps(),
            deadline=42.5,
        )
    assert captured.value.reason is TopologyFailureReason.DIRECT_TOPOLOGY_INVALID
