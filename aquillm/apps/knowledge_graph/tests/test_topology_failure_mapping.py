"""Typed topology failures must retain their local or shared classification."""

import pytest

from apps.knowledge_graph.retrieval.topology.contracts import (
    HybridBranchKind,
    ProjectedSeedV1,
    TopologyCapsV1,
    TopologyFailureReason,
)
from apps.knowledge_graph.retrieval.topology.gateway_client import (
    TopologyGatewayRequestError,
)
from apps.knowledge_graph.retrieval.topology.gateway_contracts import (
    GatewayFailureReason,
)
from apps.knowledge_graph.retrieval.topology.memgraph import (
    MemgraphProjectedTopologyLoader,
    TopologyLoadError,
)
from apps.knowledge_graph.tests.test_memgraph_topology import K, ready


class FailingDriver:
    def __init__(self, reason: TopologyFailureReason):
        self.reason = reason

    def execute_read(self, **_kwargs):
        raise TopologyLoadError(self.reason)


def _load(reason: TopologyFailureReason, branch: HybridBranchKind) -> None:
    MemgraphProjectedTopologyLoader(FailingDriver(reason)).load(
        ready=ready(),
        seeds=(ProjectedSeedV1(K[4], 1.0),),
        caps=TopologyCapsV1(branch, 32, 2, 200, 1_000, 20),
        deadline=42.5,
    )


@pytest.mark.parametrize(
    "reason",
    (
        TopologyFailureReason.BACKEND_AUTHENTICATION,
        TopologyFailureReason.BACKEND_UNAVAILABLE,
        TopologyFailureReason.BACKEND_PROVENANCE_MISMATCH,
        TopologyFailureReason.BACKEND_SCHEMA_MISMATCH,
    ),
)
def test_memgraph_preserves_typed_shared_driver_failures(reason) -> None:
    with pytest.raises(TopologyLoadError) as captured:
        _load(reason, HybridBranchKind.DIRECT)
    assert captured.value.reason is reason


@pytest.mark.parametrize(
    ("branch", "reason"),
    (
        (
            HybridBranchKind.DIRECT,
            TopologyFailureReason.DIRECT_TOPOLOGY_INVALID,
        ),
        (
            HybridBranchKind.EXTENDED,
            TopologyFailureReason.EXTENDED_TOPOLOGY_INVALID,
        ),
    ),
)
def test_memgraph_preserves_typed_branch_local_query_failures(branch, reason) -> None:
    with pytest.raises(TopologyLoadError) as captured:
        _load(reason, branch)
    assert captured.value.reason is reason


class GatewayFailingDriver:
    def __init__(self, reason: GatewayFailureReason):
        self.reason = reason

    def execute_read(self, **_kwargs):
        raise TopologyGatewayRequestError(self.reason)


@pytest.mark.parametrize(
    ("branch", "gateway_reason", "topology_reason"),
    (
        (
            HybridBranchKind.DIRECT,
            GatewayFailureReason.RESULT_CAP,
            TopologyFailureReason.DIRECT_TOPOLOGY_INVALID,
        ),
        (
            HybridBranchKind.EXTENDED,
            GatewayFailureReason.RESULT_CAP,
            TopologyFailureReason.EXTENDED_TOPOLOGY_INVALID,
        ),
        (
            HybridBranchKind.DIRECT,
            GatewayFailureReason.DEADLINE,
            TopologyFailureReason.DIRECT_TOPOLOGY_TIMEOUT,
        ),
        (
            HybridBranchKind.EXTENDED,
            GatewayFailureReason.DEADLINE,
            TopologyFailureReason.EXTENDED_TOPOLOGY_TIMEOUT,
        ),
    ),
)
def test_gateway_request_failures_remain_branch_local(
    branch, gateway_reason, topology_reason
) -> None:
    loader = MemgraphProjectedTopologyLoader(GatewayFailingDriver(gateway_reason))
    with pytest.raises(TopologyLoadError) as captured:
        loader.load(
            ready=ready(),
            seeds=(ProjectedSeedV1(K[4], 1.0),),
            caps=TopologyCapsV1(branch, 32, 2, 200, 1_000, 20),
            deadline=42.5,
        )
    assert captured.value.reason is topology_reason
