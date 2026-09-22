"""Known local wire overflow must not cancel a successful graph sibling."""

from dataclasses import replace
from hashlib import sha256

import pytest

from apps.knowledge_graph.retrieval.branch_contracts import (
    BranchStatusV1,
    DirectBranchFailureReason,
    ExtendedBranchFailureReason,
)
from apps.knowledge_graph.retrieval.scheduler import HybridGraphBranchScheduler
from apps.knowledge_graph.retrieval.topology import gateway_client, gateway_contracts
from apps.knowledge_graph.retrieval.topology.contracts import (
    AuthorizedProjectedDocumentV1,
    HybridBranchKind,
    ProjectedSeedV1,
    ReadyGenerationBundleV1,
    TopologyQueryName,
    ready_generation_bundle_checksum,
)
from apps.knowledge_graph.retrieval.topology.gateway_client import (
    TopologyGatewayClient,
    TopologyGatewayRequestError,
)
from apps.knowledge_graph.retrieval.topology.gateway_contracts import (
    GatewayFailureReason,
)
from apps.knowledge_graph.retrieval.topology.memgraph import (
    MemgraphProjectedTopologyLoader,
)
from apps.knowledge_graph.tests.test_projected_topology_adapter import _caps, _ready
from apps.knowledge_graph.tests.test_projection_records import _bundle
from apps.knowledge_graph.tests.test_retrieval_branch_scheduler import (
    _Runtime,
    _settings,
)


def _client(monkeypatch):
    network_calls = []

    def forbidden_network(*args):
        network_calls.append(args)
        raise AssertionError("local validation must precede HTTP")

    monkeypatch.setattr(gateway_client, "build_opener", forbidden_network)
    monkeypatch.setattr(gateway_client.time, "monotonic", lambda: 40.0)
    return TopologyGatewayClient(
        "http://gateway.internal", "secret-canary", 1.5
    ), network_calls


@pytest.mark.parametrize("branch", tuple(HybridBranchKind))
def test_actual_oversized_scoped_request_preserves_successful_sibling(
    monkeypatch, branch
):
    # Reproduce the deployed 16KiB boundary with a complete, valid 31-document,
    # 64-seed request. Exercise real serialization, not an injected exception.
    monkeypatch.setattr(gateway_contracts, "MAX_REQUEST_BYTES", 16_384)
    original = _ready(_bundle())
    selected = original.selected_generations[0]
    documents = tuple(
        AuthorizedProjectedDocumentV1(
            key, selected.collection_key, selected.generation_key
        )
        for key in sorted(
            sha256(f"document:{i}".encode()).hexdigest() for i in range(31)
        )
    )
    ready = ReadyGenerationBundleV1(
        original.selected_generations,
        documents,
        original.authorization_context_signature,
        ready_generation_bundle_checksum(
            original.selected_generations,
            documents,
            original.authorization_context_signature,
        ),
    )
    seeds = tuple(
        ProjectedSeedV1(key, 1.0 / 64)
        for key in sorted(sha256(f"seed:{i}".encode()).hexdigest() for i in range(64))
    )
    caps = replace(_caps(), branch_kind=branch, max_seeds=64)
    client, network_calls = _client(monkeypatch)
    runtime = _Runtime()

    def load(**kwargs):
        return MemgraphProjectedTopologyLoader(client).load(
            ready=ready, seeds=seeds, caps=caps, deadline=42.5
        )

    failing = "direct" if branch is HybridBranchKind.DIRECT else "extended"
    sibling = "extended" if failing == "direct" else "direct"
    setattr(runtime, "run_" + failing, load)
    result = HybridGraphBranchScheduler(runtime, clock=lambda: 40.0).run(
        query="q",
        baseline=object(),
        authorization=object(),
        settings=_settings(),
        deadline=42.5,
    )
    expected = (
        DirectBranchFailureReason.DIRECT_TOPOLOGY_INVALID
        if branch is HybridBranchKind.DIRECT
        else ExtendedBranchFailureReason.EXTENDED_TOPOLOGY_INVALID
    )
    assert getattr(result, failing).failure_reason is expected
    assert getattr(result, sibling).status is BranchStatusV1.SUCCEEDED
    assert result.shared_failure_reason is None
    assert network_calls == []


@pytest.mark.parametrize(
    "parameters",
    (
        {"a": "x" * 200},
        {"a": "x" * 60, "b": "y" * 60},
        {"a": "x" * 50},
    ),
)
def test_scalar_mapping_and_envelope_byte_caps_are_local_and_redacted(
    monkeypatch, parameters
):
    monkeypatch.setattr(gateway_contracts, "MAX_REQUEST_BYTES", 100)
    client, network_calls = _client(monkeypatch)
    with pytest.raises(TopologyGatewayRequestError) as captured:
        client.execute_read(
            query=TopologyQueryName.GENERATION_MANIFESTS,
            parameters=parameters,
            deadline=42.5,
            max_records=1,
        )
    assert captured.value.reason is GatewayFailureReason.RESULT_CAP
    assert "secret-canary" not in repr(captured.value)
    assert "xxxxxxxx" not in repr(captured.value)
    assert network_calls == []


@pytest.mark.parametrize("value", ("forbidden\ncontrol", object(), float("nan")))
def test_malformed_local_parameters_are_not_reclassified_as_capacity(
    monkeypatch, value
):
    client, network_calls = _client(monkeypatch)
    with pytest.raises((TypeError, ValueError)) as captured:
        client.execute_read(
            query=TopologyQueryName.GENERATION_MANIFESTS,
            parameters={"authority": value},
            deadline=42.5,
            max_records=1,
        )
    assert not isinstance(captured.value, TopologyGatewayRequestError)
    assert network_calls == []
