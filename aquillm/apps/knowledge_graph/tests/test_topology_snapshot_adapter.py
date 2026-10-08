"""The optional transport preserves the complete V1 result and fresh attestation."""

from unittest.mock import patch

import pytest

from apps.knowledge_graph.projection.topology_adapter import (
    Neo4jProjectedTopologyQueryAdapter,
)
from apps.knowledge_graph.retrieval.projected_types import (
    canonical_projected_snapshot_bytes,
)
from apps.knowledge_graph.retrieval.topology import memgraph
from apps.knowledge_graph.retrieval.topology.contracts import (
    ProjectedSeedV1,
    TopologyFailureReason,
)
from apps.knowledge_graph.retrieval.topology.failures import TopologyLoadError
from apps.knowledge_graph.tests.test_projected_topology_adapter import (
    ProjectionDriver,
    _caps,
    _ready,
)


def fixture():
    driver = ProjectionDriver()
    ready = _ready(driver.bundle)
    seeds = (ProjectedSeedV1(driver.bundle.entities[0].entity_key, 1.0),)
    args = dict(ready=ready, seeds=seeds, caps=_caps(), deadline=42.5)
    return driver, args, memgraph._parameters(ready, seeds, _caps())


def test_snapshot_operation_decodes_once_and_attests_every_warm_request():
    driver, args, parameters = fixture()
    adapter = Neo4jProjectedTopologyQueryAdapter(driver, clock=lambda: 40.0)
    golden = memgraph.MemgraphProjectedTopologyLoader(adapter).load(**args)
    operation = getattr(adapter, "execute_snapshot", None)
    assert callable(operation), "adapter must expose one bounded snapshot operation"
    driver.calls.clear()
    with patch.object(adapter, "_decode", wraps=adapter._decode) as decode:
        result = operation(parameters=parameters, deadline=42.5)
        assert decode.call_count == 1
    assert result.snapshot_json.encode() == canonical_projected_snapshot_bytes(golden)
    assert memgraph._manifest_matches(result.manifests, args["ready"])
    assert len(driver.calls) == 1  # cache hit still attests
    driver.manifest_checksum = "e" * 64
    with pytest.raises(TopologyLoadError) as caught:
        operation(parameters=parameters, deadline=42.5)
    assert caught.value.reason is TopologyFailureReason.READINESS_MISMATCH


def test_loader_uses_one_explicitly_enabled_snapshot_and_keeps_v1_golden(monkeypatch):
    driver, args, _ = fixture()
    adapter = Neo4jProjectedTopologyQueryAdapter(driver, clock=lambda: 40.0)
    golden = memgraph.MemgraphProjectedTopologyLoader(adapter).load(**args)
    assert callable(getattr(adapter, "execute_snapshot", None))
    monkeypatch.setattr(memgraph, "monotonic", lambda: 40.0)

    class SnapshotDriver:
        snapshot_enabled = True

        def __init__(self):
            self.calls = 0

        def execute_read(self, **kwargs):
            pytest.fail("a complete snapshot must need no family calls")

        def execute_snapshot(self, **kwargs):
            self.calls += 1
            return adapter.execute_snapshot(**kwargs)

    snapshot_driver = SnapshotDriver()
    actual = memgraph.MemgraphProjectedTopologyLoader(snapshot_driver).load(**args)
    assert canonical_projected_snapshot_bytes(
        actual
    ) == canonical_projected_snapshot_bytes(golden)
    assert snapshot_driver.calls == 1


@pytest.mark.parametrize(
    "field", ["state", "graph_checksum", "active_artifact_key", "membership_checksum"]
)
def test_warm_snapshot_cannot_bypass_any_fresh_attestation_field(monkeypatch, field):
    driver, _, parameters = fixture()
    adapter = Neo4jProjectedTopologyQueryAdapter(driver, clock=lambda: 40.0)
    adapter.execute_snapshot(parameters=parameters, deadline=42.5)
    original = driver.execute_read

    def changed(*args, **kwargs):
        rows = original(*args, **kwargs)
        if "AS collection_key" in args[0]:
            return ({**rows[0], field: "stale" if field == "state" else "e" * 64},)
        return rows

    monkeypatch.setattr(driver, "execute_read", changed)
    monkeypatch.setattr(
        adapter,
        "_snapshot",
        lambda *_args, **_kwargs: pytest.fail("stale manifest reached cache"),
    )
    with pytest.raises(TopologyLoadError) as caught:
        adapter.execute_snapshot(parameters=parameters, deadline=42.5)
    assert caught.value.reason is TopologyFailureReason.READINESS_MISMATCH


@pytest.mark.parametrize(
    "field,value",
    [
        ("bundle_checksum", "0" * 64),
        ("seed_checksum", "0" * 64),
        ("generation_keys_json", "[]"),
        ("document_keys_json", "[]"),
    ],
)
def test_snapshot_rejects_forged_authority_before_provider_io(field, value):
    driver, _, parameters = fixture()
    adapter = Neo4jProjectedTopologyQueryAdapter(driver, clock=lambda: 40.0)
    with pytest.raises(TopologyLoadError) as caught:
        adapter.execute_snapshot(parameters={**parameters, field: value}, deadline=42.5)
    assert caught.value.reason is TopologyFailureReason.AUTHORIZATION_CONTEXT_INVALID
    assert driver.calls == []
