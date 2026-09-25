from dataclasses import replace

import pytest

from apps.knowledge_graph.projection.topology_adapter import (
    Neo4jProjectedTopologyQueryAdapter,
)
from apps.knowledge_graph.retrieval.topology.contracts import (
    ProjectedSeedV1,
    ReadyGenerationBundleV1,
    TopologyFailureReason,
    TopologyQueryName,
    ready_generation_bundle_checksum,
)
from apps.knowledge_graph.retrieval.topology.failures import TopologyLoadError
from apps.knowledge_graph.retrieval.topology.memgraph import _parameters
from apps.knowledge_graph.tests.test_projected_topology_adapter import _caps, _ready
from apps.knowledge_graph.tests.test_projection_records import _bundle


def _request(count=2):
    original = _ready(_bundle())
    selected = tuple(
        replace(
            original.selected_generations[0],
            collection_key=collection * 64,
            generation_key=generation * 64,
            projection_key=projection * 64,
            active_artifact_key=artifact * 64,
            membership_checksum="8" * 64,
            graph_checksum="9" * 64,
        )
        for collection, generation, projection, artifact in (
            ("1", "b", "3", "4"), ("2", "a", "5", "6")
        )
    )
    selected = selected[:count]
    signature = original.authorization_context_signature
    ready = ReadyGenerationBundleV1(
        selected, (), signature, ready_generation_bundle_checksum(selected, (), signature)
    )
    return _parameters(ready, (ProjectedSeedV1("7" * 64, 1.0),), _caps()), tuple(
        {
            "collection_key": row.collection_key,
            "generation_key": row.generation_key,
            "projection_key": row.projection_key,
            "active_artifact_key": row.active_artifact_key,
            "graph_checksum": row.graph_checksum,
            "membership_checksum": row.membership_checksum,
            "state": "ready",
        }
        for row in selected
    )


class ManifestDriver:
    """Only replace the external database boundary; support old and batched reads."""

    def __init__(self, rows):
        self.rows, self.calls = rows, []

    def execute_read(self, cypher, parameters, *, timeout_seconds, max_records):
        self.calls.append((cypher, parameters, timeout_seconds, max_records))
        if "generation_key" in parameters:
            return tuple(
                row for row in self.rows
                if row["generation_key"] == parameters["generation_key"]
            )
        return self.rows


def _read(driver, *, clock=lambda: 40.0, max_records=2, count=2):
    parameters, _ = _request(count)
    return Neo4jProjectedTopologyQueryAdapter(driver, clock=clock).execute_read(
        query=TopologyQueryName.GENERATION_MANIFESTS,
        parameters=parameters,
        deadline=42.5,
        max_records=max_records,
    )


def test_manifests_use_one_bounded_scalar_read_and_restore_selected_order():
    _, rows = _request()
    driver = ManifestDriver(rows[::-1])
    result = _read(driver)
    assert result == tuple({k: v for k, v in row.items() if k != "state"} for row in rows)
    assert [row["generation_key"] for row in result] == ["b" * 64, "a" * 64]
    assert len(driver.calls) == 1
    cypher, parameters, timeout, cap = driver.calls[0]
    assert parameters == {"generation_keys_csv": "b" * 64 + "," + "a" * 64, "row_limit": 3}
    assert "LIMIT $row_limit" in cypher
    assert "b" * 64 not in cypher
    assert timeout == 2.5
    assert cap == 2


@pytest.mark.parametrize("change", ["missing", "duplicate", "unexpected", "unready", "over_cap"])
def test_manifests_reject_inexact_membership_or_unready_rows(change):
    _, rows = _request()
    if change == "missing":
        rows = rows[:1]
    elif change == "duplicate":
        rows = (rows[0], rows[0])
    elif change == "unexpected":
        rows = (rows[0], {**rows[1], "generation_key": "f" * 64})
    elif change == "unready":
        rows = (rows[0], {**rows[1], "state": "staging"})
    else:
        rows = rows + ({**rows[0], "generation_key": "f" * 64},)
    with pytest.raises(TopologyLoadError) as captured:
        _read(ManifestDriver(rows))
    assert captured.value.reason is TopologyFailureReason.READINESS_MISMATCH


@pytest.mark.parametrize("field", [
    "collection_key", "projection_key", "active_artifact_key",
    "graph_checksum", "membership_checksum",
])
def test_manifests_reject_stale_selected_provenance(field):
    _, rows = _request()
    rows = (rows[0], {**rows[1], field: "f" * 64})
    with pytest.raises(TopologyLoadError) as captured:
        _read(ManifestDriver(rows))
    assert captured.value.reason is TopologyFailureReason.READINESS_MISMATCH


def test_manifest_batch_rejects_completion_after_original_deadline():
    _, rows = _request(1)
    times = iter((40.0, 42.6))
    with pytest.raises(TimeoutError):
        _read(ManifestDriver(rows), clock=lambda: next(times), count=1, max_records=1)


def test_manifest_batch_rejects_expired_deadline_before_database_read():
    _, rows = _request()
    driver = ManifestDriver(rows)
    with pytest.raises(TimeoutError):
        _read(driver, clock=lambda: 42.6)
    assert driver.calls == []


def test_manifest_batch_rejects_wrong_requested_cap_before_database_read():
    _, rows = _request()
    driver = ManifestDriver(rows)
    with pytest.raises(TopologyLoadError) as captured:
        _read(driver, max_records=1)
    assert captured.value.reason is TopologyFailureReason.BACKEND_SCHEMA_MISMATCH
    assert driver.calls == []
