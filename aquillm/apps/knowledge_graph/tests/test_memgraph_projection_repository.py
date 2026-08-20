from __future__ import annotations

import uuid
from dataclasses import asdict
from hashlib import sha256

import pytest

from apps.knowledge_graph.projection.memgraph_driver import (
    MemgraphDriverError,
    MemgraphWriteSummaryV1,
    Neo4jMemgraphDriver,
)
from apps.knowledge_graph.projection.memgraph_repository import (
    MemgraphProjectionRepository,
)
from apps.knowledge_graph.projection.records import (
    CollectionGraphProjectionBundleV1,
    ProjectionGenerationManifestV1,
    ProjectionLifecycleState,
)
from apps.knowledge_graph.projection.serialization import projection_checksum
from apps.knowledge_graph.retrieval.topology.contracts import (
    HybridBranchKind,
    TopologyCapsV1,
)
from apps.knowledge_graph.tests.memgraph_test_support import (
    isolated_memgraph_container as _isolated_memgraph_container,
)
from apps.knowledge_graph.tests.test_projection_postgres_repository import _BundleSource


@pytest.fixture
def isolated_memgraph_container():
    yield from _isolated_memgraph_container.__wrapped__()


class _FakeDriver:
    def __init__(self) -> None:
        self.writes = []
        self.reads = []
        self.read_results = []

    def execute_write(self, cypher, parameters, *, timeout_seconds):
        self.writes.append((cypher, parameters, timeout_seconds))
        return MemgraphWriteSummaryV1({"nodes_created": 0})

    def execute_read(self, cypher, parameters, *, timeout_seconds, max_records):
        self.reads.append((cypher, parameters, timeout_seconds, max_records))
        return self.read_results.pop(0) if self.read_results else ()


def _bundle():
    return _BundleSource().load_projection_rows(
        projection_id=uuid.uuid4(), batch_size=10
    )


def _expected_manifest(bundle):
    checksum = projection_checksum(bundle)
    return ProjectionGenerationManifestV1(
        bundle.generation.generation_key,
        bundle.generation.schema_version,
        bundle.generation.projection_version,
        bundle.generation.identifier_key_version,
        checksum,
        checksum,
        sha256(b"[]").hexdigest(),
        bundle.counts,
        ProjectionLifecycleState.BUILDING,
    )


def _manifest_row(manifest):
    row = asdict(manifest)
    row.update(row.pop("counts"))
    row["state"] = manifest.state.value
    return row


def test_repository_uses_parameterized_idempotent_staging_and_ready_last() -> None:
    from apps.knowledge_graph.projection.records import (
        CollectionGraphProjectionBundleV1,
    )

    driver = _FakeDriver()
    repository = MemgraphProjectionRepository(driver)
    bundle = CollectionGraphProjectionBundleV1(**_bundle())

    repository.write_staging_generation(
        bundle=bundle, batch_size=2, timeout_seconds=1.0
    )

    assert all("MERGE" in cypher for cypher, _parameters, _timeout in driver.writes)
    assert all(
        "1" * 64 not in cypher for cypher, _parameters, _timeout in driver.writes
    )
    assert driver.writes[0][1]["state"] == "staging"
    assert not any(call[1].get("state") == "ready" for call in driver.writes)


def test_ready_marker_requires_matching_validated_manifest_and_is_last_write() -> None:
    driver = _FakeDriver()
    repository = MemgraphProjectionRepository(driver)
    bundle = CollectionGraphProjectionBundleV1(**_bundle())
    expected = _expected_manifest(bundle)
    driver.read_results.append((_manifest_row(expected),))
    driver.read_results.extend(
        [({"count": value},) for value in asdict(bundle.counts).values()]
        + [({"invalid": 0},)] * 5
    )

    validation = repository.validate_generation(
        expected=expected,
        timeout_seconds=1.0,
    )
    driver.read_results.append(
        (
            {
                "validation_checksum": validation.validation_checksum,
                "graph_checksum": validation.validation_checksum,
                "state": "building",
            },
        )
    )

    repository.mark_generation_ready(
        generation_key=repository.opaque_generation_key("1" * 64),
        validation_checksum=validation.validation_checksum,
        timeout_seconds=1.0,
    )

    assert validation.valid is True
    assert driver.writes[-2][1]["validation_checksum"] == expected.graph_checksum
    assert driver.writes[-1][1]["state"] == "ready"
    assert "MATCH" in driver.writes[-1][0]


def test_validation_rejects_count_or_endpoint_drift_without_publishing_token() -> None:
    driver = _FakeDriver()
    repository = MemgraphProjectionRepository(driver)
    bundle = CollectionGraphProjectionBundleV1(**_bundle())
    expected = _expected_manifest(bundle)
    driver.read_results.append((_manifest_row(expected),))
    driver.read_results.extend(
        [
            ({"count": value + (1 if index == 0 else 0)},)
            for index, value in enumerate(asdict(bundle.counts).values())
        ]
        + [({"invalid": 1},)]
        + [({"invalid": 0},)] * 4
    )

    validation = repository.validate_generation(
        expected=expected,
        timeout_seconds=1.0,
    )

    assert validation.valid is False
    assert driver.writes == []


def test_generation_deletion_is_generation_scoped_and_parameterized() -> None:
    driver = _FakeDriver()
    repository = MemgraphProjectionRepository(driver)
    key = repository.opaque_generation_key("a" * 64)

    repository.delete_generation(generation_key=key, timeout_seconds=1.0)

    cypher, parameters, _timeout = driver.writes[-1]
    assert "DETACH DELETE" in cypher
    assert "a" * 64 not in cypher
    assert parameters == {"generation_key": "a" * 64}


def test_generation_records_are_bounded_and_endpoint_closed() -> None:
    from apps.knowledge_graph.projection.records import (
        CollectionGraphProjectionBundleV1,
    )

    bundle = CollectionGraphProjectionBundleV1(**_bundle())
    driver = _FakeDriver()
    driver.read_results = [
        ({"record": asdict(bundle.generation)},),
        tuple({"record": asdict(row)} for row in bundle.entities),
        tuple({"record": asdict(row)} for row in bundle.automatic_memberships),
        (),
        (),
        (),
        (),
        tuple({"record": asdict(row)} for row in bundle.artifact_provenance),
    ]
    caps = TopologyCapsV1(HybridBranchKind.DIRECT, 2, 1, 2, 2, 2)

    observed = MemgraphProjectionRepository(driver).read_generation_records(
        generation_key=MemgraphProjectionRepository.opaque_generation_key("1" * 64),
        caps=caps,
        timeout_seconds=1.0,
    )

    assert observed == bundle
    assert all(call[3] <= 2 for call in driver.reads)


class _Record(dict):
    def data(self):
        return dict(self)


class _Result:
    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)

    def consume(self):
        return type("Summary", (), {"counters": {"nodes_created": 1}})()


class _Transaction:
    def __init__(self):
        self.calls = []

    def run(self, cypher, parameters, timeout):
        self.calls.append((cypher, parameters, timeout))
        return _Result([_Record(ok=1)])


class _Session:
    def __init__(self, transaction):
        self.transaction = transaction

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def execute_read(self, callback):
        return callback(self.transaction)

    def execute_write(self, callback):
        return callback(self.transaction)


class _Neo4jClient:
    def __init__(self):
        self.transaction = _Transaction()
        self.databases = []

    def session(self, *, database):
        self.databases.append(database)
        return _Session(self.transaction)


def test_driver_applies_fixed_database_timeout_parameters_and_bounds() -> None:
    client = _Neo4jClient()
    driver = Neo4jMemgraphDriver(
        "bolt://memgraph:7687", "reader", "secret", database="memgraph", driver=client
    )

    rows = driver.execute_read(
        "RETURN $value AS ok", {"value": 1}, timeout_seconds=0.5, max_records=1
    )

    assert rows == ({"ok": 1},)
    assert client.databases == ["memgraph"]
    assert client.transaction.calls == [("RETURN $value AS ok", {"value": 1}, 0.5)]


def test_driver_errors_are_fixed_and_do_not_expose_credentials_or_cypher() -> None:
    class Broken:
        def session(self, **_kwargs):
            raise RuntimeError("secret RETURN private")

    driver = Neo4jMemgraphDriver(
        "bolt://memgraph:7687", "reader", "secret", database="memgraph", driver=Broken()
    )
    with pytest.raises(MemgraphDriverError) as captured:
        driver.execute_read("RETURN private", {}, timeout_seconds=1.0, max_records=1)
    assert str(captured.value) == "memgraph_read_failed"
    assert "secret" not in repr(captured.value)


@pytest.mark.container
def test_memgraph_repository_against_isolated_container(
    isolated_memgraph_container,
) -> None:
    target = isolated_memgraph_container
    driver = Neo4jMemgraphDriver(target["uri"], "", "", database=target["database"])
    repository = MemgraphProjectionRepository(driver)
    repository.ensure_schema(timeout_seconds=5.0)
