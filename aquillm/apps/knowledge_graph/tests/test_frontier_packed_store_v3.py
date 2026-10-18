"""Reader-first publication and stored-recipe reconstruction of derived indexes."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest

from apps.knowledge_graph.projection import frontier_index_store_v2 as store
from apps.knowledge_graph.projection.frontier_index_build_v2 import build_frontier_index
from apps.knowledge_graph.projection.frontier_index_contracts_v2 import VERSION
from apps.knowledge_graph.projection.memgraph_driver import MemgraphWriteSummaryV1
from apps.knowledge_graph.projection.memgraph_pagination import (
    FAMILY_IDENTITY_FIELDS,
    canonical_record_key,
    full_family_page_query,
)
from apps.knowledge_graph.projection.memgraph_streaming import _BUNDLE_FIELDS
from apps.knowledge_graph.projection.serialization import projection_checksum
from apps.knowledge_graph.tests.test_frontier_packed_directory_v3 import PACKED
from apps.knowledge_graph.tests.test_projection_records import _bundle


class StoreDriver:
    """Independent source rows and exact stored records behind existing queries."""

    def __init__(self, version=None, state="staging"):
        self.bundle = _bundle()
        self.index = build_frontier_index(
            self.bundle,
            **(
                {"directory_version": version} if version not in {None, VERSION} else {}
            ),
        )
        self.marker = {
            **asdict(self.bundle.generation),
            "state": state,
            "graph_checksum": projection_checksum(self.bundle),
        }
        if version is not None:
            self.marker.update(self.index.marker)
        self.writes = []

    def execute_read(self, query, parameters, *, timeout_seconds, max_records):
        if "RETURN g AS record" in query:
            return ({"record": self.marker.copy()},)
        if "n:FrontierIndexV2" in query:
            return tuple(
                {"record": r}
                for r in self.index.records
                if r["index_key"] > parameters["after_key"]
            )[:max_records]
        for field, label in _BUNDLE_FIELDS:
            if label and query == full_family_page_query(label):
                rows = sorted(
                    getattr(self.bundle, field),
                    key=lambda r: canonical_record_key(label, r),
                )
                return tuple(
                    {
                        "record": {
                            **asdict(r),
                            "opaque_key": getattr(r, FAMILY_IDENTITY_FIELDS[label]),
                        }
                    }
                    for r in rows
                )[:max_records]
        raise AssertionError(query)

    def execute_write(self, query, parameters, *, timeout_seconds):
        self.writes.append((query, parameters.copy()))
        if "g.validated_frontier_checksum=$frontier_checksum" in query:
            self.marker["validated_frontier_checksum"] = parameters["frontier_checksum"]
        elif "g.frontier_version=$frontier_version" in query:
            self.marker.update(
                {k: v for k, v in parameters.items() if k != "generation_key"}
            )
        return MemgraphWriteSummaryV1({"nodes_created": 0})

    def validate(self):
        expected = SimpleNamespace(
            generation_key=self.bundle.generation.generation_key,
            graph_checksum=projection_checksum(self.bundle),
            counts=self.bundle.counts,
        )
        return store.validate_frontier_if_present(
            self, expected, timeout=5.0, marker=self.marker
        )

    def write(self):
        store.write_frontier_index(self, self.bundle, batch_size=32, timeout=5.0)


def test_new_unmarked_staging_uses_packed_recipe_after_explicit_activation(monkeypatch):
    monkeypatch.setenv("KG_FRONTIER_PACKED_WRITES_ENABLED", "1")
    driver = StoreDriver()
    driver.write()
    assert driver.marker["frontier_version"] == PACKED


@pytest.mark.parametrize("setting", [None, "0"])
def test_unmarked_staging_keeps_compatible_recipe_until_activation(
    monkeypatch, setting
):
    monkeypatch.delenv("KG_FRONTIER_PACKED_WRITES_ENABLED", raising=False)
    if setting is not None:
        monkeypatch.setenv("KG_FRONTIER_PACKED_WRITES_ENABLED", setting)
    driver = StoreDriver()
    driver.write()
    assert driver.marker["frontier_version"] == VERSION


@pytest.mark.parametrize("setting", ["true", "false", "", " 1", "2"])
def test_invalid_packed_activation_refuses_before_writes(monkeypatch, setting):
    monkeypatch.setenv("KG_FRONTIER_PACKED_WRITES_ENABLED", setting)
    driver = StoreDriver()
    with pytest.raises(ValueError, match="KG_FRONTIER_PACKED_WRITES_ENABLED"):
        driver.write()
    assert not driver.writes


def test_private_writer_rollback_only_selects_unmarked_staging(monkeypatch):
    monkeypatch.setenv("KG_FRONTIER_PACKED_WRITES_ENABLED", "1")
    monkeypatch.setattr(store, "_NEW_INDEX_VERSION", VERSION)
    driver = StoreDriver()
    driver.write()
    assert driver.marker["frontier_version"] == VERSION


@pytest.mark.parametrize("version", [VERSION, PACKED])
@pytest.mark.parametrize("state", ["ready", "staging"])
def test_verifier_reconstructs_stored_recipe_independent_of_new_writer(
    monkeypatch, version, state
):
    driver = StoreDriver(version, state)
    if state == "staging":
        driver.marker["validated_frontier_checksum"] = "0" * 64
    monkeypatch.setattr(
        store,
        "_NEW_INDEX_VERSION",
        PACKED if version == VERSION else VERSION,
        raising=False,
    )
    driver.validate()
    assert (
        driver.marker["validated_frontier_checksum"]
        == driver.index.marker["frontier_checksum"]
    )


@pytest.mark.parametrize("version", [VERSION, PACKED])
def test_complete_staging_resume_uses_existing_recipe(monkeypatch, version):
    driver = StoreDriver(version)
    driver.marker["validated_frontier_checksum"] = "0" * 64
    monkeypatch.setattr(
        store,
        "_NEW_INDEX_VERSION",
        PACKED if version == VERSION else VERSION,
        raising=False,
    )
    driver.write()
    assert driver.marker["frontier_version"] == version
    assert (
        driver.marker["frontier_checksum"] == driver.index.marker["frontier_checksum"]
    )


@pytest.mark.parametrize(
    "mutation", ["partial", "unknown", "wrong_recipe", "digest_type"]
)
def test_invalid_existing_recipe_refuses_before_writes(mutation):
    driver = StoreDriver(VERSION)
    if mutation == "partial":
        del driver.marker["frontier_root_key"]
    elif mutation == "unknown":
        driver.marker["frontier_version"] = "unknown"
    elif mutation == "wrong_recipe":
        driver.marker["frontier_version"] = PACKED
    else:
        driver.marker["frontier_checksum"] = []
    with pytest.raises(ValueError):
        driver.write()
    assert not driver.writes


def test_mixed_recipe_marker_cannot_validate_or_supply_ready_cas():
    driver = StoreDriver(VERSION, "ready")
    driver.marker["frontier_version"] = PACKED
    with pytest.raises(ValueError):
        driver.validate()
    assert not driver.writes
    clean, clause = store.frontier_ready_parameters(
        StoreDriver(VERSION, "ready").marker
    )
    assert (
        clean["frontier_version"] == VERSION
        and "g.frontier_version=$frontier_version" in clause
    )


def test_ready_generation_cannot_be_rewritten():
    driver = StoreDriver(VERSION, "ready")
    with pytest.raises(ValueError, match="staging"):
        driver.write()
    assert not driver.writes


@pytest.mark.parametrize("version", [VERSION, PACKED])
def test_owned_memgraph_staging_validation_and_ready_audit_keep_recipe(
    monkeypatch, version
):
    from apps.knowledge_graph.tests.owned_memgraph_test_support import owned_memgraph

    with owned_memgraph() as target:
        if target is None:
            pytest.skip("requires the sanctioned owned Memgraph/settings harness")
        from dataclasses import replace

        from apps.knowledge_graph.projection.memgraph_driver import Neo4jMemgraphDriver
        from apps.knowledge_graph.projection.memgraph_repository import (
            MemgraphProjectionRepository,
        )
        from apps.knowledge_graph.projection.records import ProjectionLifecycleState
        from apps.knowledge_graph.tests.test_memgraph_projection_repository import (
            _expected_manifest,
        )

        monkeypatch.setattr(store, "_NEW_INDEX_VERSION", version)
        monkeypatch.setenv("KG_FRONTIER_PACKED_WRITES_ENABLED", "1")
        driver = Neo4jMemgraphDriver(
            target["uri"],
            "",
            "",
            database=target["database"],
            max_transaction_retry_time=0.0,
        )
        try:
            bundle = _bundle()
            repository = MemgraphProjectionRepository(driver)
            expected = replace(
                _expected_manifest(bundle), private_mapping_checksum="d" * 64
            )
            repository.write_staging_generation(
                bundle=bundle,
                private_mapping_checksum="d" * 64,
                batch_size=32,
                timeout_seconds=5.0,
                frontier_index=True,
            )
            validation = repository.validate_generation(
                expected=expected, timeout_seconds=5.0
            )
            assert validation.valid
            generation = repository.opaque_generation_key(
                bundle.generation.generation_key
            )
            repository.mark_generation_ready(
                generation_key=generation,
                validation_checksum=validation.validation_checksum,
                timeout_seconds=5.0,
            )
            marker = store.read_marker(driver, generation.value, 5.0)
            assert marker["frontier_version"] == version
            clean, clause = store.frontier_ready_parameters(marker)
            assert clean["frontier_version"] == version
            assert "g.frontier_version=$frontier_version" in clause
            monkeypatch.setattr(
                store, "_NEW_INDEX_VERSION", VERSION if version == PACKED else PACKED
            )
            assert repository.validate_generation(
                expected=replace(expected, state=ProjectionLifecycleState.READY),
                timeout_seconds=5.0,
            ).valid
            assert store.read_marker(driver, generation.value, 5.0) == marker
        finally:
            driver.close()
