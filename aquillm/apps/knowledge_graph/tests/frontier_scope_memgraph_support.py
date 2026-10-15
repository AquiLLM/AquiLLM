"""Guarded canonical staging and actual driver transcript for scope acceptance."""

from dataclasses import replace

import pytest

from apps.knowledge_graph.projection.frontier_reads_v2 import QUERIES
from apps.knowledge_graph.projection.memgraph_driver import Neo4jMemgraphDriver
from apps.knowledge_graph.projection.memgraph_repository import (
    MemgraphProjectionRepository,
)
from apps.knowledge_graph.tests.frontier_scope_window_support import MixedScopeFixture
from apps.knowledge_graph.tests.owned_memgraph_test_support import owned_memgraph
from apps.knowledge_graph.tests.test_memgraph_projection_repository import (
    _expected_manifest,
)


class LiveTranscript:
    def __init__(self, driver):
        self.driver, self.calls, self.transcript = driver, [], []
        self.scanned, self.phase = 0, "initialize"

    def execute_read(self, query, parameters, *, timeout_seconds, max_records):
        rows = self.driver.execute_read(
            query, parameters, timeout_seconds=timeout_seconds, max_records=max_records
        )
        self.calls.append((query, parameters.copy()))
        self.scanned += len(rows)
        if "lookups_csv" in parameters:
            pairs = tuple(
                tuple(p.split(":")) for p in parameters["lookups_csv"].split(",")
            )
        elif "keys_csv" in parameters:
            pairs = tuple(
                (parameters["generation_key"], key)
                for key in parameters["keys_csv"].split(",")
            )
        else:
            pairs = ()
        self.transcript.append(
            {
                "phase": self.phase,
                "kind": next(k for k, q in QUERIES.items() if q == query),
                "pairs": pairs,
                "rows": len(rows),
                "maximum": max_records,
            }
        )
        return rows


@pytest.fixture
def live_scope():
    with owned_memgraph() as target:
        if target is None:
            pytest.skip("requires the sanctioned owned Memgraph/settings harness")
        driver = Neo4jMemgraphDriver(
            target["uri"],
            "",
            "",
            database=target["database"],
            max_transaction_retry_time=0.0,
        )
        try:
            repository = MemgraphProjectionRepository(driver)
            staged = MixedScopeFixture()
            for bundle in staged.bundles:
                expected = replace(
                    _expected_manifest(bundle), private_mapping_checksum="d" * 64
                )
                repository.write_staging_generation(
                    bundle=bundle,
                    private_mapping_checksum="d" * 64,
                    batch_size=128,
                    timeout_seconds=5.0,
                    frontier_index=True,
                )
                validation = repository.validate_generation(
                    expected=expected, timeout_seconds=5.0
                )
                assert validation.valid
                repository.mark_generation_ready(
                    generation_key=repository.opaque_generation_key(
                        expected.generation_key
                    ),
                    validation_checksum=validation.validation_checksum,
                    timeout_seconds=5.0,
                )
            # Offline staging ends before this original invocation clock/ledger starts.
            fixture = MixedScopeFixture()
            fixture.driver = LiveTranscript(driver)
            yield fixture, driver
        finally:
            driver.close()
