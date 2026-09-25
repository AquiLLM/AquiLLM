from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

import pytest

from apps.knowledge_graph.projection.memgraph_driver import _parameters
from apps.knowledge_graph.projection.memgraph_edges import write_parameterized_batches
from apps.knowledge_graph.projection.memgraph_repository import (
    MemgraphProjectionRepository,
)
from apps.knowledge_graph.tests.test_memgraph_projection_repository import _FakeDriver
from apps.knowledge_graph.tests.test_projection_records import _bundle


@pytest.mark.parametrize("batch_size, expected_transactions", [(1, 19), (2, 15)])
def test_projection_writes_one_transaction_per_nonempty_family_batch(
    batch_size, expected_transactions
):
    driver = _FakeDriver()
    MemgraphProjectionRepository(driver).write_staging_generation(
        bundle=_bundle(),
        private_mapping_checksum="d" * 64,
        batch_size=batch_size,
        timeout_seconds=1.0,
    )

    assert len(driver.writes) == expected_transactions


def _large_bundle():
    bundle = _bundle()
    mentions = tuple(
        sorted(
            (
                replace(
                    bundle.entity_mentions[0],
                    mention_key=sha256(f"mention:{index}".encode()).hexdigest(),
                )
                for index in range(257)
            ),
            key=lambda row: row.mention_key,
        )
    )
    return replace(
        bundle,
        entity_mentions=mentions,
        counts=replace(bundle.counts, entity_mention_count=len(mentions)),
    )


def test_large_requested_batches_preserve_every_mention_with_bounded_scalar_queries():
    driver = _FakeDriver()
    bundle = _large_bundle()
    MemgraphProjectionRepository(driver).write_staging_generation(
        bundle=bundle,
        private_mapping_checksum="d" * 64,
        batch_size=5_000,
        timeout_seconds=1.0,
    )

    for label in ("ProjectedEntityMention", "ENTITY_MENTION"):
        calls = [call for call in driver.writes if label in call[0]]
        keys = [
            [value for key, value in params.items() if key.endswith("_mention_key")]
            for _, params, _ in calls
        ]
        assert [len(batch) for batch in keys] == [128, 128, 1]
        assert [key for batch in keys for key in batch] == [
            row.mention_key for row in bundle.entity_mentions
        ]
    for query, params, timeout in driver.writes[1:]:
        assert _parameters(params) == params
        assert len(params) <= 4_096
        assert len(query.encode()) <= 131_072
        assert timeout == 1.0
        assert bundle.generation.generation_key not in query


@pytest.mark.parametrize("failure", ["exception", "summary"])
def test_batch_failure_stops_later_writes_and_retry_replays_exact_parameters(failure):
    calls = []

    class Driver:
        def execute_write(self, query, params, **kwargs):
            calls.append((query, params, kwargs))
            if failure == "exception":
                raise RuntimeError("failed batch")
            return None

    rows = tuple({"value": index} for index in range(3))
    kwargs = dict(generation_key="a" * 64, batch_size=2, timeout_seconds=0.5)
    with pytest.raises(RuntimeError if failure == "exception" else TypeError):
        write_parameterized_batches(Driver(), "RETURN row.value", rows, **kwargs)
    assert len(calls) == 1
    retry = _FakeDriver()
    write_parameterized_batches(retry, "RETURN row.value", rows, **kwargs)
    assert retry.writes[0][:2] == calls[0][:2]
    assert len(retry.writes) == 2


@pytest.mark.parametrize("batch_size", [0, 5001, True])
def test_invalid_edge_batch_size_does_not_write(batch_size):
    driver = _FakeDriver()
    with pytest.raises(ValueError, match="batch_size"):
        write_parameterized_batches(
            driver,
            "RETURN row.value",
            [{"value": 1}],
            generation_key="a" * 64,
            batch_size=batch_size,
            timeout_seconds=1.0,
        )
    assert not driver.writes
