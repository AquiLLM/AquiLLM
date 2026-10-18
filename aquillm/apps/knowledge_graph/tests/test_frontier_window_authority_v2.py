"""Window grouping retains current source and final authority checks."""

import pytest

from apps.knowledge_graph.projection.frontier_reads_v2 import QUERIES
from apps.knowledge_graph.retrieval.topology.failures import TopologyLoadError
from apps.knowledge_graph.tests.frontier_copy_batch_support import CopyFixture
from apps.knowledge_graph.tests.frontier_window_test_support import WindowOpener


@pytest.mark.parametrize("mutation", ["duplicate", "foreign", "absent", "raw"])
def test_window_source_corruption_rejects_without_publishing_or_retrying(
    monkeypatch, mutation
):
    fixture = CopyFixture()
    transport = WindowOpener(fixture, monkeypatch)
    cache = dict(fixture.session.reads._index_records)

    def mutate(query, parameters, rows):
        if mutation == "raw" and query == QUERIES["raw_batch_multi"]:
            rows[0]["record"]["generation_key"] = "0" * 64
        elif mutation != "raw" and query == QUERIES["index_batch_multi"]:
            if mutation == "duplicate":
                return (*rows, rows[0])
            if mutation == "foreign":
                rows[0]["generation_key"] = "0" * 64
            else:
                return rows[1:]
        return rows

    fixture.driver.transform = mutate
    with pytest.raises(Exception):
        fixture.session.prepare_copies(fixture.pairs, relation_heads=True)
    assert fixture.session.reads._index_records == cache
    assert fixture.ledger._uncertain and fixture.ledger._window is not None
    assert len(transport.requests) == 1
    assert transport.responses[0].closed


def test_fresh_final_marker_revocation_after_valid_window_is_rejected(monkeypatch):
    fixture = CopyFixture()
    WindowOpener(fixture, monkeypatch)
    assert fixture.session.prepare_copies(fixture.pairs, relation_heads=True)
    marker = next(iter(fixture.driver.markers.values()))
    marker["frontier_root_key"] = "0" * 64
    with pytest.raises(TopologyLoadError):
        fixture.session.final_attestation()
    assert fixture.ledger.queries == len(fixture.driver.calls)
    assert fixture.ledger.source_rows == fixture.driver.scanned
