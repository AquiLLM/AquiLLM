"""Generation-index pagination must not hide incident foreign edges."""

import pytest

from apps.knowledge_graph.projection import memgraph_edge_validation as edges
from apps.knowledge_graph.projection.memgraph_driver import MemgraphDriverError


class EdgeReader:
    def __init__(self, guard=()):
        self.guard = guard
        self.calls = []

    def execute_read(self, query, parameters, *, timeout_seconds, max_records):
        self.calls.append((query, parameters, timeout_seconds, max_records))
        if "AS invalid_incident_edge" in query:
            if isinstance(self.guard, Exception):
                raise self.guard
            return self.guard
        start = parameters["cursor_id"] + 1
        return tuple(
            {"cursor_key": "tie", "cursor_id": i, "identity": i}
            for i in range(start, min(3, start + parameters["page_limit"]))
        )


@pytest.mark.parametrize("query", edges._EDGE_QUERIES)
def test_generation_pages_run_one_bounded_unlabelled_incident_guard(query, monkeypatch):
    monkeypatch.setattr(edges, "PAGE_SIZE", 2)
    driver = EdgeReader()
    assert tuple(edges._stream_edge_family(driver, "g", query, 3, 10, 0.3)) == (
        {"identity": 0},
        {"identity": 1},
        {"identity": 2},
    )
    guards = [call for call in driver.calls if "AS invalid_incident_edge" in call[0]]
    assert len(guards) == 1
    guard, parameters, timeout, maximum = guards[0]
    assert "MATCH (source)-[edge:" in guard
    assert "->(target)" in guard
    assert (
        "source.generation_key = $generation_key "
        "OR target.generation_key = $generation_key" in guard
    )
    assert (
        "edge.generation_key IS NULL OR edge.generation_key <> $generation_key" in guard
    )
    assert "LIMIT 1" in guard
    assert (parameters, timeout, maximum) == ({"generation_key": "g"}, 0.3, 1)
    pages = [call for call in driver.calls if call not in guards]
    assert len(pages) == 2
    assert all(
        "WHERE edge.generation_key = $generation_key AND" in call[0] for call in pages
    )


@pytest.mark.parametrize("guard", [({"invalid_incident_edge": True},), None, [], ({},)])
def test_incident_guard_rejects_corruption_and_malformed_results_before_paging(guard):
    driver = EdgeReader(guard)
    with pytest.raises(ValueError, match="incident"):
        tuple(
            edges._stream_edge_family(driver, "g", edges._EDGE_QUERIES[0], 3, 10, 0.3)
        )
    assert len(driver.calls) == 1


def test_incident_guard_backend_failure_cannot_attest():
    driver = EdgeReader(MemgraphDriverError("memgraph_timeout"))
    with pytest.raises(MemgraphDriverError, match="memgraph_timeout"):
        tuple(
            edges._stream_edge_family(driver, "g", edges._EDGE_QUERIES[0], 3, 10, 0.3)
        )
    assert len(driver.calls) == 1
