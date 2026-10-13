# ruff: noqa: F401, F811
"""The existing staging worker is the index producer and keeps its ready gate."""

import pytest

from apps.knowledge_graph.projection.frontier_index_contracts_v2 import marker_present
from apps.knowledge_graph.projection.frontier_index_store_v2 import read_marker
from apps.knowledge_graph.projection.memgraph_repository import (
    MemgraphProjectionRepository,
)
from apps.knowledge_graph.tests.test_canonical_frontier_memgraph import (
    frontier_memgraph,
)
from apps.knowledge_graph.tests.test_projection_worker_heartbeat import (
    run_projection,  # noqa: F401
)


def test_existing_worker_requests_frontier_inside_original_stage_lifecycle(
    run_projection,
):
    calls = []
    run_projection.graph.write_staging_generation = lambda **kwargs: calls.append(
        kwargs
    )
    result = run_projection.run()
    assert result.ready
    assert calls[0]["frontier_index"] is True
    assert len(calls) == 1
    # Setup consumes the original stage allowance; it cannot restart or enlarge it.
    assert 0 < calls[0]["timeout_seconds"] <= 12.0
    assert run_projection.state.published == 1


@pytest.mark.parametrize("corrupt", [False, True])
def test_worker_real_index_must_validate_before_ready(
    run_projection, frontier_memgraph, corrupt
):
    fixture = run_projection
    repository = MemgraphProjectionRepository(frontier_memgraph)
    bundle = fixture.source.load_projection_bundle()
    generation = repository.opaque_generation_key(bundle.generation.generation_key)
    repository.delete_generation(generation_key=generation, timeout_seconds=5.0)

    def stage(**kwargs):
        repository.write_staging_generation(**kwargs)
        if corrupt:
            frontier_memgraph.execute_write(
                "MATCH (g:CollectionGeneration {generation_key:$generation_key}) "
                "SET g.frontier_root_key=$wrong",
                {"generation_key": generation.value, "wrong": "f" * 64},
                timeout_seconds=5.0,
            )

    fixture.graph.write_staging_generation = stage
    fixture.graph.validate_generation = repository.validate_generation
    fixture.graph.mark_generation_ready = repository.mark_generation_ready
    try:
        result = fixture.run()
        marker = read_marker(frontier_memgraph, generation.value, 5.0)
        if corrupt:
            assert not result.ready
            assert marker["state"] != "ready"
            assert fixture.state.published == 0
        else:
            assert result.ready and marker["state"] == "ready"
            assert marker_present(marker)
            assert fixture.state.published == 1
    finally:
        repository.delete_generation(generation_key=generation, timeout_seconds=5.0)


def test_ready_or_unclaimed_projection_never_enters_staging(run_projection):
    run_projection.state.claim = lambda **kwargs: None
    run_projection.graph.write_staging_generation = lambda **kwargs: pytest.fail(
        "ready generation was restaged"
    )
    assert not run_projection.run().ready
    assert run_projection.state.published == 0
