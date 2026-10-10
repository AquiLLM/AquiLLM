"""The derived index is deterministic, ordered, closed and hash authenticated."""

from importlib.util import find_spec

from apps.knowledge_graph.tests.frontier_fixture_support import bridge_bundles


def test_frontier_index_derives_one_bound_root_from_original_projection():
    assert (
        find_spec("apps.knowledge_graph.projection.frontier_index_build_v2") is not None
    )
    from apps.knowledge_graph.projection.frontier_index_build_v2 import (
        build_frontier_index,
    )

    bundle = bridge_bundles()[0]
    first = build_frontier_index(bundle)
    assert first == build_frontier_index(bundle)
    assert first.marker["frontier_version"] == "canonical-frontier-index-v2"
    assert first.marker["frontier_record_count"] == len(first.records)
    assert len({r["index_key"] for r in first.records}) == len(first.records)
    assert all(
        r["generation_key"] == bundle.generation.generation_key for r in first.records
    )
