"""Captured competing pairs use the bounded exact generation/identity bitmask."""

from apps.knowledge_graph.projection.frontier_copy_preparation_v2 import CopyPreparation
from apps.knowledge_graph.retrieval.topology.frontier_window_checkpoint_v2 import (
    restore,
    snapshot,
)
from apps.knowledge_graph.tests.frontier_copy_batch_support import CopyFixture


def test_checkpoint_uses_exact_pair_mask_and_preserves_non_cartesian_pairs():
    fixture = CopyFixture()
    pairs = (fixture.pairs[0], (fixture.pairs[1][0], "f" * 64))
    state = CopyPreparation(fixture.session, pairs, relation_heads=False)
    state.next_read()
    operation, checkpoint = snapshot(state)
    assert "pair_mask" in operation and "pairs" not in operation
    assert len(bytes.fromhex(operation["pair_mask"])) == 1
    resumed = restore(fixture.session.reads, operation, checkpoint)
    assert set(resumed.pairs) == set(pairs)
