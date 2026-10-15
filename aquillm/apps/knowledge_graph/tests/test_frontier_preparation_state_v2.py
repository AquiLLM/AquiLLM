"""Finite preparation preserves physical reads and publication boundaries."""

import pytest

from apps.knowledge_graph.projection import frontier_copy_batch_v2 as copies
from apps.knowledge_graph.projection import frontier_scope_batch_v2 as scope
from apps.knowledge_graph.tests.frontier_copy_batch_support import CopyFixture
from apps.knowledge_graph.tests.frontier_task2_preparation_reference_v2 import (
    _prepare_copies_reference as reference_prepare,
)
from apps.knowledge_graph.tests.frontier_task2_preparation_reference_v2 import (
    _scope_rows_all_reference as reference_scope,
)


@pytest.mark.parametrize("relation_heads", [False, True])
def test_copy_finite_driver_preserves_complete_physical_plan(relation_heads):
    factory = getattr(copies, "CopyPreparation", None)
    assert factory is not None, "copy preparation needs resumable transitions"
    reference, actual = CopyFixture(), CopyFixture()
    expected = reference_prepare(
        reference.session, reference.pairs, relation_heads=relation_heads
    )
    state = factory(actual.session, actual.pairs, relation_heads=relation_heads)
    while (effect := state.next_read()) is not None:
        state.apply_result(actual.session.reads.read_batch(effect.kind, effect.pairs))
    assert state.result == expected
    assert actual.driver.calls == reference.driver.calls
    assert actual.ledger.source_rows == reference.ledger.source_rows
    assert actual.session.reads._index_records == reference.session.reads._index_records


def test_scope_finite_driver_preserves_breadth_waves():
    factory = getattr(scope, "ScopePreparation", None)
    assert factory is not None, "scope preparation needs resumable transitions"
    reference, actual = CopyFixture(), CopyFixture()
    sources = {
        g: [
            selected.collection_key,
            *(
                d.document_key
                for d in reference.session.ready.authorized_documents
                if d.generation_key == g
            ),
        ]
        for g, selected in reference.session.selected.items()
    }
    for fixture in (reference, actual):
        fixture.session.reads._index_records.clear()
    expected = reference_scope(reference.session.reads, sources)
    state = factory(actual.session.reads, sources)
    while (effect := state.next_read()) is not None:
        state.apply_result(actual.session.reads.read_batch(effect.kind, effect.pairs))
    assert state.result == expected
    assert actual.driver.calls == reference.driver.calls
    assert actual.ledger.source_rows == reference.ledger.source_rows
    assert actual.session.reads._index_records == reference.session.reads._index_records
