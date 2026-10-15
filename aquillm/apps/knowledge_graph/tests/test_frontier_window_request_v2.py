"""Bounded finite stage DTOs reject untyped or arbitrary continuations."""

from copy import deepcopy

import pytest

from apps.knowledge_graph.projection.frontier_copy_preparation_v2 import CopyPreparation
from apps.knowledge_graph.retrieval.topology import (
    frontier_window_contracts_v2 as contracts,
)
from apps.knowledge_graph.retrieval.topology.frontier_window_checkpoint_v2 import (
    snapshot,
)
from apps.knowledge_graph.retrieval.topology.frontier_window_validation_v2 import (
    validate_checkpoint,
)
from apps.knowledge_graph.tests.frontier_copy_batch_support import CopyFixture


@pytest.mark.parametrize(
    "mutation",
    [
        "boolean_target",
        "foreign_target",
        "extra",
        "raw_body",
        "depth",
        "guard_overflow",
        "duplicate_guard",
        "sequence",
        "grant",
        "ready",
    ],
)
def test_request_rejects_invalid_checkpoint_and_original_grant(mutation):
    fixture = CopyFixture()
    state = CopyPreparation(fixture.session, fixture.pairs, relation_heads=True)
    state.next_read()
    operation, checkpoint = snapshot(state)
    if mutation == "boolean_target":
        checkpoint["paths"][0][1] = False
    elif mutation == "foreign_target":
        checkpoint["paths"][0][0] = 128
    elif mutation == "extra":
        checkpoint["program"] = []
    elif mutation == "raw_body":
        checkpoint["fresh"] = []
    elif mutation == "depth":
        checkpoint["paths"][0][-1] = -1
    elif mutation == "guard_overflow":
        checkpoint["guard"] = [[0, "a" * 64]] * 5001
    elif mutation == "duplicate_guard":
        checkpoint["guard"] = [[0, "a" * 64]] * 2
    else:
        grant = fixture.ledger.reserve_window()
        body = contracts.encode_request(
            fixture.session.ready,
            operation,
            checkpoint,
            grant,
            1,
            fixture.ledger.check_active,
        )
        value = contracts._load(body, maximum=4194304)
        if mutation == "sequence":
            value["sequence"] = True
        elif mutation == "grant":
            value["grant"]["rows"] = 3200
        else:
            value["ready"]["authorized_documents"] *= 10001
        with pytest.raises((ValueError, TypeError)):
            contracts.decode_request(contracts._canonical(value))
        return
    with pytest.raises((ValueError, TypeError)):
        validate_checkpoint(fixture.session.ready, operation, checkpoint)


def test_checkpoint_does_not_encode_cached_index_bodies_or_fresh_source_rows():
    fixture = CopyFixture()
    state = CopyPreparation(fixture.session, fixture.pairs, relation_heads=True)
    state.next_read()
    operation, checkpoint = snapshot(state)
    original = deepcopy(checkpoint)
    fixture.session.reads._index_records = {
        pair: ("forbidden", {"body": "never send"})
        for pair in fixture.session.reads._index_records
    }
    assert snapshot(state) == (operation, original)
