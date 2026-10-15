"""Finite workers execute the same fixed physical prefix in one occupation."""

from hashlib import sha256
from importlib import import_module
from importlib.util import find_spec
from time import monotonic

from apps.knowledge_graph.projection.frontier_copy_preparation_v2 import CopyPreparation
from apps.knowledge_graph.retrieval.topology.frontier_window_checkpoint_v2 import (
    snapshot,
)
from apps.knowledge_graph.retrieval.topology.frontier_window_codec_v2 import (
    decode_container,
)
from apps.knowledge_graph.tests.frontier_copy_batch_support import CopyFixture

PREFIX = "apps.knowledge_graph.retrieval.topology."


def test_worker_returns_replayable_exact_completed_physical_prefix():
    assert find_spec(PREFIX + "frontier_window_worker_v2"), (
        "finite occupied worker missing"
    )
    contracts = import_module(PREFIX + "frontier_window_contracts_v2")
    worker = import_module(PREFIX + "frontier_window_worker_v2")
    fixture = CopyFixture()
    state = CopyPreparation(fixture.session, fixture.pairs, relation_heads=True)
    state.next_read()
    operation, checkpoint = snapshot(state)
    grant = fixture.ledger.reserve_window()
    body = contracts.encode_request(
        fixture.session.ready,
        operation,
        checkpoint,
        grant,
        1,
        fixture.ledger.check_active,
    )
    request = contracts.decode_request(body)
    before = len(fixture.driver.calls)
    payload = worker.read_window(fixture.driver, request, monotonic() + 5, 1048576)
    frames = decode_container(payload, sha256(body).hexdigest())
    rows = 0
    for kind, physical in frames:
        effect = state.next_read(grant.planning_rows - rows)
        assert effect.kind == kind
        state.apply_result(physical)
        rows += len(physical)
    assert len(frames) == len(fixture.driver.calls) - before
    assert 1 < len(frames) <= 8
    fixture.ledger.settle_window(grant, queries=len(frames), scanned_rows=rows)
    while (effect := state.next_read()) is not None:
        state.apply_result(fixture.session.reads.read_batch(effect.kind, effect.pairs))
    assert sum(map(len, state.result.values())) == 2
