"""The shared worker remains occupied through timed-out window cleanup."""

import asyncio
from threading import Event
from time import monotonic

import pytest

from apps.knowledge_graph.projection.frontier_copy_preparation_v2 import CopyPreparation
from apps.knowledge_graph.retrieval.topology import frontier_binary_codec_v2 as codec
from apps.knowledge_graph.retrieval.topology.frontier_window_checkpoint_v2 import (
    snapshot,
)
from apps.knowledge_graph.retrieval.topology.frontier_window_contracts_v2 import (
    decode_request,
    encode_request,
)
from apps.knowledge_graph.retrieval.topology.frontier_window_worker_v2 import (
    read_window,
)
from apps.knowledge_graph.retrieval.topology.gateway_workers import GatewayWorkerPool
from apps.knowledge_graph.tests.frontier_copy_batch_support import CopyFixture
from apps.knowledge_graph.tests.frontier_scope_complete_support import scope_state
from apps.knowledge_graph.tests.frontier_window_test_support import WindowOpener


@pytest.mark.asyncio
@pytest.mark.parametrize("complete_scope", [False, True])
async def test_window_encoding_keeps_one_worker_occupied_after_timeout(
    monkeypatch, complete_scope
):
    fixture, release, pool = CopyFixture(), Event(), GatewayWorkerPool()
    if complete_scope:
        from apps.knowledge_graph.retrieval.topology.frontier_scope_window_contracts_v3 import (
            decode_request as decode,
        )
        from apps.knowledge_graph.retrieval.topology.frontier_scope_window_contracts_v3 import (
            encode_request as encode,
        )

        fixture, state = scope_state()
        ledger = fixture.budget
    else:
        state = CopyPreparation(fixture.session, fixture.pairs, relation_heads=True)
        ledger = fixture.ledger
        encode, decode = encode_request, decode_request
    state.next_read()
    operation, checkpoint = snapshot(state)
    grant = ledger.reserve_window()
    body = encode(
        fixture.session.ready,
        operation,
        checkpoint,
        grant,
        1,
        ledger.check_active,
    )
    request = decode(body)
    encode, entered = codec.encode_binary_response, []

    def blocked(payload):
        entered.append(True)
        assert release.wait(3)
        return encode(payload)

    monkeypatch.setattr(codec, "encode_binary_response", blocked)

    def call():
        deadline = monotonic() + 0.1
        return pool.run(
            read_window,
            fixture.driver,
            request,
            deadline,
            1048576,
            expires=deadline,
            complete_scope=complete_scope,
        )

    before = len(fixture.driver.calls)
    tasks = [asyncio.create_task(call()) for _ in range(4)]
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert all(isinstance(result, TimeoutError) for result in results)
        assert len(entered) == len(fixture.driver.calls) - before == 4
        with pytest.raises(RuntimeError, match="capacity"):
            await call()
        assert len(fixture.driver.calls) - before == 4
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        pool._executor.shutdown(wait=True)
    assert len(fixture.driver.calls) - before == 4


@pytest.mark.parametrize("failure", ["late", "close"])
def test_late_or_cleanup_failure_retains_original_hold_and_cache(monkeypatch, failure):
    fixture = CopyFixture()
    transport = WindowOpener(fixture, monkeypatch)
    before = fixture.ledger.queries, fixture.ledger.source_rows
    cache = dict(fixture.session.reads._index_records)
    open_response = transport.open

    def open(*args, **kwargs):
        response = open_response(*args, **kwargs)
        if failure == "late":
            fixture.session.reads.deadline = fixture.ledger.clock() - 1
        else:
            close = response.close

            def fail_close():
                close()
                raise OSError("synthetic cleanup uncertainty")

            response.close = fail_close
        return response

    transport.open = open
    with pytest.raises((TimeoutError, OSError)):
        fixture.session.prepare_copies(fixture.pairs, relation_heads=True)
    assert fixture.ledger._window is not None and fixture.ledger._uncertain
    assert (fixture.ledger.queries, fixture.ledger.source_rows) == before
    assert fixture.session.reads._index_records == cache
    assert transport.responses[0].closed
