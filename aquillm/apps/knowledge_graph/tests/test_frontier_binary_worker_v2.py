"""Binary encoding remains inside the occupied worker after caller timeout."""

import asyncio
from threading import Event
from time import monotonic

import pytest

from apps.knowledge_graph.retrieval.topology import frontier_binary_codec_v2 as codec
from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
    FrontierBudgetExceeded,
)
from apps.knowledge_graph.retrieval.topology.frontier_gateway_contracts_v2 import (
    BINARY_PATH,
    FrontierCompletedBudgetFailureV2,
    FrontierGatewayBatchRequestV2,
)
from apps.knowledge_graph.retrieval.topology.frontier_gateway_service_v2 import (
    read_payload,
)
from apps.knowledge_graph.retrieval.topology.gateway_workers import GatewayWorkerPool
from apps.knowledge_graph.tests.frontier_generation_batch_support import (
    GenerationFixture,
)
from apps.knowledge_graph.tests.test_frontier_generation_gateway_v2 import (
    http_gateway as http_gateway,
)


@pytest.mark.asyncio
async def test_encoding_holds_worker_until_cleanup_finishes_even_after_timeout(
    monkeypatch,
):
    fixture, release = GenerationFixture(2, 4), Event()
    pool, entered = GatewayWorkerPool(), []
    original = codec.encode_binary_response
    generation = fixture.ready.selected_generations[0].generation_key
    request = FrontierGatewayBatchRequestV2(
        "manifest_batch", ((generation, generation),), fixture.ready, 1048576
    )

    def blocked(payload):
        entered.append(True)
        assert release.wait(3)
        return original(payload)

    monkeypatch.setattr(codec, "encode_binary_response", blocked)

    def call():
        deadline = monotonic() + 0.1
        return pool.run(
            read_payload,
            fixture.driver,
            request,
            deadline,
            1048576,
            binary=True,
            expires=deadline,
        )

    tasks = [asyncio.create_task(call()) for _ in range(4)]
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert all(isinstance(result, TimeoutError) for result in results)
        assert len(entered) == len(fixture.driver.calls) == 4
        with pytest.raises(RuntimeError, match="capacity"):
            await call()
        assert len(fixture.driver.calls) == 4
        release.set()
        for _ in range(200):
            try:
                response = await call()
                break
            except RuntimeError:
                await asyncio.sleep(0.001)
        else:
            pytest.fail("occupied workers did not release after encoding finished")
        assert codec.decode_binary_response(response)["scanned_rows"] == 1
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        pool._executor.shutdown(wait=True)


def test_real_binary_http_preserves_conservative_final_hold_and_scalar_route(
    http_gateway,
):
    fixture, requests = http_gateway
    fixture.session.reads.driver.binary_batches = True
    fixture.session.initialize()
    hold = (
        fixture.budget.max_wire_bytes
        - fixture.budget.wire_bytes
        - fixture.budget.available_wire_bytes()
    )
    assert hold > 0
    fixture.budget.charge_page(
        scanned_rows=0, wire_bytes=fixture.budget.available_wire_bytes()
    )
    before = len(requests)
    with pytest.raises(FrontierBudgetExceeded):
        fixture.session.reads.read(
            "index", fixture.ready.selected_generations[0].generation_key, ("a" * 64,)
        )
    assert len(requests) == before
    used = fixture.budget.wire_bytes
    fixture.session.final_attestation()
    assert 0 < fixture.budget.wire_bytes - used < hold
    assert all(path == BINARY_PATH for path, _, _ in requests)
    assert not fixture.budget._authority_reservations and not fixture.budget._uncertain
    assert fixture.budget.queries == len(fixture.driver.calls) == len(requests)


def test_binary_completed_overflow_through_real_http_charges_actual_rows(http_gateway):
    fixture, requests = http_gateway
    fixture.session.reads.driver.binary_batches = True
    fixture.budget.charge_page(scanned_rows=0, wire_bytes=1048576 - 100)
    with pytest.raises(FrontierCompletedBudgetFailureV2):
        fixture.session.initialize()
    assert fixture.budget.queries == len(requests) == 1
    assert fixture.budget.source_rows == fixture.driver.scanned == 2
    assert not fixture.budget._uncertain
