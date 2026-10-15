"""Completed aggregate overflow receipts account for every executed query."""

from hashlib import sha256
from time import monotonic

import pytest

from apps.knowledge_graph.projection.frontier_generation_batch_v2 import attest_all
from apps.knowledge_graph.projection.frontier_reads_v2 import QUERIES
from apps.knowledge_graph.projection.frontier_scope_preparation_v2 import (
    ScopePreparation,
)
from apps.knowledge_graph.retrieval.topology.frontier_binary_codec_v2 import (
    encode_binary_response,
)
from apps.knowledge_graph.retrieval.topology.frontier_window_checkpoint_v2 import (
    snapshot,
)
from apps.knowledge_graph.retrieval.topology.frontier_window_codec_v2 import (
    decode_container,
)
from apps.knowledge_graph.retrieval.topology.frontier_window_contracts_v2 import (
    CompletedWindowBudgetFailure,
    decode_request,
    encode_request,
)
from apps.knowledge_graph.retrieval.topology.frontier_window_worker_v2 import (
    read_window,
)
from apps.knowledge_graph.retrieval.topology.gateway_contracts import _canonical
from apps.knowledge_graph.tests.frontier_generation_batch_support import (
    GenerationFixture,
)


def test_second_raw_query_aggregate_overflow_receipt_includes_both_reads():
    fixture = GenerationFixture(2, 100)
    reads = fixture.session.reads
    reads.reserve_final(multi=True)
    attest_all(reads, fixture.session.selected)
    sources = {
        g: [
            s.collection_key,
            *(
                d.document_key
                for d in fixture.ready.authorized_documents
                if d.generation_key == g
            ),
        ]
        for g, s in fixture.session.selected.items()
    }
    state = ScopePreparation(reads, sources)
    while (effect := state.next_read()).kind != "raw_batch_multi":
        state.apply_result(reads.read_batch(effect.kind, effect.pairs))
    operation, checkpoint = snapshot(state)
    grant = fixture.budget.reserve_window()
    body = encode_request(
        fixture.ready, operation, checkpoint, grant, 1, fixture.budget.check_active
    )
    count = 0

    def duplicate_second_raw(query, parameters, rows):
        nonlocal count
        if query == QUERIES["raw_batch_multi"]:
            count += 1
            if count == 2:
                return (rows * 320)[:320]
        return rows

    fixture.driver.transform = duplicate_second_raw
    before = len(fixture.driver.calls), fixture.driver.scanned
    with pytest.raises(CompletedWindowBudgetFailure) as caught:
        read_window(fixture.driver, decode_request(body), monotonic() + 5, 1048576)
    receipt = caught.value
    assert receipt.request_digest == sha256(body).hexdigest()
    assert receipt.queries == len(fixture.driver.calls) - before[0] == 2
    assert receipt.scanned_rows == fixture.driver.scanned - before[1] > 320
    fixture.budget.settle_window(
        grant, queries=receipt.queries, scanned_rows=receipt.scanned_rows
    )
    assert fixture.budget.queries == len(fixture.driver.calls)
    assert fixture.budget.source_rows == fixture.driver.scanned
    assert not fixture.budget._uncertain


@pytest.mark.parametrize("mutation", ["expanded", "wire", "count"])
def test_container_rejects_aggregate_bounds_even_when_each_frame_is_valid(mutation):
    payload = _canonical(
        {
            "rows": [
                {
                    "generation_key": "a" * 64,
                    "lookup_key": "b" * 64,
                    "record": {"padding": "x" * 550000},
                }
            ],
            "queries": 1,
            "scanned_rows": 1,
        }
    )
    frame = encode_binary_response(payload)
    count = 9 if mutation == "count" else 2
    body = (
        b"KGFW0001"
        + bytes.fromhex("a" * 64)
        + bytes([count])
        + (b"\x01" + len(frame).to_bytes(4, "big") + frame) * count
    )
    if mutation == "wire":
        body += b"x" * 1048576
    with pytest.raises(ValueError):
        decode_container(body, "a" * 64)
