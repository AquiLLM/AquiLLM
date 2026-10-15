"""Scope capacity and uncertain replies preserve the original receipt ordering."""

import json
from copy import deepcopy

import pytest

from apps.knowledge_graph.projection.frontier_reads_v2 import QUERIES
from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
    FrontierBudgetExceeded,
)
from apps.knowledge_graph.retrieval.topology.frontier_window_contracts_v2 import (
    CompletedWindowBudgetFailure,
)
from apps.knowledge_graph.retrieval.topology.frontier_window_owner_v2 import (
    run_preparation,
)
from apps.knowledge_graph.tests.frontier_scope_complete_support import scope_state
from apps.knowledge_graph.tests.frontier_window_test_support import WindowOpener


def test_saturation_after_entry_frame_settles_every_row_and_publishes_nothing(
    monkeypatch,
):
    fixture, state = scope_state(stage="scope_entries")
    cache = deepcopy(state.reads._index_records)
    transport = WindowOpener(fixture, monkeypatch, complete_scope=True)

    def saturate(query, parameters, rows):
        return (
            (rows * 320)[:320] if query == QUERIES["scope_raw_complete_batch"] else rows
        )

    fixture.driver.transform = saturate
    before = len(fixture.driver.calls), fixture.driver.scanned
    with pytest.raises(CompletedWindowBudgetFailure) as caught:
        run_preparation(state)
    assert (caught.value.queries, caught.value.scanned_rows) == (2, 331)
    assert len(fixture.driver.calls) - before[0] == 2
    assert fixture.driver.scanned - before[1] == 331
    assert (fixture.budget.queries, fixture.budget.source_rows) == (
        len(fixture.driver.calls),
        fixture.driver.scanned,
    )
    assert (
        state.reads._index_records == cache and state.result is None and not state.fresh
    )
    assert fixture.budget._window is None and not fixture.budget._uncertain
    assert len(transport.requests) == 1
    assert fixture.budget.wire_bytes == int(
        transport.responses[0].headers["Content-Length"]
    )
    fixture.session.final_attestation()


@pytest.mark.parametrize(
    "mutation", ["digest", "later", "kind", "foreign", "receipt", "close", "late"]
)
def test_corrupt_or_unknown_scope_work_never_retries_or_publishes(
    monkeypatch, mutation
):
    fixture, state = scope_state(stage="scope_entries")
    transport = WindowOpener(fixture, monkeypatch, complete_scope=True)
    cache = deepcopy(state.reads._index_records)
    before = fixture.budget.queries, fixture.budget.source_rows

    def mutate(body):
        assert body.startswith(b"KGFS0001") and body[40] == 2
        if mutation == "digest":
            return body[:8] + bytes([body[8] ^ 1]) + body[9:]
        if mutation == "kind":
            return body[:41] + b"\x03" + body[42:]
        return body[:-1] + bytes([body[-1] ^ 1])

    if mutation in {"digest", "kind", "later"}:
        transport.mutate = mutate
    elif mutation == "foreign":

        def foreign(query, parameters, rows):
            if query == QUERIES["scope_raw_complete_batch"]:
                rows = deepcopy(rows)
                rows[0]["record"]["generation_key"] = "a" * 64
            return rows

        fixture.driver.transform = foreign
    elif mutation == "receipt":

        def saturate(query, parameters, rows):
            return (
                (rows * 320)[:320]
                if query == QUERIES["scope_raw_complete_batch"]
                else rows
            )

        fixture.driver.transform = saturate

        def wrong_receipt(body):
            value = json.loads(body)
            value["request_digest"] = "a" * 64
            return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

        transport.mutate = wrong_receipt
    else:
        original = transport.open

        def open_response(*args, **kwargs):
            response = original(*args, **kwargs)
            if mutation == "close":
                close = response.close

                def broken():
                    close()
                    raise OSError("scope cleanup uncertain")

                response.close = broken
            else:
                state.reads.deadline = fixture.budget.clock() - 1
            return response

        transport.open = open_response
    with pytest.raises(Exception):
        run_preparation(state)
    assert fixture.budget._uncertain and fixture.budget._window is not None
    assert (fixture.budget.queries, fixture.budget.source_rows) == before
    assert (
        state.reads._index_records == cache and not state.fresh and state.result is None
    )
    with pytest.raises(FrontierBudgetExceeded):
        run_preparation(state)
    assert len(transport.requests) == 1 and transport.responses[0].closed


@pytest.mark.parametrize("allowance", [100, 200])
def test_small_wire_exact_receipt_or_empty_unavailable(monkeypatch, allowance):
    fixture, state = scope_state(stage="scope_raw")
    transport = WindowOpener(fixture, monkeypatch, complete_scope=True)
    fixture.budget.charge_page(
        scanned_rows=0, wire_bytes=fixture.budget.available_wire_bytes() - allowance
    )
    before = fixture.budget.queries, fixture.budget.source_rows
    with pytest.raises(Exception) as caught:
        run_preparation(state)
    assert len(transport.requests) == 1 and not any(state.fresh.values())
    if allowance == 200:
        assert isinstance(caught.value, CompletedWindowBudgetFailure)
        assert (caught.value.queries, caught.value.scanned_rows) == (1, 21)
        assert fixture.budget._window is None and not fixture.budget._uncertain
        fixture.session.final_attestation()
    else:
        assert fixture.budget._uncertain and fixture.budget._window is not None
        assert (fixture.budget.queries, fixture.budget.source_rows) == before


@pytest.mark.parametrize("available", [319, 320])
def test_tail_raw_still_requires_full_320_before_dispatch(monkeypatch, available):
    fixture, state = scope_state(stage="scope_raw")
    transport = WindowOpener(fixture, monkeypatch, complete_scope=True)
    fixture.budget.charge_page(
        scanned_rows=fixture.budget.available_read_rows() - available, wire_bytes=0
    )
    if available == 319:
        with pytest.raises(FrontierBudgetExceeded):
            run_preparation(state)
        assert not transport.requests
    else:
        assert sum(map(len, run_preparation(state).values())) == 21
        assert len(transport.requests) == 1
    fixture.session.final_attestation()
