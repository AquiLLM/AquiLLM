"""Only a predispatch oversized raw checkpoint may select the fixed single mode."""

import json

import pytest

from apps.knowledge_graph.retrieval.topology import (
    frontier_window_contracts_v2 as bounded,
)
from apps.knowledge_graph.retrieval.topology.frontier_window_owner_v2 import (
    run_preparation,
)
from apps.knowledge_graph.tests.frontier_scope_complete_support import (
    request_body,
    scope_state,
)
from apps.knowledge_graph.tests.frontier_window_test_support import WindowOpener


def test_oversized_raw_checkpoint_uses_one_128_pair_single_then_resumes(monkeypatch):
    fixture, state = scope_state(stage="scope_raw", generations=2, documents=200)
    transport = WindowOpener(fixture, monkeypatch, complete_scope=True)
    # Exercise the real encoder's oversized branch at a smaller test-only bound.
    window_size = len(request_body(fixture, state))
    single_size = len(request_body(fixture, state, single=True))
    assert single_size < window_size
    monkeypatch.setattr(bounded, "MAX_REQUEST_BYTES", single_size + 100)
    before = len(fixture.driver.calls)
    result = run_preparation(state)
    assert len(result) == 2
    bodies = [json.loads(request.data) for request in transport.requests]
    assert len(fixture.driver.calls) - before == len(bodies) == 2
    assert [len(body["operation"]["lookups"]) for body in bodies] == [128, 74]
    assert all(body["mode"] == "single" and "checkpoint" not in body for body in bodies)
    assert all(
        "/v3/topology/frontier/scope-window" in request.full_url
        for request in transport.requests
    )
    assert fixture.budget.queries == len(fixture.driver.calls)
    assert fixture.budget.source_rows == fixture.driver.scanned
    assert fixture.budget._window is None and not fixture.budget._uncertain
    fixture.session.final_attestation()


@pytest.mark.parametrize("stage", ["scope_directory", "scope_entries", "scope_raw"])
def test_oversized_index_or_single_body_fails_before_dispatch(monkeypatch, stage):
    fixture, state = scope_state(stage=stage)
    transport = WindowOpener(fixture, monkeypatch, complete_scope=True)
    monkeypatch.setattr(bounded, "MAX_REQUEST_BYTES", 1)
    before = fixture.budget.queries, fixture.budget.source_rows
    with pytest.raises(bounded.CheckpointTooLarge):
        run_preparation(state)
    assert not transport.requests
    assert (fixture.budget.queries, fixture.budget.source_rows) == before
    assert fixture.budget._window is None and not fixture.budget._uncertain


@pytest.mark.parametrize("failure", ["digest", "receipt", "transport"])
def test_single_unknown_work_retains_hold_without_downgrade(monkeypatch, failure):
    fixture, state = scope_state(stage="scope_raw")
    transport = WindowOpener(fixture, monkeypatch, complete_scope=True)
    monkeypatch.setattr(
        bounded,
        "MAX_REQUEST_BYTES",
        len(request_body(fixture, state, single=True)) + 10,
    )
    if failure == "transport":
        original = transport.open

        def fail(*args, **kwargs):
            response = original(*args, **kwargs)
            response.close()
            raise OSError("scope connection uncertainty")

        transport.open = fail
    elif failure == "receipt":
        from apps.knowledge_graph.projection.frontier_reads_v2 import QUERIES

        def saturate(query, parameters, rows):
            return (
                (rows * 320)[:320]
                if query == QUERIES["scope_raw_complete_batch"]
                else rows
            )

        fixture.driver.transform = saturate

        def bad_receipt(body):
            return body.replace(b'"queries":1', b'"queries":2')

        transport.mutate = bad_receipt
    else:
        transport.mutate = lambda body: body[:8] + b"z" + body[9:]
    with pytest.raises(Exception):
        run_preparation(state)
    assert len(transport.requests) == 1 and fixture.budget._uncertain
    assert fixture.budget._window is not None and not any(state.fresh.values())
