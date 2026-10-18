"""Scope versioning rejects unknown protocol headers before data authority."""

import pytest

from apps.knowledge_graph.retrieval.topology.frontier_window_owner_v2 import (
    run_preparation,
)
from apps.knowledge_graph.tests.frontier_scope_complete_support import scope_state
from apps.knowledge_graph.tests.frontier_window_test_support import WindowOpener


@pytest.mark.parametrize("side", ["request", "response"])
def test_unexpected_protocol_header_cannot_create_scope_authority(monkeypatch, side):
    fixture, state = scope_state(stage="scope_raw")
    transport = WindowOpener(fixture, monkeypatch, complete_scope=True)
    original = transport.open
    before = len(fixture.driver.calls)

    def open_response(request, timeout):
        if side == "request":
            request.add_header("X-Frontier-Family", "scope")
        response = original(request, timeout)
        if side == "response":
            response.headers["X-Frontier-Family"] = "scope"
        return response

    transport.open = open_response
    with pytest.raises(Exception):
        run_preparation(state)
    assert len(transport.requests) == 1 and fixture.budget._uncertain
    assert fixture.budget._window is not None and not any(state.fresh.values())
    assert len(fixture.driver.calls) == before + (side == "response")
    assert fixture.budget.wire_bytes > 0


def test_original_deadline_expires_inside_json_parsing_before_ready_validation():
    import json

    from apps.knowledge_graph.retrieval.topology import (
        frontier_scope_window_contracts_v3 as contracts,
    )
    from apps.knowledge_graph.retrieval.topology.gateway_contracts import _canonical
    from apps.knowledge_graph.tests.frontier_scope_complete_support import request_body

    fixture, state = scope_state()
    value = json.loads(request_body(fixture, state))
    value["ready"] = {"a": 1, "b": 2}
    calls = 0

    def check():
        nonlocal calls
        calls += 1
        if calls >= 3:
            raise TimeoutError("original request expired during parse")

    with pytest.raises(TimeoutError, match="during parse"):
        contracts.decode_request(_canonical(value), check)
