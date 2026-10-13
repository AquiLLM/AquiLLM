"""Actual body-byte charging and original-deadline private transport."""

from email.message import Message
from io import BytesIO

import pytest

from apps.knowledge_graph.retrieval.topology.failures import TopologyLoadError
from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
    FrontierBudgetExceeded,
)
from apps.knowledge_graph.retrieval.topology.frontier_gateway_client_v2 import (
    FrontierGatewayClientV2,
)
from apps.knowledge_graph.retrieval.topology.frontier_gateway_contracts_v2 import (
    CHECKSUM,
    VERSION,
    FrontierGatewayRequestV2,
    decode_request,
    encode_request,
)
from apps.knowledge_graph.retrieval.topology.gateway_client import TopologyGatewayClient
from apps.knowledge_graph.retrieval.topology.gateway_contracts import _canonical
from apps.knowledge_graph.tests.frontier_fixture_support import FrontierFixture


class Response(BytesIO):
    status = 200

    def __init__(self, body, status=200, encoding=None):
        super().__init__(body)
        self.status = status
        self.headers = Message()
        for name, value in {
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
            "X-Topology-Schema-Version": VERSION,
            "X-Topology-Schema-Checksum": CHECKSUM,
        }.items():
            self.headers[name] = value
        if encoding:
            self.headers["Content-Encoding"] = encoding


def client():
    return FrontierGatewayClientV2(
        TopologyGatewayClient("http://localhost", "synthetic-token", 5.0)
    )


@pytest.mark.parametrize(
    "body,status,encoding",
    [
        (b"invalid JSON", 200, None),
        (b'{"reason":"unavailable"}', 503, None),
        (b"encoded bytes", 200, "gzip"),
    ],
)
def test_invalid_error_and_encoded_bodies_are_charged_before_rejection(
    body, status, encoding
):
    f = FrontierFixture()
    budget = f.budget()
    with pytest.raises((TopologyLoadError, ValueError)):
        client()._decode(
            Response(body, status, encoding),
            allowance=1048576,
            budget=budget,
            deadline=f.deadline,
        )
    assert budget.wire_bytes == len(body)


def test_wire_limit_stops_before_fetching_byte_limit_plus_one():
    f = FrontierFixture()
    budget = f.budget(max_wire_bytes=10)
    response = Response(b"x" * 11)
    with pytest.raises(FrontierBudgetExceeded):
        client()._decode(response, allowance=10, budget=budget, deadline=f.deadline)
    assert budget.wire_bytes == response.tell() == 10


def test_exact_success_body_charges_original_bytes_not_reserialization():
    f = FrontierFixture()
    body = _canonical({"rows": [], "scanned_rows": 0, "queries": 1})
    budget = f.budget(max_wire_bytes=len(body))
    assert (
        client()._decode(
            Response(body), allowance=len(body), budget=budget, deadline=f.deadline
        )
        == ()
    )
    assert budget.wire_bytes == len(body)


def test_frontier_wire_carries_no_absolute_deadline_and_rejects_unknown_generation():
    f = FrontierFixture()
    request = FrontierGatewayRequestV2(
        "manifest",
        f.bundles[0].generation.generation_key,
        (),
        f.group.scope.ready,
        1048576,
    )
    assert decode_request(encode_request(request)) == request
    assert b"deadline" not in encode_request(request)
    with pytest.raises(ValueError):
        FrontierGatewayRequestV2("raw", "f" * 64, ("a" * 64,), f.group.scope.ready, 100)


def test_transport_failure_poisoning_prevents_continuation(monkeypatch):
    from apps.knowledge_graph.retrieval.topology import (
        frontier_gateway_client_v2 as module,
    )

    class Opener:
        calls = 0

        def open(self, request, timeout):
            self.calls += 1
            assert request.get_header("Accept-encoding") == "identity"
            assert request.get_header("X-frontier-timeout-ms").isdigit()
            raise TimeoutError("synthetic abandoned server work")

    opener = Opener()
    monkeypatch.setattr(module, "build_opener", lambda *args: opener)
    f, transport = FrontierFixture(), client()
    budget = f.budget()
    args = dict(
        kind="manifest",
        generation=f.bundles[0].generation.generation_key,
        keys=(),
        ready=f.group.scope.ready,
        deadline=f.deadline,
        budget=budget,
    )
    with pytest.raises(TimeoutError):
        transport.execute_frontier_read(**args)
    with pytest.raises(FrontierBudgetExceeded):
        transport.execute_frontier_read(**args)
    assert opener.calls == 1
