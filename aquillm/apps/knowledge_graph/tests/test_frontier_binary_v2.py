"""Lossless binary batches keep strict wire, work and final-authority accounting."""

import json
import zlib

import pytest

from apps.knowledge_graph.retrieval.topology import frontier_binary_codec_v2 as codec
from apps.knowledge_graph.retrieval.topology import frontier_gateway_client_v2 as client
from apps.knowledge_graph.retrieval.topology.failures import TopologyLoadError
from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
    FrontierBudgetExceeded,
)
from apps.knowledge_graph.retrieval.topology.frontier_gateway_codec_v2 import (
    decode_batch_response,
    encode_batch_response,
)
from apps.knowledge_graph.retrieval.topology.frontier_gateway_contracts_v2 import (
    BATCH_CHECKSUM,
    BATCH_VERSION,
    BINARY_PATH,
    FrontierCompletedBudgetFailureV2,
)
from apps.knowledge_graph.retrieval.topology.gateway_contracts import _canonical
from apps.knowledge_graph.tests.frontier_binary_test_support import (
    Response,
    decode,
    payload,
    transport,
)
from apps.knowledge_graph.tests.frontier_generation_batch_support import (
    GenerationFixture,
)
from apps.knowledge_graph.tests.test_frontier_generation_gateway_v2 import (
    http_gateway as http_gateway,
)


def test_binary_and_legacy_decode_identical_payload_with_conservative_final_hold():
    plain = payload()
    old = encode_batch_response(plain)
    binary = codec.encode_binary_response(plain)
    assert binary == b"KGFZ0001" + len(plain).to_bytes(4, "big") + zlib.compress(
        plain, 1
    )
    assert codec.decode_binary_response(binary) == decode_batch_response(
        json.loads(old)
    )
    assert len(binary) < len(old)
    fixture = GenerationFixture(2, 4)
    assert decode(fixture, Response(binary)) == tuple(json.loads(plain)["rows"])
    assert fixture.budget.wire_bytes == len(binary)


@pytest.mark.parametrize(
    "mutation",
    [
        "short",
        "magic",
        "size0",
        "sizebig",
        "sizemismatch",
        "truncated",
        "trailing",
        "concatenated",
        "bomb",
        "json",
        "duplicate_json",
        "queries",
        "bool_queries",
        "rows321",
        "count",
        "bool_count",
        "row_shape",
        "labels",
        "key",
    ],
)
def test_malformed_binary_is_charged_before_rejection(mutation):
    plain = payload()
    value = json.loads(plain)
    if mutation in {"queries", "bool_queries"}:
        value["queries"] = 2 if mutation == "queries" else True
    elif mutation == "rows321":
        value["rows"] *= 321
        value["scanned_rows"] = 321
    elif mutation in {"count", "bool_count"}:
        value["scanned_rows"] = 0 if mutation == "count" else True
    elif mutation == "row_shape":
        value["rows"][0]["extra"] = 1
    elif mutation == "labels":
        value["rows"][0]["labels"] = [1]
    elif mutation == "key":
        value["rows"][0]["lookup_key"] = "invalid"
    plain = _canonical(value)
    if mutation == "json":
        plain = b"not JSON"
    elif mutation == "duplicate_json":
        plain = b'{"rows":[],"queries":1,"queries":1,"scanned_rows":0}'
    body = codec.encode_binary_response(plain)
    if mutation == "short":
        body = body[:11]
    elif mutation == "magic":
        body = b"KGFZ0002" + body[8:]
    elif mutation in {"size0", "sizebig", "sizemismatch"}:
        size = {"size0": 0, "sizebig": 1048577, "sizemismatch": len(plain) + 1}[
            mutation
        ]
        body = body[:8] + size.to_bytes(4, "big") + body[12:]
    elif mutation == "truncated":
        body = body[:-1]
    elif mutation == "trailing":
        body += b"x"
    elif mutation == "concatenated":
        body += zlib.compress(plain)
    elif mutation == "bomb":
        body = (
            b"KGFZ0001" + (1048576).to_bytes(4, "big") + zlib.compress(b"x" * 1048577)
        )
    fixture = GenerationFixture(2, 4)
    with pytest.raises((ValueError, TopologyLoadError)):
        decode(fixture, Response(body))
    assert fixture.budget.wire_bytes == len(body)


@pytest.mark.parametrize(
    "mutation",
    [
        "type",
        "type_duplicate",
        "length",
        "length_duplicate",
        "version",
        "version_duplicate",
        "checksum",
        "checksum_duplicate",
        "encoding",
        "transfer",
        "json_success",
        "binary_on_legacy",
    ],
)
def test_binary_headers_are_exact_and_charged(mutation):
    body = codec.encode_binary_response(payload())
    if mutation == "json_success":
        body = payload()
    response = Response(body)
    names = {
        "type": "Content-Type",
        "length": "Content-Length",
        "version": "X-Topology-Schema-Version",
        "checksum": "X-Topology-Schema-Checksum",
    }
    base = mutation.removesuffix("_duplicate")
    if base in names:
        name = names[base]
        if mutation.endswith("duplicate"):
            response.headers[name] = response.headers[name]
        else:
            response.headers.replace_header(name, "invalid")
    elif mutation in {"encoding", "transfer"}:
        response.headers[
            "Content-Encoding" if mutation == "encoding" else "Transfer-Encoding"
        ] = "gzip"
    fixture = GenerationFixture(2, 4)
    with pytest.raises((ValueError, TopologyLoadError)):
        if mutation == "binary_on_legacy":
            response.headers.replace_header("Content-Type", "application/json")
            response.headers.replace_header("X-Topology-Schema-Version", BATCH_VERSION)
            response.headers.replace_header(
                "X-Topology-Schema-Checksum", BATCH_CHECKSUM
            )
            transport()._decode(
                response,
                allowance=1048576,
                budget=fixture.budget,
                deadline=fixture.budget.deadline,
                version=BATCH_VERSION,
                checksum=BATCH_CHECKSUM,
            )
        else:
            decode(fixture, response)
    assert fixture.budget.wire_bytes == len(body)


@pytest.mark.parametrize("invalid", [False, True])
def test_completed_receipts_charge_once_and_uncertain_failure_never_retries(
    monkeypatch, invalid
):
    fixture = GenerationFixture(2, 4)
    body = _canonical(
        {
            "reason": "frontier_budget",
            "queries": True if invalid else 1,
            "scanned_rows": 2,
        }
    )
    response, calls = Response(body, status=422), []

    class Opener:
        def open(self, request, timeout):
            calls.append(request.full_url)
            assert request.full_url.endswith(BINARY_PATH)
            return response

    monkeypatch.setattr(client, "build_opener", lambda *_: Opener())
    fixture.session.reads.driver = transport()
    with pytest.raises(
        TopologyLoadError if invalid else FrontierCompletedBudgetFailureV2
    ):
        fixture.session.initialize()
    assert response.closed and len(calls) == fixture.budget.queries == 1
    assert fixture.budget.wire_bytes == len(body)
    assert fixture.budget._uncertain is invalid
    assert fixture.budget.source_rows == (0 if invalid else 2)
    if invalid:
        with pytest.raises(FrontierBudgetExceeded):
            fixture.session.initialize()
        assert len(calls) == 1


def test_late_binary_decode_charges_wire_closes_response_and_poisons_transport(
    monkeypatch,
):
    from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
        FrontierBudgetV2,
    )
    from lib.retrieval.turn_budget import TurnBudget, TurnLimits

    now = [1.0]
    budget = FrontierBudgetV2(
        turn_budget=TurnBudget(TurnLimits(), clock=lambda: now[0]),
        deadline=10.0,
        clock=lambda: now[0],
        max_identities=200,
        max_edges=1000,
        max_source_rows=5000,
        max_queries=64,
        max_wire_bytes=1048576,
    )
    fixture = GenerationFixture(2, 4, budget=budget)
    response = Response(codec.encode_binary_response(payload()))
    original = codec.decode_binary_response

    def late(body):
        value = original(body)
        now[0] = 11.0
        return value

    class Opener:
        def open(self, request, timeout):
            return response

    monkeypatch.setattr(codec, "decode_binary_response", late)
    monkeypatch.setattr(client, "build_opener", lambda *_: Opener())
    fixture.session.reads.driver = transport()
    size = len(response.getvalue())
    with pytest.raises(TimeoutError):
        fixture.session.initialize()
    assert response.closed and fixture.budget.wire_bytes == size
    assert fixture.budget.queries == 1 and fixture.budget._uncertain
