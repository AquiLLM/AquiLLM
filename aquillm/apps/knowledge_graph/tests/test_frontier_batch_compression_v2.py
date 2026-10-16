"""Lossless batch-only wire encoding has an independent expanded-size ceiling."""

import base64
import zlib
from email.message import Message
from io import BytesIO
from random import Random
from time import monotonic

import pytest

from apps.knowledge_graph.retrieval.topology.failures import TopologyLoadError
from apps.knowledge_graph.retrieval.topology.frontier_gateway_client_v2 import (
    FrontierGatewayClientV2,
)
from apps.knowledge_graph.retrieval.topology.frontier_gateway_contracts_v2 import (
    BATCH_CHECKSUM,
    BATCH_VERSION,
    FrontierCompletedBudgetFailureV2,
    FrontierGatewayBatchRequestV2,
)
from apps.knowledge_graph.retrieval.topology.frontier_gateway_service_v2 import (
    read_payload,
)
from apps.knowledge_graph.retrieval.topology.gateway_client import TopologyGatewayClient
from apps.knowledge_graph.retrieval.topology.gateway_contracts import _canonical
from apps.knowledge_graph.tests.frontier_generation_batch_support import (
    GenerationFixture,
)


class Response(BytesIO):
    status = 200

    def __init__(self, body):
        super().__init__(body)
        self.headers = Message()
        for name, value in {
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
            "X-Topology-Schema-Version": BATCH_VERSION,
            "X-Topology-Schema-Checksum": BATCH_CHECKSUM,
        }.items():
            self.headers[name] = value


def envelope(raw):
    return _canonical(
        {"encoding": "zlib-base64", "payload": base64.b64encode(raw).decode("ascii")}
    )


def decode(fixture, body):
    client = FrontierGatewayClientV2(
        TopologyGatewayClient("http://localhost", "synthetic", 5.0)
    )
    return client._decode(
        Response(body),
        allowance=1048576,
        budget=fixture.budget,
        deadline=fixture.budget.deadline,
        version=BATCH_VERSION,
        checksum=BATCH_CHECKSUM,
    )


def test_valid_compressed_response_returns_identical_rows_and_charges_envelope_only():
    fixture = GenerationFixture(2, 4)
    rows = [
        {
            "generation_key": "a" * 64,
            "lookup_key": "a" * 64,
            "record": {"test": "x" * 3000},
        }
    ]
    raw = _canonical({"rows": rows, "scanned_rows": 1, "queries": 1})
    body = envelope(zlib.compress(raw, 1))
    assert decode(fixture, body) == tuple(rows)
    assert fixture.budget.wire_bytes == len(body) < len(raw)


@pytest.mark.parametrize(
    "mutation",
    [
        "bomb",
        "truncated",
        "trailing",
        "multiple",
        "duplicate_json",
        "wrong_count",
        "bool_count",
        "321",
        "wrong_query",
        "base64",
        "identity",
    ],
)
def test_malformed_compression_is_charged_and_rejected(mutation):
    fixture = GenerationFixture(2, 4)
    value = {"rows": [], "scanned_rows": 0, "queries": 1}
    if mutation == "wrong_count":
        value["scanned_rows"] = 1
    elif mutation == "bool_count":
        value["scanned_rows"] = False
    elif mutation == "321":
        value.update(rows=[{}] * 321, scanned_rows=321)
    elif mutation == "wrong_query":
        value["queries"] = 2
    raw = _canonical(value)
    if mutation == "bomb":
        raw = b"x" * 1048577
    elif mutation == "duplicate_json":
        raw = b'{"queries":1,"queries":1,"rows":[],"scanned_rows":0}'
    compressed = zlib.compress(raw, 1)
    if mutation == "truncated":
        compressed = compressed[:-1]
    elif mutation == "trailing":
        compressed += b"x"
    elif mutation == "multiple":
        compressed += zlib.compress(raw, 1)
    body = envelope(compressed)
    if mutation == "base64":
        body = _canonical({"encoding": "zlib-base64", "payload": "invalid!!"})
    elif mutation == "identity":
        body = raw
    with pytest.raises((ValueError, TopologyLoadError)):
        decode(fixture, body)
    assert fixture.budget.wire_bytes == len(body)


def test_incompressible_response_refuses_with_exact_completed_row_charge():
    fixture = GenerationFixture(2, 4)
    generation = fixture.ready.selected_generations[0].generation_key
    # Random printable material exceeds the tiny allowance after compression.
    value = Random(17).randbytes(5000).hex()
    fixture.driver.transform = lambda q, p, rows: (
        {**rows[0], "record": {"random": value}},
    )
    request = FrontierGatewayBatchRequestV2(
        "manifest_batch", ((generation, generation),), fixture.ready, 256
    )
    with pytest.raises(FrontierCompletedBudgetFailureV2) as caught:
        read_payload(fixture.driver, request, monotonic() + 5, 1048576)
    assert caught.value.scanned_rows == fixture.driver.scanned == 1
