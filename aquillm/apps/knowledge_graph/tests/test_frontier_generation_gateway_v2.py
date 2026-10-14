"""Real HTTP client/ASGI boundary with physical fixed-read instrumentation."""

import asyncio
import json
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from time import monotonic

import pytest

from apps.knowledge_graph.retrieval.topology import (
    frontier_gateway_client_v2 as client_module,
)
from apps.knowledge_graph.retrieval.topology import gateway_service as service
from apps.knowledge_graph.retrieval.topology.frontier_gateway_contracts_v2 import (
    BATCH_CHECKSUM,
    BATCH_PATH,
    CHECKSUM,
    FrontierCompletedBudgetFailureV2,
    FrontierGatewayBatchRequestV2,
    decode_batch_request,
    encode_request,
)
from apps.knowledge_graph.retrieval.topology.frontier_gateway_service_v2 import (
    read_payload,
)
from apps.knowledge_graph.retrieval.topology.gateway_client import TopologyGatewayClient
from apps.knowledge_graph.tests.frontier_generation_batch_support import (
    GenerationFixture,
)
from apps.knowledge_graph.tests.test_topology_gateway_service import _call, _settings


@pytest.fixture
def http_gateway(monkeypatch, request):
    fixture = GenerationFixture(*getattr(request, "param", (2, 4)))
    runtime = service.TopologyGatewayRuntime(
        replace(_settings(), timeout_ms=5000), fixture.driver, object()
    )
    monkeypatch.setattr(
        service, "load_topology_gateway_settings", lambda _: runtime.settings
    )
    monkeypatch.setattr(service, "_get_runtime", lambda *_: runtime)
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            received.append((self.path, dict(self.headers), body))
            messages = asyncio.run(
                _call(
                    path=self.path,
                    body=body,
                    headers=[
                        (k.lower().encode(), v.encode())
                        for k, v in self.headers.items()
                    ],
                )
            )
            self.send_response(messages[0]["status"])
            for k, v in messages[0]["headers"]:
                self.send_header(k.decode(), v.decode())
            self.end_headers()
            self.wfile.write(b"".join(m.get("body", b"") for m in messages[1:]))

        def log_message(self, *_):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(
        target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
    )
    thread.start()
    transport = client_module.FrontierGatewayClientV2(
        TopologyGatewayClient(
            f"http://127.0.0.1:{server.server_port}", "private-gateway-token", 5.0
        )
    )
    fixture.session.reads.driver = transport
    try:
        yield fixture, received
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_real_http_batch_initialization_finalization_and_old_route_compatibility(
    http_gateway,
):
    fixture, received = http_gateway
    fixture.session.initialize()
    fixture.session.final_attestation()
    assert len(fixture.session.scope_rows) == 2
    assert fixture.budget.queries == len(fixture.driver.calls) == len(received)
    assert 0 < fixture.budget.wire_bytes < 1048576
    assert all(path == BATCH_PATH for path, _, _ in received)
    assert all(
        headers["X-Topology-Schema-Checksum"] == BATCH_CHECKSUM
        for _, headers, _ in received
    )
    before = fixture.budget.queries
    fixture.session.reads.attest(fixture.ready.selected_generations[0])
    assert fixture.budget.queries == before + 1
    assert received[-1][1]["X-Topology-Schema-Checksum"] == CHECKSUM


@pytest.mark.parametrize(
    "kind,limit",
    [("manifest_batch", 2), ("index_batch_multi", 160), ("raw_batch_multi", 32)],
)
def test_exact_batch_contract_and_single_statement_service(kind, limit):
    fixture = GenerationFixture(2, 100)
    if kind == "manifest_batch":
        pairs = tuple(
            sorted(
                (s.generation_key, s.generation_key)
                for s in fixture.ready.selected_generations
            )
        )
    elif kind == "index_batch_multi":
        pairs = tuple(
            sorted((v["generation_key"], k) for k, v in fixture.driver.index.items())
        )[:limit]
    else:
        pairs = tuple(sorted(fixture.driver.raw))[:limit]
    request = FrontierGatewayBatchRequestV2(kind, pairs, fixture.ready, 1048576)
    assert decode_batch_request(encode_request(request)) == request
    from apps.knowledge_graph.retrieval.topology.frontier_gateway_codec_v2 import (
        decode_batch_response,
    )

    result = decode_batch_response(
        json.loads(read_payload(fixture.driver, request, monotonic() + 5, 1048576))
    )
    assert result["queries"] == 1
    assert result["scanned_rows"] == len(result["rows"])
    assert len(fixture.driver.calls) == 1
    assert set(fixture.driver.calls[0][1]) == {"lookups_csv"}
    assert fixture.driver.bounds == [
        len(pairs) * (10 if kind == "raw_batch_multi" else 2)
    ]


def test_completed_batch_overflow_charges_rows_without_uncertainty(http_gateway):
    fixture, received = http_gateway
    # A complete response cannot fit; gateway still reports actual physical rows.
    fixture.budget.charge_page(scanned_rows=0, wire_bytes=1048576 - 100)
    with pytest.raises(FrontierCompletedBudgetFailureV2):
        fixture.session.initialize()
    assert fixture.budget.queries == len(received) == 1
    assert fixture.budget.source_rows == fixture.driver.scanned == 2
    assert not fixture.budget._uncertain
    assert fixture.session.scope_rows == {}


def test_failed_batch_transport_poisoning_forbids_retry(monkeypatch):
    fixture = GenerationFixture(2, 4)
    transport = client_module.FrontierGatewayClientV2(
        TopologyGatewayClient("http://localhost", "test-token", 5.0)
    )
    fixture.session.reads.driver = transport

    class Broken:
        calls = 0

        def open(self, request, timeout):
            self.calls += 1
            raise TimeoutError("uncertain server work")

    broken = Broken()
    monkeypatch.setattr(client_module, "build_opener", lambda *_: broken)
    for _ in range(2):
        with pytest.raises(TimeoutError):
            fixture.session.initialize()
    assert fixture.budget._uncertain and broken.calls == 1


@pytest.mark.parametrize("http_gateway", [(31, 530)], indirect=True)
def test_large_full_scope_compressed_wire_fits_original_cumulative_limit(http_gateway):
    fixture, received = http_gateway
    fixture.session.initialize()
    fixture.session.final_attestation()
    assert len(fixture.session.scope_rows) == 31
    assert (
        sum(
            label == "ProjectedDocument"
            for rows in fixture.session.scope_rows.values()
            for label, _ in rows
        )
        == 530
    )
    assert fixture.budget.queries == len(fixture.driver.calls) == len(received)
    assert fixture.budget.source_rows == fixture.driver.scanned
    assert fixture.budget.wire_bytes <= 1048576
    assert not fixture.budget._uncertain


def test_exploration_cannot_consume_fresh_final_manifest_wire_allowance(http_gateway):
    fixture, received = http_gateway
    fixture.session.initialize()
    available = getattr(fixture.budget, "available_wire_bytes", None)
    assert callable(available), "fresh final wire reservation is missing"
    fixture.budget.charge_page(scanned_rows=0, wire_bytes=available())
    assert fixture.budget.wire_bytes < fixture.budget.max_wire_bytes
    before = len(received)
    with pytest.raises(TimeoutError):
        fixture.session.reads.read(
            "index", fixture.ready.selected_generations[0].generation_key, ("a" * 64,)
        )
    assert not fixture.budget._uncertain
    assert len(received) == before
    fixture.session.final_attestation()
    assert len(received) == before + 1
    assert fixture.budget.wire_bytes == fixture.budget.max_wire_bytes


@pytest.mark.parametrize(
    "bad",
    [
        "empty",
        "duplicate",
        "unsorted",
        "unknown_generation",
        "manifest_key",
        "161",
        "raw33",
    ],
)
def test_batch_contract_rejects_invalid_pairs_before_dispatch(bad):
    from apps.knowledge_graph.tests.frontier_generation_batch_support import key

    fixture = GenerationFixture(2, 4)
    generation = fixture.ready.selected_generations[0].generation_key
    kind = "index_batch_multi"
    pairs = tuple(sorted((generation, key(str(i))) for i in range(2)))
    if bad == "empty":
        pairs = ()
    elif bad == "duplicate":
        pairs = pairs[:1] * 2
    elif bad == "unsorted":
        pairs = tuple(reversed(pairs))
    elif bad == "unknown_generation":
        pairs = (("f" * 64, pairs[0][1]),)
    elif bad == "manifest_key":
        kind = "manifest_batch"
    else:
        kind = "raw_batch_multi" if bad == "raw33" else kind
        pairs = tuple(
            sorted(
                (generation, key(str(i))) for i in range(33 if bad == "raw33" else 161)
            )
        )
    with pytest.raises(ValueError):
        FrontierGatewayBatchRequestV2(kind, pairs, fixture.ready, 1048576)
    assert fixture.driver.calls == []
