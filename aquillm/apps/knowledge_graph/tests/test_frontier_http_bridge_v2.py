"""Owned loopback HTTP proves raw body metering across the entire bridge."""

import asyncio
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from apps.knowledge_graph.retrieval.topology import gateway_service as service
from apps.knowledge_graph.retrieval.topology.factory import (
    create_canonical_frontier_loader_v2,
)
from apps.knowledge_graph.retrieval.topology.frontier_loader_v2 import (
    FrontierUnavailableV2,
)
from apps.knowledge_graph.retrieval.topology.gateway_client import TopologyGatewayClient
from apps.knowledge_graph.tests.frontier_fixture_support import FrontierFixture
from apps.knowledge_graph.tests.frontier_index_fixture_driver import (
    FrontierIndexFixtureDriver,
)
from apps.knowledge_graph.tests.test_topology_gateway_service import _settings


@pytest.mark.parametrize("wire_cap", [1048576, 256])
def test_real_http_bridge_counts_actual_raw_bodies_and_preserves_original_deadline(
    monkeypatch,
    wire_cap,
):
    fixture = FrontierFixture()
    driver = FrontierIndexFixtureDriver(fixture.bundles)
    settings = replace(_settings(), timeout_ms=1000)
    runtime = service.TopologyGatewayRuntime(settings, driver, object())
    monkeypatch.setattr(service, "load_topology_gateway_settings", lambda _: settings)
    monkeypatch.setattr(service, "_get_runtime", lambda *_: runtime)
    body_sizes, timeouts, physical_counts = [], [], []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            timeouts.append(int(self.headers["X-Frontier-Timeout-Ms"]))
            request = self.rfile.read(int(self.headers["Content-Length"]))
            scope = {
                "type": "http",
                "method": "POST",
                "path": self.path,
                "headers": [
                    (name.lower().encode(), value.encode())
                    for name, value in self.headers.items()
                ],
            }

            async def receive():
                return {"type": "http.request", "body": request, "more_body": False}

            sent = []

            async def send(message):
                sent.append(message)

            asyncio.run(service.app(scope, receive, send))
            self.send_response(sent[0]["status"])
            for name, value in sent[0]["headers"]:
                self.send_header(name.decode(), value.decode())
            self.end_headers()
            body = sent[1]["body"]
            body_sizes.append(len(body))
            physical_counts.append(body[40] if body.startswith(b"KGFW0001") else 1)
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        gateway = TopologyGatewayClient(
            f"http://127.0.0.1:{server.server_port}", "private-gateway-token", 5.0
        )
        loader = create_canonical_frontier_loader_v2(
            selection=fixture.selection, codec=fixture.codec, gateway_client=gateway
        )
        budget = fixture.budget(max_wire_bytes=wire_cap)
        deadline = budget.deadline
        if wire_cap == 256:
            with pytest.raises(FrontierUnavailableV2) as caught:
                loader.load(
                    group=fixture.group,
                    seeds=fixture.seeds,
                    request=fixture.request,
                    caps=fixture.caps,
                    budget=budget,
                    deadline=deadline,
                )
            assert caught.value.coverage.reasons == ("frontier_budget",)
            assert caught.value.reserved_source_rows == 0
            assert caught.value.work_uncertain is False
            assert budget.source_rows == 2
            assert budget.queries == len(driver.calls) == 1
            assert budget.wire_bytes == sum(body_sizes) <= wire_cap
            print(
                "FRONTIER_HTTP_COMPLETED_BUDGET",
                budget.queries,
                budget.source_rows,
                budget.wire_bytes,
            )
            return
        result = loader.load(
            group=fixture.group,
            seeds=fixture.seeds,
            request=fixture.request,
            caps=fixture.caps,
            budget=budget,
            deadline=deadline,
        )
        assert fixture.answer_identity in result.snapshot.identity_keys
        assert budget.wire_bytes == sum(body_sizes) > 0
        assert budget.queries == sum(physical_counts) == len(driver.calls) <= 64
        assert len(body_sizes) < budget.queries
        assert budget.wire_bytes <= 1048576 and budget.deadline == deadline
        assert all(0 < value <= 5000 for value in timeouts)
        print(
            "FRONTIER_HTTP_BRIDGE",
            budget.queries,
            budget.source_rows,
            budget.wire_bytes,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
