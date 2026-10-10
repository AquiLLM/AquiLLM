"""Owned loopback transport for whole-runtime frontier/body integration."""

import asyncio
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from apps.knowledge_graph.retrieval.topology import gateway_service as service
from apps.knowledge_graph.retrieval.topology.gateway_client import TopologyGatewayClient
from apps.knowledge_graph.tests.test_topology_gateway_service import _settings


@contextmanager
def graph_http_transport(monkeypatch, driver):
    settings = replace(_settings(), timeout_ms=4500)
    runtime = service.TopologyGatewayRuntime(settings, driver, object())
    monkeypatch.setattr(service, "load_topology_gateway_settings", lambda _: settings)
    monkeypatch.setattr(service, "_get_runtime", lambda *_: runtime)
    observations = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            request = self.rfile.read(int(self.headers["Content-Length"]))
            scope = {
                "type": "http",
                "method": "POST",
                "path": self.path,
                "headers": [
                    (k.lower().encode(), v.encode()) for k, v in self.headers.items()
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
            observations.append((len(body), int(self.headers["X-Frontier-Timeout-Ms"])))
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield (
            TopologyGatewayClient(
                f"http://127.0.0.1:{server.server_port}", "private-gateway-token", 5.0
            ),
            observations,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
