"""Gateway HTTP exchanges must not initialize unused TLS trust stores."""

import http.client
import json
import ssl
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from time import monotonic
from urllib.request import HTTPSHandler

import pytest

from apps.knowledge_graph.retrieval.topology import frontier_gateway_client_v2 as frontier
from apps.knowledge_graph.retrieval.topology import frontier_gateway_contracts_v2 as fv2
from apps.knowledge_graph.retrieval.topology import gateway_client as gateway
from apps.knowledge_graph.retrieval.topology import gateway_contracts as v1
from apps.knowledge_graph.retrieval.topology import gateway_snapshot_contracts as v2
from apps.knowledge_graph.retrieval.topology.contracts import TopologyQueryName
from apps.knowledge_graph.tests.frontier_fixture_support import FrontierFixture


MANIFEST = {name: "a" * 64 for name in (
    "collection_key", "generation_key", "projection_key", "active_artifact_key",
    "graph_checksum", "membership_checksum",
)}


def _response(protocol):
    if protocol == "frontier":
        return fv2.PATH, fv2.VERSION, fv2.CHECKSUM, {
            "queries": 1, "rows": [{"id": 7}], "scanned_rows": 1,
        }
    if protocol == "snapshot":
        return v2.SNAPSHOT_PATH, v2.SCHEMA_VERSION, v2.SCHEMA_CHECKSUM, {
            "delivery": "snapshot", "manifests": [MANIFEST], "ok": True,
            "snapshot_json": "{}",
        }
    return "/v1/topology/read", v1.SCHEMA_VERSION, v1.SCHEMA_CHECKSUM, {
        "ok": True, "rows": [{"id": 7}],
    }


@pytest.fixture
def tls_contexts(monkeypatch):
    contexts = []
    original = ssl._create_default_https_context

    def observe(*args, **kwargs):
        context = original(*args, **kwargs)
        contexts.append(context)
        return context

    monkeypatch.setattr(ssl, "_create_default_https_context", observe)
    return contexts


@pytest.fixture
def gateway_server(protocol):
    path, version, checksum, value = _response(protocol)
    body = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    received = []

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(2.0)

        def do_POST(self):
            received.append((self.path, self.headers, self.rfile.read(
                int(self.headers["Content-Length"]))))
            self.send_response(200 if self.path == path else 400)
            for name, value in {
                "Content-Type": "application/json", "Content-Length": str(len(body)),
                "X-Topology-Schema-Version": version,
                "X-Topology-Schema-Checksum": checksum,
            }.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", received, len(body)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2.0)
        assert not thread.is_alive(), "owned gateway server did not stop"


def _exchange(protocol, origin):
    client = gateway.TopologyGatewayClient(
        origin, "synthetic-gateway-token", 5.0, snapshot_enabled=protocol == "snapshot",
    )
    deadline = monotonic() + 5.0
    if protocol == "frontier":
        fixture = FrontierFixture()
        budget = fixture.budget()
        result = frontier.FrontierGatewayClientV2(client).execute_frontier_read(
            kind="manifest", generation=fixture.bundles[0].generation.generation_key,
            keys=(), ready=fixture.group.scope.ready, deadline=deadline, budget=budget,
        )
        return result, budget
    if protocol == "snapshot":
        return client.execute_snapshot(parameters={"scope": "opaque"}, deadline=deadline), None
    return client.execute_read(
        query=TopologyQueryName.GENERATION_MANIFESTS,
        parameters={"scope": "opaque"}, deadline=deadline, max_records=1,
    ), None


@pytest.mark.parametrize("protocol", ["frontier", "legacy", "snapshot"])
def test_real_http_exchange_never_constructs_unused_tls_context(
    monkeypatch, protocol, gateway_server, tls_contexts,
):
    origin, received, wire_bytes = gateway_server
    # A proxy regression reaches the same owned server with an absolute URL and fails.
    for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY"):
        monkeypatch.setenv(name, origin)
    for name in ("no_proxy", "NO_PROXY"):
        monkeypatch.setenv(name, "")
    responses = []
    original = http.client.HTTPConnection.getresponse

    def observe_response(connection):
        response = original(connection)
        responses.append(response)
        return response

    monkeypatch.setattr(http.client.HTTPConnection, "getresponse", observe_response)
    result, budget = _exchange(protocol, origin)
    if protocol == "snapshot":
        assert type(result) is v2.TopologyGatewaySnapshotV2
        assert result.manifests == (MANIFEST,) and result.snapshot_json == "{}"
    else:
        assert result == ({"id": 7},)
    assert len(responses) == 1 and responses[0].closed
    assert len(received) == 1
    path, headers, body = received[0]
    expected_path, version, checksum, _ = _response(protocol)
    assert path == expected_path
    assert headers["Authorization"] == "Bearer synthetic-gateway-token"
    assert headers["X-Topology-Schema-Version"] == version
    assert headers["X-Topology-Schema-Checksum"] == checksum
    request = json.loads(body)
    if protocol == "frontier":
        assert request["kind"] == "manifest" and request["keys"] == []
        assert headers["Accept-Encoding"] == "identity"
        assert 0 < int(headers["X-Frontier-Timeout-Ms"]) <= 5000
        assert budget.wire_bytes == wire_bytes
    else:
        assert request["parameters"] == {"scope": "opaque"}
    # Count only after the real exchange, decoding, accounting, and cleanup succeed.
    assert tls_contexts == [], "HTTP initialized a TLS context it cannot use"


@pytest.mark.parametrize("protocol", ["frontier", "legacy"])
def test_https_opener_retains_default_certificate_and_hostname_verification(
    monkeypatch, protocol, tls_contexts,
):
    module = frontier if protocol == "frontier" else gateway
    original = module.build_opener
    openers = []

    class Constructed(Exception):
        pass

    def observe_opener(*args, **kwargs):
        openers.append(original(*args, **kwargs))
        raise Constructed  # Observe actual construction; never dispatch HTTPS.

    monkeypatch.setattr(module, "build_opener", observe_opener)
    with pytest.raises(Constructed):
        _exchange(protocol, "https://localhost")
    handlers = [h for h in openers[0].handlers if isinstance(h, HTTPSHandler)]
    assert len(handlers) == 1 and len(tls_contexts) == 1
    context = handlers[0]._context
    assert context is tls_contexts[0]
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True
