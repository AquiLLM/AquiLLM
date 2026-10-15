"""ASGI socket seam; all finite client, service, worker and replay code is real."""

import asyncio
from dataclasses import replace
from email.message import Message
from io import BytesIO
from urllib.parse import urlsplit

from apps.knowledge_graph.retrieval.topology import frontier_gateway_client_v2 as client
from apps.knowledge_graph.retrieval.topology import gateway_service as service
from apps.knowledge_graph.retrieval.topology.gateway_client import TopologyGatewayClient
from apps.knowledge_graph.tests.test_topology_gateway_service import _call, _settings


class WindowOpener:
    def __init__(self, fixture, monkeypatch):
        self.responses, self.requests, self.mutate = [], [], None
        runtime = service.TopologyGatewayRuntime(
            replace(_settings(), timeout_ms=5000), fixture.driver, object()
        )
        monkeypatch.setattr(
            service, "load_topology_gateway_settings", lambda _: runtime.settings
        )
        monkeypatch.setattr(service, "_get_runtime", lambda *_: runtime)
        monkeypatch.setattr(client, "build_opener", lambda *_: self)
        fixture.session.reads.driver = client.FrontierGatewayClientV2(
            TopologyGatewayClient("http://localhost", "private-gateway-token", 5.0)
        )

    def open(self, request, timeout):
        self.requests.append(request)
        messages = asyncio.run(
            _call(
                path=urlsplit(request.full_url).path,
                body=request.data,
                headers=[
                    *(
                        (k.lower().encode(), v.encode())
                        for k, v in request.header_items()
                    ),
                    (b"content-length", str(len(request.data)).encode()),
                ],
            )
        )
        body = b"".join(m.get("body", b"") for m in messages[1:])
        if self.mutate is not None:
            body = self.mutate(body)
        response = BytesIO(body)
        response.status, response.headers = messages[0]["status"], Message()
        for name, value in messages[0]["headers"]:
            response.headers[name.decode()] = value.decode()
        self.responses.append(response)
        return response
