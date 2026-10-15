from email.message import Message
from io import BytesIO

from apps.knowledge_graph.retrieval.topology import frontier_gateway_client_v2 as client
from apps.knowledge_graph.retrieval.topology.frontier_gateway_contracts_v2 import (
    BINARY_CHECKSUM,
    BINARY_VERSION,
)
from apps.knowledge_graph.retrieval.topology.gateway_client import TopologyGatewayClient
from apps.knowledge_graph.retrieval.topology.gateway_contracts import _canonical


class Response(BytesIO):
    def __init__(self, body, *, status=200):
        super().__init__(body)
        self.status, self.headers = status, Message()
        for name, value in {
            "Content-Type": "application/octet-stream"
            if status == 200
            else "application/json",
            "Content-Length": str(len(body)),
            "X-Topology-Schema-Version": BINARY_VERSION,
            "X-Topology-Schema-Checksum": BINARY_CHECKSUM,
        }.items():
            self.headers[name] = value


def transport():
    return client.FrontierGatewayClientV2(
        TopologyGatewayClient("http://localhost", "synthetic", 5.0)
    )


def decode(fixture, response, **kwargs):
    return transport()._decode(
        response,
        allowance=1048576,
        budget=fixture.budget,
        deadline=fixture.budget.deadline,
        version=BINARY_VERSION,
        checksum=BINARY_CHECKSUM,
        **kwargs,
    )


def payload():
    return _canonical(
        {
            "rows": [
                {
                    "generation_key": "a" * 64,
                    "lookup_key": "b" * 64,
                    "record": {"nested": ["exact", 12]},
                }
            ],
            "queries": 1,
            "scanned_rows": 1,
        }
    )
