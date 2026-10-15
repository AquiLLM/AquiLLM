"""Actual competing-head acquisition and original gateway wire feasibility."""

import asyncio
import json
from dataclasses import replace
from email.message import Message
from io import BytesIO
from urllib.parse import urlsplit

import pytest

from apps.knowledge_graph.retrieval.topology import frontier_gateway_client_v2 as client
from apps.knowledge_graph.retrieval.topology import frontier_gateway_codec_v2 as codec
from apps.knowledge_graph.retrieval.topology import gateway_service as service
from apps.knowledge_graph.retrieval.topology.contracts import ProjectedSeedV1
from apps.knowledge_graph.retrieval.topology.frontier_acquisition_v2 import (
    acquire_frontier,
)
from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
    FrontierBudgetExceeded,
)
from apps.knowledge_graph.retrieval.topology.gateway_client import TopologyGatewayClient
from apps.knowledge_graph.tests.frontier_scope_window_support import MixedScopeFixture
from apps.knowledge_graph.tests.test_topology_gateway_service import _call, _settings


class GatewayOpener:
    """Replace only the socket; execute actual ASGI routes and production codecs."""

    def __init__(self, fixture, monkeypatch):
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
        self.frames = []
        self.encoded = []
        self.fixture = fixture
        encode = codec.encode_batch_response

        def measured_encode(payload):
            wire = encode(payload)
            self.encoded.append((len(payload), len(wire)))
            return wire

        monkeypatch.setattr(codec, "encode_batch_response", measured_encode)

    def open(self, request, timeout):
        before = len(self.encoded)
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
        response = BytesIO(body)
        response.status = messages[0]["status"]
        response.headers = Message()
        for name, value in messages[0]["headers"]:
            response.headers[name.decode()] = value.decode()
        self.frames.append(
            {
                "path": urlsplit(request.full_url).path,
                "phase": self.fixture.driver.phase,
                "status": response.status,
                "wire": len(body),
                "allowance": json.loads(request.data)["response_bytes"],
                "encoded": (
                    [(int.from_bytes(body[8:12], "big"), len(body))]
                    if body.startswith(b"KGFZ0001")
                    else self.encoded[before:]
                ),
                "rows": self.fixture.driver.transcript[-1]["rows"],
                "kind": self.fixture.driver.transcript[-1]["kind"],
            }
        )
        return response


@pytest.mark.parametrize("count", [1, 16])
@pytest.mark.parametrize("gateway", [False, True], ids=["physical", "gateway"])
def test_actual_competing_heads_yield_evidence_and_final_authority(
    count, gateway, monkeypatch
):
    fixture = MixedScopeFixture()
    transport = GatewayOpener(fixture, monkeypatch) if gateway else None
    seeds = tuple(
        ProjectedSeedV1(bundle.relations[0].source_entity_key, 1.0 / count)
        for bundle in sorted(
            fixture.bundles, key=lambda b: b.generation.generation_key
        )[:count]
    )
    fixture.session.initialize()
    initialized = (
        fixture.budget.queries,
        fixture.budget.source_rows,
        fixture.budget.wire_bytes,
    )
    fixture.driver.phase = "acquisition"
    failure = None
    try:
        acquire_frontier(
            fixture.session, fixture.assembly, seeds, fixture.caps, fixture.budget
        )
    except FrontierBudgetExceeded as error:
        failure = type(error).__name__
    finally:
        fixture.driver.phase = "final"
        fixture.session.final_attestation()
    print(
        {
            "seeds": count,
            "gateway": gateway,
            "initialized": initialized,
            "queries": fixture.budget.queries,
            "rows": fixture.budget.source_rows,
            "wire": fixture.budget.wire_bytes,
            "edges": fixture.budget.edges,
            "candidates": len(fixture.assembly.candidates),
            "stop": failure,
            "frames": transport.frames if transport else None,
        }
    )
    assert fixture.assembly.candidates and fixture.budget.edges
    assert fixture.budget.queries == len(fixture.driver.calls) <= 64
    assert fixture.budget.source_rows == fixture.driver.scanned <= 5000
    assert fixture.budget.wire_bytes <= 1048576
    assert not fixture.budget._authority_reservations
    assert not fixture.budget._uncertain
    assert fixture.driver.transcript[-1]["phase"] == "final"
    assert all(row["maximum"] <= 320 for row in fixture.driver.transcript)
    if transport:
        assert fixture.budget.wire_bytes == sum(
            frame["wire"] for frame in transport.frames
        )
        assert {
            f["path"]
            for f in transport.frames
            if f["kind"].endswith("batch") or f["kind"].endswith("multi")
        } == {"/v2/topology/frontier/batch-binary"}
        assert fixture.budget.wire_bytes == (517503 if count == 1 else 927493)
        if count == 16:
            assert (
                fixture.budget.queries,
                fixture.budget.source_rows,
                fixture.budget.edges,
                len(fixture.assembly.candidates),
            ) == (64, 4660, 3, 3)
