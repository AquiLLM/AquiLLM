"""Wire capacity must fit every bounded provider-neutral request scope."""

from dataclasses import replace

import pytest

from apps.knowledge_graph.projection.topology_request import decode_topology_request
from apps.knowledge_graph.retrieval.topology.contracts import (
    AuthorizedProjectedDocumentV1,
    HybridBranchKind,
    ProjectedSeedV1,
    ReadyGenerationBundleV1,
    TopologyQueryName,
    ready_generation_bundle_checksum,
)
from apps.knowledge_graph.retrieval.topology.gateway_contracts import (
    MAX_REQUEST_BYTES,
    TopologyGatewayRequestV1,
    decode_request,
    encode_request,
)
from apps.knowledge_graph.retrieval.topology.memgraph import _parameters
from apps.knowledge_graph.tests.test_projected_topology_adapter import _caps, _ready
from apps.knowledge_graph.tests.test_projection_records import _bundle
from lib.knowledge_graph.topology_gateway_config import (
    load_topology_gateway_client_settings,
)


def _scope(generation_count, document_count, token):
    base = _ready(_bundle())
    generations = tuple(
        replace(
            base.selected_generations[0],
            collection_key=f"{index + 1:064x}",
            generation_key=f"{index + 129:064x}",
            projection_key=f"{index + 257:064x}",
            active_artifact_key=f"{index + 385:064x}",
            schema_version=token * 128,
            projection_version=token * 128,
            identifier_key_version=token * 128,
            resolver_version=token * 128,
            embedding_model_signature=token * 512,
            membership_epoch=2**63 - 1,
        )
        for index in range(generation_count)
    )
    documents = tuple(
        AuthorizedProjectedDocumentV1(
            f"{index + 1:064x}",
            generations[0].collection_key,
            generations[0].generation_key,
        )
        for index in range(document_count)
    )
    ready = ReadyGenerationBundleV1(
        generations,
        documents,
        base.authorization_context_signature,
        ready_generation_bundle_checksum(
            generations,
            documents,
            base.authorization_context_signature,
        ),
    )
    seeds = tuple(ProjectedSeedV1(f"{index + 1:064x}", 1.0 / 64) for index in range(64))
    caps = replace(_caps(), branch_kind=HybridBranchKind.EXTENDED, max_seeds=64)
    return ready, seeds, caps


@pytest.mark.parametrize(
    ("generations", "documents", "token"),
    ((1, 31, "x"), (128, 10_000, "\U0001f600"), (128, 10_000, "\\")),
)
def test_all_supported_scopes_round_trip_through_outer_and_nested_wire(
    generations,
    documents,
    token,
):
    ready, seeds, caps = _scope(generations, documents, token)
    request = TopologyGatewayRequestV1(
        TopologyQueryName.GENERATION_MANIFESTS,
        _parameters(ready, seeds, caps),
        42.5,
        generations,
    )
    payload = encode_request(request)
    assert 16_384 < len(payload) <= MAX_REQUEST_BYTES
    decoded = decode_request(payload)
    assert decode_topology_request(decoded.parameters) == (ready, seeds, caps)


def test_maximum_authority_inner_json_fits_decoder_before_provider_io():
    ready, seeds, caps = _scope(128, 10_000, "x")
    parameters = _parameters(ready, seeds, caps)
    assert len(parameters["authorized_documents_json"].encode()) > 2_000_000
    assert decode_topology_request(parameters) == (ready, seeds, caps)


def test_web_client_configuration_accepts_the_supported_wire_capacity():
    settings = load_topology_gateway_client_settings(
        {
            "KG_TOPOLOGY_GATEWAY_MAX_REQUEST_BYTES": "4194304",
        }
    )
    assert settings.max_request_bytes == 4_194_304


@pytest.mark.asyncio
async def test_gateway_admits_128_generation_manifest_request(monkeypatch):
    from apps.knowledge_graph.retrieval.topology import gateway_service as service
    from apps.knowledge_graph.retrieval.topology.gateway_contracts import (
        TopologyGatewaySuccessV1,
        decode_response,
    )
    from apps.knowledge_graph.tests.test_topology_gateway_service import (
        _body,
        _call,
        _headers,
        _settings,
    )

    ready, seeds, caps = _scope(128, 10_000, "\U0001f600")

    class Adapter:
        calls = 0

        def execute_read(self, *, parameters, max_records, **_kwargs):
            assert decode_topology_request(parameters) == (ready, seeds, caps)
            assert max_records == 128
            self.calls += 1
            return tuple({"index": index} for index in range(max_records))

    adapter = Adapter()
    # This checks maximum wire admission, not a 100 ms decoding benchmark.
    settings = replace(_settings(), timeout_ms=2500)
    runtime = service.TopologyGatewayRuntime(settings, object(), adapter)
    monkeypatch.setattr(service, "monotonic", lambda: 40.0)
    monkeypatch.setattr(
        service, "load_topology_gateway_settings", lambda _env: settings
    )
    monkeypatch.setattr(service, "_get_runtime", lambda *_args: runtime)
    body = encode_request(
        TopologyGatewayRequestV1(
            TopologyQueryName.GENERATION_MANIFESTS,
            _parameters(ready, seeds, caps),
            42.5,
            128,
        )
    )
    messages = await _call(body=body, headers=_headers(body))
    response = decode_response(_body(messages))
    assert messages[0]["status"] == 200
    assert type(response) is TopologyGatewaySuccessV1
    assert len(response.rows) == 128 and adapter.calls == 1
