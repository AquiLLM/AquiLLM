import importlib.util
import json

import pytest

from apps.knowledge_graph.retrieval.topology import gateway_contracts as v1


def contract():
    name = "apps.knowledge_graph.retrieval.topology.gateway_snapshot_contracts"
    assert importlib.util.find_spec(name) is not None, (
        "a separately pinned V2 contract is required"
    )
    return importlib.import_module(name)


def manifest():
    return (
        {
            key: "a" * 64
            for key in (
                "collection_key",
                "generation_key",
                "projection_key",
                "active_artifact_key",
                "graph_checksum",
                "membership_checksum",
            )
        },
    )


def test_closed_v2_roundtrips_without_mutating_v1_descriptor():
    v2 = contract()
    assert v2.SCHEMA_VERSION == "topology-gateway-v2"
    assert (
        v2.SCHEMA_CHECKSUM
        == "311145c9b4d281b2653b57ac76f33e6afd5498d1c71191338d6a6fe25e29371e"
    )
    assert v2.SCHEMA_CHECKSUM != v1.SCHEMA_CHECKSUM
    request = v2.TopologyGatewayRequestV2({"scope": "opaque"}, 42.5)
    assert v2.encode_request(
        v2.decode_request(v2.encode_request(request))
    ) == v2.encode_request(request)
    for response in (
        v2.TopologyGatewaySnapshotV2(manifest(), "{}"),
        v2.TopologyGatewayFamilyTransportV2(manifest()),
        v1.TopologyGatewayFailureV1(v1.GatewayFailureReason.DEADLINE),
    ):
        raw = v2.encode_response(response)
        assert v2.encode_response(v2.decode_response(raw)) == raw


@pytest.mark.parametrize(
    "raw",
    [
        b"{}",
        b'{"deadline":42.5,"parameters":{},"query":"generation_manifests"}',
        b'{"deadline":42.5,"deadline":42.5,"parameters":{}}',
        b'{"deadline":42,"parameters":{}}',
        b'{"deadline":42.5,"parameters":{} }',
        b'{"deadline":42.5,"parameters":{"x":[]}}',
    ],
)
def test_rejects_malformed_closed_requests(raw):
    with pytest.raises(ValueError):
        contract().decode_request(raw)


@pytest.mark.parametrize(
    "value",
    [
        {"ok": True, "delivery": "unknown", "manifests": []},
        {"ok": True, "delivery": "snapshot", "manifests": []},
        {
            "ok": True,
            "delivery": "use_family_transport",
            "manifests": [],
            "snapshot_json": "{}",
        },
        {"ok": 1, "delivery": "use_family_transport", "manifests": []},
        {"ok": True, "delivery": "use_family_transport", "manifests": [{"query": "x"}]},
        {"ok": False, "reason": "deadline", "status": True},
    ],
)
def test_rejects_malformed_closed_responses(value):
    with pytest.raises(ValueError):
        contract().decode_response(v1._canonical(value))


def test_wire_bound_is_distinct_from_decoded_snapshot_bound():
    v2 = contract()
    result = v2.TopologyGatewaySnapshotV2(manifest(), '"' * 600_000)
    with pytest.raises(OverflowError):
        v2.encode_response(result)
    with pytest.raises(OverflowError):
        v2.TopologyGatewaySnapshotV2(manifest(), "x" * 2_000_001)
    with pytest.raises(OverflowError):
        v2.TopologyGatewaySnapshotV2(manifest(), "é" * 1_000_001)


def test_exact_combined_wire_threshold_counts_escaping_and_utf8():
    v2 = contract()
    base = v2.TopologyGatewaySnapshotV2(manifest(), 'é"')
    raw = v2.encode_response(base)
    assert json.loads(raw)["snapshot_json"] == 'é"'
    for delta in (-1, 0, 1):
        result = v2.TopologyGatewaySnapshotV2(
            manifest(), 'é"' + "x" * (v1.MAX_RESPONSE_BYTES - len(raw) + delta)
        )
        if delta == 1:
            with pytest.raises(OverflowError):
                v2.encode_response(result)
        else:
            assert len(v2.encode_response(result)) == v1.MAX_RESPONSE_BYTES + delta


@pytest.mark.parametrize("token", ["\U0001f600", "\\"])
def test_maximum_scope_roundtrips_through_v2_and_existing_authority_decoder(token):
    from apps.knowledge_graph.projection.topology_request import decode_topology_request
    from apps.knowledge_graph.retrieval.topology.memgraph import _parameters
    from apps.knowledge_graph.tests.test_topology_gateway_request_capacity import _scope

    v2 = contract()
    ready, seeds, caps = _scope(128, 10_000, token)
    wire = v2.encode_request(
        v2.TopologyGatewayRequestV2(_parameters(ready, seeds, caps), 42.5)
    )
    assert len(wire) <= v1.MAX_REQUEST_BYTES
    assert decode_topology_request(v2.decode_request(wire).parameters) == (
        ready,
        seeds,
        caps,
    )


def test_duplicate_fields_and_noncanonical_responses_rejected():
    v2 = contract()
    raw = v2.encode_response(v2.TopologyGatewayFamilyTransportV2(manifest()))
    for malformed in (raw.replace(b'"ok":true', b'"ok":true,"ok":true'), raw + b" "):
        with pytest.raises(ValueError):
            v2.decode_response(malformed)
