from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest
import yaml

from apps.knowledge_graph.services.ontology import load_ontology, load_ontology_yaml
from lib.knowledge_graph.query_extractor.contracts import (
    QueryExtractionRequestV1,
    canonical_query_extraction_request_bytes,
    parse_query_extraction_request,
)
from lib.knowledge_graph.query_extractor.ontology_payload import (
    load_ontology_definition,
    ontology_definition_payload,
)


def _checksum(document):
    return sha256(
        json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _definition():
    return load_ontology(
        Path(__file__).resolve().parents[3]
        / "apps/knowledge_graph/ontologies/research-v1.yaml"
    )


@pytest.mark.parametrize("custom", [False, True])
def test_transport_roundtrip_matches_real_ontology_canonical_identity(custom):
    original = _definition()
    document = yaml.safe_load(original.canonical_yaml)
    if custom:
        document["version"] = "0.0.7+collection.223"
        document["entity_types"][0]["description"] = "Generated collection entity."
    ontology = load_ontology_yaml(yaml.safe_dump(document))
    payload = ontology_definition_payload(ontology)
    selected = load_ontology_definition(payload, expected_checksum=ontology.checksum)
    assert selected.version == ontology.version
    assert set(selected.entity_types) == set(ontology.entity_types)
    assert selected.checksum == _checksum(payload) == ontology.checksum
    request = QueryExtractionRequestV1(
        "query-request-v1",
        "Graph retrieval",
        ontology.checksum,
        64,
        32,
        4,
        ontology_definition=payload,
    )
    encoded = canonical_query_extraction_request_bytes(request)
    parsed = parse_query_extraction_request(encoded)
    assert parsed == request
    assert canonical_query_extraction_request_bytes(parsed) == encoded
    with pytest.raises(TypeError):
        selected.entity_types["new"] = {}
    with pytest.raises(TypeError):
        selected.entity_types[next(iter(selected.entity_types))]["description"] = (
            "changed"
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "bool",
        "unknown_field",
        "reserved",
        "noncanonical",
        "long_name",
        "bad_version",
        "bad_endpoint",
        "duplicate_endpoint",
        "duplicate_alias",
        "alias_collision",
        "duplicate_entity",
        "duplicate_relation",
        "entity_count",
        "relation_count",
        "alias_count",
        "long_description",
        "bad_direction",
        "bad_extension",
    ],
)
def test_invalid_semantics_rejected_even_with_matching_payload_hash(mutation):
    document = ontology_definition_payload(_definition())
    entity = document["entity_types"][0]
    relation = document["relations"][0]
    if mutation == "bool":
        entity["default_retrieval_weight"] = True
    elif mutation == "unknown_field":
        document["unknown"] = 1
    elif mutation == "reserved":
        entity["name"] = "entities"
    elif mutation == "noncanonical":
        entity["name"] = "InvalidName"
    elif mutation == "long_name":
        entity["name"] = "a" * 65
    elif mutation == "bad_version":
        document["version"] = "not-semver"
    elif mutation == "bad_endpoint":
        relation["allowed_head_types"] = ["foreign"]
    elif mutation == "duplicate_endpoint":
        relation["allowed_head_types"] *= 2
    elif mutation == "duplicate_alias":
        entity["aliases"] = ["duplicate", "duplicate"]
    elif mutation == "alias_collision":
        entity["aliases"] = [document["entity_types"][1]["name"]]
    elif mutation == "duplicate_entity":
        document["entity_types"].append(dict(entity))
    elif mutation == "duplicate_relation":
        document["relations"].append(dict(relation))
    elif mutation == "entity_count":
        document["entity_types"] = [entity] * 65
    elif mutation == "relation_count":
        document["relations"] = [relation] * 129
    elif mutation == "alias_count":
        entity["aliases"] = [f"alias-{index}" for index in range(33)]
    elif mutation == "long_description":
        entity["description"] = "x" * 513
    elif mutation == "bad_direction":
        relation["direction"] = "both"
    else:
        entity["extension_enabled"] = 1
    with pytest.raises(ValueError):
        load_ontology_definition(document, expected_checksum=_checksum(document))


def test_definition_byte_limit_and_digest_are_enforced():
    document = ontology_definition_payload(_definition())
    with pytest.raises(ValueError, match="checksum"):
        load_ontology_definition(document, expected_checksum="f" * 64)
    document["entity_types"][0]["description"] = "x" * 65_536
    with pytest.raises(ValueError, match="byte cap"):
        load_ontology_definition(document, expected_checksum=_checksum(document))


def test_request_revalidates_payload_mutated_after_construction():
    ontology = _definition()
    request = QueryExtractionRequestV1(
        "query-request-v1",
        "Graph retrieval",
        ontology.checksum,
        64,
        32,
        4,
        ontology_definition=ontology_definition_payload(ontology),
    )
    request.ontology_definition["entity_types"][0]["description"] = "Tampered"
    with pytest.raises(ValueError, match="checksum"):
        canonical_query_extraction_request_bytes(request)


@pytest.mark.parametrize("mutation", ["padding", "order", "integer"])
def test_inner_wire_definition_requires_canonical_semantic_representation(mutation):
    ontology = _definition()
    document = ontology_definition_payload(ontology)
    if mutation == "padding":
        document["entity_types"][0]["description"] += " "
    elif mutation == "order":
        document["entity_types"].reverse()
    else:
        document["entity_types"][0]["default_retrieval_weight"] = 1
        # Choose a field whose original semantic value is exactly one.
        original = ontology_definition_payload(ontology)
        for row in original["entity_types"]:
            row["default_retrieval_weight"] = 1.0
        document = original
        ontology = load_ontology_yaml(yaml.safe_dump(original))
        document["entity_types"][0]["default_retrieval_weight"] = 1
    with pytest.raises(ValueError, match="canonical"):
        load_ontology_definition(document, expected_checksum=ontology.checksum)
