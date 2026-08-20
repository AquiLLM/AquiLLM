import json
from dataclasses import FrozenInstanceError, fields, replace
from enum import StrEnum

import pytest

from lib.knowledge_graph.query_extractor import contracts as extractor_contracts
from lib.knowledge_graph.query_extractor.contracts import (
    MAX_QUERY_REQUEST_BODY_BYTES,
    QUERY_EXTRACTION_RESPONSE_SCHEMA_CHECKSUM,
    QUERY_EXTRACTION_RESPONSE_SCHEMA_VERSION,
    QueryEntitySpanV1,
    QueryExtractionRequestV1,
    QueryExtractionResponseV1,
    QueryExtractorFailureReason,
    QueryExtractorProvenanceV1,
    canonical_query_extraction_request_bytes,
    canonical_query_extraction_response_bytes,
    parse_query_extraction_request,
    parse_query_extraction_response,
)

DIGEST = "a" * 64
REVISION = "b" * 40


def _request(query: str = "A😀B") -> QueryExtractionRequestV1:
    return QueryExtractionRequestV1(
        schema_version="query-request-v1",
        query=query,
        ontology_checksum=DIGEST,
        max_query_utf8_bytes=32,
        max_query_code_points=16,
        max_spans=4,
    )


def _provenance() -> QueryExtractorProvenanceV1:
    return QueryExtractorProvenanceV1(
        model_identifier="fastino/gliner2-base-v1",
        model_revision=REVISION,
        schema_version=QUERY_EXTRACTION_RESPONSE_SCHEMA_VERSION,
        schema_checksum=QUERY_EXTRACTION_RESPONSE_SCHEMA_CHECKSUM,
        ontology_checksum=DIGEST,
        build_hash="c" * 64,
    )


def _response() -> QueryExtractionResponseV1:
    return QueryExtractionResponseV1(
        provenance=_provenance(),
        query_utf8_bytes=6,
        query_code_points=3,
        spans=(QueryEntitySpanV1("person", 1, 2, 0.75),),
    )


def test_schema_vector_field_order_and_closed_failures() -> None:
    assert QUERY_EXTRACTION_RESPONSE_SCHEMA_VERSION == "query-entities-v1"
    assert QUERY_EXTRACTION_RESPONSE_SCHEMA_CHECKSUM == (
        "45bc8f86637a73324d2edae3096aac61d242fb0bcbab3c481cfa7599456cd271"
    )
    assert tuple(field.name for field in fields(QueryEntitySpanV1)) == (
        "ontology_type",
        "start",
        "end",
        "confidence",
    )
    assert tuple(field.name for field in fields(QueryExtractionResponseV1)) == (
        "provenance",
        "query_utf8_bytes",
        "query_code_points",
        "spans",
    )
    assert tuple(QueryExtractorFailureReason) == (
        "extractor_timeout",
        "extractor_auth",
        "extractor_provenance",
    )
    assert issubclass(QueryExtractorFailureReason, StrEnum)


def test_unicode_code_point_spans_are_text_free_bounded_and_canonical() -> None:
    response = _response()
    assert not hasattr(response.spans[0], "text")
    assert "A😀B"[response.spans[0].start : response.spans[0].end] == "😀"
    assert "😀".encode() not in canonical_query_extraction_response_bytes(response)
    with pytest.raises(FrozenInstanceError):
        response.query_code_points = 4  # type: ignore[misc]
    assert not hasattr(response, "__dict__")
    with pytest.raises(ValueError):
        QueryEntitySpanV1("person", 2, 2, 0.5)
    for mutation in (
        {"spans": (QueryEntitySpanV1("person", 2, 4, 0.5),)},
        {
            "spans": (
                QueryEntitySpanV1("person", 0, 2, 0.5),
                QueryEntitySpanV1("org", 1, 3, 0.5),
            )
        },
        {
            "spans": (
                QueryEntitySpanV1("person", 1, 2, 0.5),
                QueryEntitySpanV1("person", 0, 1, 0.5),
            )
        },
    ):
        with pytest.raises((TypeError, ValueError)):
            replace(response, **mutation)


def test_exact_builtin_types_tokens_revisions_digests_and_caps() -> None:
    class Text(str):
        pass

    for value in (True, -1, 2**31):
        with pytest.raises((TypeError, ValueError)):
            replace(_response(), query_code_points=value)
    for value in (1, float("nan"), -0.1, 1.1):
        with pytest.raises((TypeError, ValueError)):
            QueryEntitySpanV1("person", 0, 1, value)  # type: ignore[arg-type]
    for value in (" Person", "person\n", "", Text("person")):
        with pytest.raises((TypeError, ValueError)):
            QueryEntitySpanV1(value, 0, 1, 0.5)
    for value in (REVISION.upper(), "f" * 39, Text(REVISION)):
        with pytest.raises((TypeError, ValueError)):
            replace(_provenance(), model_revision=value)
    with pytest.raises(TypeError):
        replace(_request(), schema_version=Text("query-request-v1"))
    for changes in (
        {"schema_version": Text(QUERY_EXTRACTION_RESPONSE_SCHEMA_VERSION)},
        {"schema_checksum": Text(QUERY_EXTRACTION_RESPONSE_SCHEMA_CHECKSUM)},
    ):
        with pytest.raises(TypeError):
            replace(_provenance(), **changes)
    with pytest.raises(ValueError, match="UTF-8"):
        QueryEntitySpanV1(chr(0xD800), 0, 1, 0.5)
    with pytest.raises(ValueError):
        _request("x" * 33)
    for character in ("\x00", "\t", "\n", "\x1f", "\x7f"):
        with pytest.raises(ValueError, match="control"):
            _request(f"a{character}b")
    with pytest.raises(ValueError, match="cap"):
        parse_query_extraction_request(b"x" * (MAX_QUERY_REQUEST_BODY_BYTES + 1))
    with pytest.raises(ValueError, match="cap"):
        replace(_response(), spans=_response().spans * 129)
    with pytest.raises(ValueError):
        QueryExtractionRequestV1("query-request-v1", chr(0xD800), DIGEST, 32, 16, 4)


def test_canonical_json_has_exact_fields_and_rejects_unknown_or_noncanonical() -> None:
    request_bytes = canonical_query_extraction_request_bytes(_request())
    assert parse_query_extraction_request(request_bytes) == _request()
    response_bytes = canonical_query_extraction_response_bytes(_response())
    assert parse_query_extraction_response(response_bytes) == _response()
    assert b'"confidence":"0x1.8000000000000p-1"' in response_bytes
    request_payload = json.loads(request_bytes)
    request_payload["unknown"] = 1
    with pytest.raises(ValueError, match="field"):
        parse_query_extraction_request(json.dumps(request_payload).encode())
    response_payload = json.loads(response_bytes)
    response_payload["spans"][0]["text"] = "secret"
    with pytest.raises(ValueError, match="field"):
        parse_query_extraction_response(
            json.dumps(
                response_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
    with pytest.raises(ValueError, match="canonical"):
        parse_query_extraction_request(request_bytes.replace(b"{", b"{ ", 1))
    response_payload = json.loads(response_bytes)
    response_payload["spans"][0]["confidence"] = "0X1.8P-1"
    alternate = json.dumps(
        response_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    with pytest.raises(ValueError, match="canonical hexadecimal"):
        parse_query_extraction_response(alternate)
    response_payload["spans"][0]["confidence"] = (
        "0x1.0000000000000p+999999999999999999999"
    )
    overflow = json.dumps(
        response_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    with pytest.raises(ValueError, match="canonical hexadecimal"):
        parse_query_extraction_response(overflow)
    request_payload = json.loads(request_bytes)
    request_payload["query"] = chr(0xD800)
    escaped_surrogate = json.dumps(
        request_payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode()
    with pytest.raises(ValueError, match="UTF-8"):
        parse_query_extraction_request(escaped_surrogate)


def test_response_parser_rejects_span_overflow_before_construction(monkeypatch) -> None:
    payload = json.loads(canonical_query_extraction_response_bytes(_response()))
    payload["spans"] *= 129
    body = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    monkeypatch.setattr(
        extractor_contracts,
        "QueryEntitySpanV1",
        lambda *args: pytest.fail("span constructed before cap check"),
    )
    with pytest.raises(ValueError, match="cap"):
        parse_query_extraction_response(body)


def test_contract_package_exports_data_only_without_optional_runtime() -> None:
    import sys

    loaded_before = set(sys.modules)
    __import__("lib.knowledge_graph.query_extractor")
    newly_loaded = set(sys.modules) - loaded_before
    forbidden = {"django", "gliner2", "torch", "httpx", "requests"}
    assert not {name.partition(".")[0] for name in newly_loaded} & forbidden
