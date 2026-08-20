# ruff: noqa: E501
# fmt: off
"""Provider-neutral, text-minimizing query extraction wire contracts."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import final

QUERY_EXTRACTION_REQUEST_SCHEMA_VERSION = "query-request-v1"
QUERY_EXTRACTION_RESPONSE_SCHEMA_VERSION = "query-entities-v1"
QUERY_EXTRACTION_RESPONSE_SCHEMA_CHECKSUM = (
    "45bc8f86637a73324d2edae3096aac61d242fb0bcbab3c481cfa7599456cd271"
)
MAX_QUERY_UTF8_BYTES = 16_384
MAX_QUERY_CODE_POINTS = 8_192
MAX_QUERY_SPANS = 128
MAX_QUERY_REQUEST_BODY_BYTES = 32_768
MAX_QUERY_RESPONSE_BODY_BYTES = 131_072
_DIGEST = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
class QueryExtractorFailureReason(StrEnum):
    EXTRACTOR_TIMEOUT = "extractor_timeout"
    EXTRACTOR_AUTH = "extractor_auth"
    EXTRACTOR_PROVENANCE = "extractor_provenance"
def _int(value: object, name: str, minimum: int, maximum: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact int")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} is outside its bound")
def _token(value: object, name: str, maximum: int = 256) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact str")
    if not value or value != value.strip() or len(value) > maximum:
        raise ValueError(f"{name} must be a bounded canonical token")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{name} contains a forbidden control character")
def _digest(value: object, name: str, pattern: re.Pattern[str] = _DIGEST) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact str")
    if pattern.fullmatch(value) is None:
        raise ValueError(f"{name} must use its lowercase hexadecimal encoding")
@final
@dataclass(frozen=True, slots=True)
class QueryExtractionRequestV1:
    schema_version: str
    query: str
    ontology_checksum: str
    max_query_utf8_bytes: int
    max_query_code_points: int
    max_spans: int
    def __post_init__(self) -> None:
        if self.schema_version != QUERY_EXTRACTION_REQUEST_SCHEMA_VERSION:
            raise ValueError("schema_version is not query-request-v1")
        if type(self.query) is not str:
            raise TypeError("query must be an exact str")
        try:
            encoded = self.query.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("query must be valid UTF-8") from error
        if any(ord(character) < 32 or ord(character) == 127 for character in self.query):
            raise ValueError("query contains a forbidden C0/DEL control character")
        _digest(self.ontology_checksum, "ontology_checksum")
        _int(self.max_query_utf8_bytes, "max_query_utf8_bytes", 1, MAX_QUERY_UTF8_BYTES)
        _int(
            self.max_query_code_points,
            "max_query_code_points",
            1,
            MAX_QUERY_CODE_POINTS,
        )
        _int(self.max_spans, "max_spans", 1, MAX_QUERY_SPANS)
        if not self.query or len(encoded) > self.max_query_utf8_bytes:
            raise ValueError("query exceeds its UTF-8 byte bound")
        if len(self.query) > self.max_query_code_points:
            raise ValueError("query exceeds its code-point bound")
@final
@dataclass(frozen=True, slots=True)
class QueryExtractorProvenanceV1:
    model_identifier: str
    model_revision: str
    schema_version: str
    schema_checksum: str
    ontology_checksum: str
    build_hash: str
    def __post_init__(self) -> None:
        _token(self.model_identifier, "model_identifier")
        _digest(self.model_revision, "model_revision", _REVISION)
        if self.schema_version != QUERY_EXTRACTION_RESPONSE_SCHEMA_VERSION:
            raise ValueError("schema_version is not query-entities-v1")
        if self.schema_checksum != QUERY_EXTRACTION_RESPONSE_SCHEMA_CHECKSUM:
            raise ValueError("schema_checksum does not bind the frozen schema")
        _digest(self.ontology_checksum, "ontology_checksum")
        _digest(self.build_hash, "build_hash")
@final
@dataclass(frozen=True, slots=True)
class QueryEntitySpanV1:
    ontology_type: str
    start: int
    end: int
    confidence: float
    def __post_init__(self) -> None:
        _token(self.ontology_type, "ontology_type", 128)
        _int(self.start, "start", 0, MAX_QUERY_CODE_POINTS - 1)
        _int(self.end, "end", 1, MAX_QUERY_CODE_POINTS)
        if self.start >= self.end:
            raise ValueError("span must be nonempty and half-open")
        if type(self.confidence) is not float:
            raise TypeError("confidence must be an exact float")
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")
@final
@dataclass(frozen=True, slots=True)
class QueryExtractionResponseV1:
    provenance: QueryExtractorProvenanceV1
    query_utf8_bytes: int
    query_code_points: int
    spans: tuple[QueryEntitySpanV1, ...]
    def __post_init__(self) -> None:
        if type(self.provenance) is not QueryExtractorProvenanceV1:
            raise TypeError("provenance must be exact")
        _int(self.query_utf8_bytes, "query_utf8_bytes", 1, MAX_QUERY_UTF8_BYTES)
        _int(self.query_code_points, "query_code_points", 1, MAX_QUERY_CODE_POINTS)
        if self.query_utf8_bytes < self.query_code_points:
            raise ValueError("UTF-8 bytes cannot be fewer than code points")
        if type(self.spans) is not tuple:
            raise TypeError("spans must be an exact tuple")
        if len(self.spans) > MAX_QUERY_SPANS:
            raise ValueError("spans exceed the hard cap")
        if any(type(span) is not QueryEntitySpanV1 for span in self.spans):
            raise TypeError("spans must contain exact QueryEntitySpanV1 values")
        keys = tuple((span.start, span.end, span.ontology_type) for span in self.spans)
        if len(set(keys)) != len(keys) or keys != tuple(sorted(keys)):
            raise ValueError("spans must be unique and canonically ordered")
        if any(span.end > self.query_code_points for span in self.spans):
            raise ValueError("span exceeds the query code-point length")
        if any(
            left.end > right.start for left, right in zip(self.spans, self.spans[1:])
        ):
            raise ValueError("spans must not overlap")
def _canonical(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
def canonical_query_extraction_request_bytes(value: QueryExtractionRequestV1) -> bytes:
    if type(value) is not QueryExtractionRequestV1:
        raise TypeError("value must be an exact QueryExtractionRequestV1")
    return _canonical(
        {
            "max_query_code_points": value.max_query_code_points,
            "max_query_utf8_bytes": value.max_query_utf8_bytes,
            "max_spans": value.max_spans,
            "ontology_checksum": value.ontology_checksum,
            "query": value.query,
            "schema_version": value.schema_version,
        }
    )
def canonical_query_extraction_response_bytes(
    value: QueryExtractionResponseV1,
) -> bytes:
    if type(value) is not QueryExtractionResponseV1:
        raise TypeError("value must be an exact QueryExtractionResponseV1")
    provenance = {
        name: getattr(value.provenance, name)
        for name in (
            "model_identifier",
            "model_revision",
            "schema_version",
            "schema_checksum",
            "ontology_checksum",
            "build_hash",
        )
    }
    spans = [
        {
            "confidence": span.confidence.hex(),
            "end": span.end,
            "ontology_type": span.ontology_type,
            "start": span.start,
        }
        for span in value.spans
    ]
    return _canonical(
        {
            "provenance": provenance,
            "query_code_points": value.query_code_points,
            "query_utf8_bytes": value.query_utf8_bytes,
            "spans": spans,
        }
    )
def _payload(data: bytes, fields: frozenset[str], maximum: int) -> dict[str, object]:
    if type(data) is not bytes:
        raise TypeError("wire data must be exact bytes")
    if len(data) > maximum:
        raise ValueError("wire data exceeds its byte cap")
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("wire data must be valid JSON") from error
    if type(value) is not dict or set(value) != fields:
        raise ValueError("wire object has an invalid field set")
    if data != _canonical(value):
        raise ValueError("wire object must use canonical JSON")
    return value
def parse_query_extraction_request(data: bytes) -> QueryExtractionRequestV1:
    value = _payload(
        data,
        frozenset(
            {
                "schema_version",
                "query",
                "ontology_checksum",
                "max_query_utf8_bytes",
                "max_query_code_points",
                "max_spans",
            }
        ),
        MAX_QUERY_REQUEST_BODY_BYTES,
    )
    return QueryExtractionRequestV1(**value)  # type: ignore[arg-type]
def parse_query_extraction_response(data: bytes) -> QueryExtractionResponseV1:
    value = _payload(
        data,
        frozenset({"provenance", "query_utf8_bytes", "query_code_points", "spans"}),
        MAX_QUERY_RESPONSE_BODY_BYTES,
    )
    provenance = value["provenance"]
    spans = value["spans"]
    if type(provenance) is not dict or set(provenance) != {
        "model_identifier",
        "model_revision",
        "schema_version",
        "schema_checksum",
        "ontology_checksum",
        "build_hash",
    }:
        raise ValueError("provenance has an invalid field set")
    if type(spans) is not list:
        raise TypeError("spans must be a JSON array")
    parsed_spans: list[QueryEntitySpanV1] = []
    for span in spans:
        if type(span) is not dict or set(span) != {
            "ontology_type",
            "start",
            "end",
            "confidence",
        }:
            raise ValueError("span has an invalid field set")
        confidence = span["confidence"]
        if type(confidence) is not str:
            raise TypeError("confidence must use canonical hexadecimal text")
        try:
            parsed_confidence = float.fromhex(confidence)
        except (ValueError, OverflowError) as error:
            raise ValueError("confidence must use canonical hexadecimal text") from error
        if confidence != parsed_confidence.hex():
            raise ValueError("confidence must use canonical hexadecimal text")
        parsed_spans.append(
            QueryEntitySpanV1(
                span["ontology_type"],  # type: ignore[arg-type]
                span["start"],  # type: ignore[arg-type]
                span["end"],  # type: ignore[arg-type]
                parsed_confidence,
            )
        )
    return QueryExtractionResponseV1(
        QueryExtractorProvenanceV1(**provenance),  # type: ignore[arg-type]
        value["query_utf8_bytes"],  # type: ignore[arg-type]
        value["query_code_points"],  # type: ignore[arg-type]
        tuple(parsed_spans),
    )
