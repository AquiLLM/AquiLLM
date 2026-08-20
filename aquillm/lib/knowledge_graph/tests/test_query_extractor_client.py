from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from lib.knowledge_graph.query_extractor.client import (
    QueryExtractorClient,
    QueryExtractorClientError,
    QueryExtractorHTTPResponse,
    reconstruct_entity_texts,
)
from lib.knowledge_graph.query_extractor.config import (
    QueryExtractorConfigError,
    load_query_extractor_settings,
)
from lib.knowledge_graph.query_extractor.contracts import (
    QUERY_EXTRACTION_RESPONSE_SCHEMA_CHECKSUM,
    QUERY_EXTRACTION_RESPONSE_SCHEMA_VERSION,
    QueryEntitySpanV1,
    QueryExtractionResponseV1,
    QueryExtractorFailureReason,
    QueryExtractorProvenanceV1,
    canonical_query_extraction_response_bytes,
)

DIGEST = "a" * 64
REVISION = "8437ba583a733d87f56ae902f3b197934eedd58e"
BUILD = "b" * 64
EMOJI = chr(0x1F600)


@dataclass(frozen=True)
class Ontology:
    checksum: str = DIGEST
    entity_types: tuple[str, ...] = ("model",)


def _environment(**overrides: str) -> dict[str, str]:
    values = {
        "KG_QUERY_EXTRACTOR_URL": "https://extractor.internal/v1/extract",
        "KG_QUERY_EXTRACTOR_BEARER_TOKEN": "private-token",
        "KG_QUERY_EXTRACTOR_MODEL": "fastino/gliner2-base-v1",
        "KG_QUERY_EXTRACTOR_MODEL_REVISION": REVISION,
        "KG_QUERY_EXTRACTOR_BUILD_HASH": BUILD,
        "KG_QUERY_EXTRACTOR_EXPECTED_SCHEMA_VERSION": (
            QUERY_EXTRACTION_RESPONSE_SCHEMA_VERSION
        ),
        "KG_QUERY_EXTRACTOR_EXPECTED_SCHEMA_CHECKSUM": (
            QUERY_EXTRACTION_RESPONSE_SCHEMA_CHECKSUM
        ),
        "KG_QUERY_EXTRACTOR_ONTOLOGY_PATH": str(Path("ontology.yaml")),
        "KG_QUERY_EXTRACTOR_ONTOLOGY_CHECKSUM": DIGEST,
        "KG_QUERY_EXTRACTOR_TIMEOUT_MS": "75",
        "KG_QUERY_MAX_BYTES": "64",
        "KG_QUERY_MAX_CODEPOINTS": "32",
        "KG_QUERY_MAX_SPANS": "4",
    }
    values.update(overrides)
    return values


def _response(query: str, *, provenance_checksum: str = DIGEST) -> bytes:
    return canonical_query_extraction_response_bytes(
        QueryExtractionResponseV1(
            provenance=QueryExtractorProvenanceV1(
                model_identifier="fastino/gliner2-base-v1",
                model_revision=REVISION,
                schema_version=QUERY_EXTRACTION_RESPONSE_SCHEMA_VERSION,
                schema_checksum=QUERY_EXTRACTION_RESPONSE_SCHEMA_CHECKSUM,
                ontology_checksum=provenance_checksum,
                build_hash=BUILD,
            ),
            query_utf8_bytes=len(query.encode()),
            query_code_points=len(query),
            spans=(QueryEntitySpanV1("model", 1, 2, 0.75),),
        )
    )


def test_settings_are_strict_and_keep_bearer_out_of_repr() -> None:
    settings = load_query_extractor_settings(_environment())
    assert settings.timeout_ms == 75
    assert settings.max_query_utf8_bytes == 64
    assert "private-token" not in repr(settings)
    assert "private-token" not in repr(settings.bearer_token)

    for key, value in (
        ("KG_QUERY_EXTRACTOR_URL", "https://EXTRACTOR/v1/extract"),
        ("KG_QUERY_EXTRACTOR_BEARER_TOKEN", ""),
        ("KG_QUERY_EXTRACTOR_MODEL_REVISION", REVISION.upper()),
        ("KG_QUERY_EXTRACTOR_BUILD_HASH", "c" * 63),
        ("KG_QUERY_MAX_SPANS", "05"),
    ):
        with pytest.raises(QueryExtractorConfigError, match=key):
            load_query_extractor_settings(_environment(**{key: value}))


def test_client_posts_one_canonical_body_and_reconstructs_code_point_spans() -> None:
    calls: list[dict[str, object]] = []
    query = f"A{EMOJI}B"

    def request_once(**kwargs: object) -> QueryExtractorHTTPResponse:
        calls.append(kwargs)
        return QueryExtractorHTTPResponse(200, _response(query))

    client = QueryExtractorClient(
        load_query_extractor_settings(_environment()),
        request_once=request_once,
        monotonic=lambda: 10.0,
    )
    response = client.extract(query=query, ontology=Ontology(), deadline=10.05)

    assert len(calls) == 1
    assert calls[0]["url"] == "https://extractor.internal/v1/extract"
    assert calls[0]["headers"] == {
        "Authorization": "Bearer private-token",
        "Content-Type": "application/json",
    }
    assert calls[0]["body"] == (
        b'{"max_query_code_points":32,"max_query_utf8_bytes":64,'
        b'"max_spans":4,"ontology_checksum":"'
        + DIGEST.encode()
        + b'","query":"A'
        + EMOJI.encode()
        + b'B","schema_version":"query-request-v1"}'
    )
    assert calls[0]["timeout_seconds"] == pytest.approx(0.05)
    assert reconstruct_entity_texts(query=query, response=response) == (EMOJI,)


def test_client_enforces_local_caps_before_io() -> None:
    calls = 0

    def request_once(**_kwargs: object) -> QueryExtractorHTTPResponse:
        nonlocal calls
        calls += 1
        raise AssertionError("I/O reached")

    client = QueryExtractorClient(
        load_query_extractor_settings(_environment()),
        request_once=request_once,
        monotonic=lambda: 1.0,
    )
    for query in ("x" * 33, EMOJI * 17):
        with pytest.raises(ValueError):
            client.extract(query=query, ontology=Ontology(), deadline=2.0)
    assert calls == 0


@pytest.mark.parametrize(
    ("response", "expected"),
    (
        (
            QueryExtractorHTTPResponse(307, b""),
            QueryExtractorFailureReason.EXTRACTOR_PROVENANCE,
        ),
        (
            QueryExtractorHTTPResponse(401, b""),
            QueryExtractorFailureReason.EXTRACTOR_AUTH,
        ),
        (
            QueryExtractorHTTPResponse(
                200, _response(f"A{EMOJI}B", provenance_checksum="c" * 64)
            ),
            QueryExtractorFailureReason.EXTRACTOR_PROVENANCE,
        ),
    ),
)
def test_client_has_fixed_failures_and_never_retries_or_follows_redirects(
    response: QueryExtractorHTTPResponse,
    expected: QueryExtractorFailureReason,
) -> None:
    calls = 0

    def request_once(**_kwargs: object) -> QueryExtractorHTTPResponse:
        nonlocal calls
        calls += 1
        return response

    client = QueryExtractorClient(
        load_query_extractor_settings(_environment()),
        request_once=request_once,
        monotonic=lambda: 1.0,
    )
    with pytest.raises(QueryExtractorClientError) as exc_info:
        client.extract(query=f"A{EMOJI}B", ontology=Ontology(), deadline=2.0)
    assert exc_info.value.reason is expected
    assert calls == 1


def test_expired_deadline_is_a_fixed_timeout_without_io() -> None:
    client = QueryExtractorClient(
        load_query_extractor_settings(_environment()),
        request_once=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("I/O")),
        monotonic=lambda: 5.0,
    )
    with pytest.raises(QueryExtractorClientError) as exc_info:
        client.extract(query="model", ontology=Ontology(), deadline=5.0)
    assert exc_info.value.reason is QueryExtractorFailureReason.EXTRACTOR_TIMEOUT
