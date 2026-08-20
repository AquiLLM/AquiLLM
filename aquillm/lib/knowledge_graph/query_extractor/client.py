"""One-shot authenticated client for the query extraction sidecar."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .config import QueryExtractorSettings
from .contracts import (
    QUERY_EXTRACTION_REQUEST_SCHEMA_VERSION,
    QueryExtractionRequestV1,
    QueryExtractionResponseV1,
    QueryExtractorFailureReason,
    canonical_query_extraction_request_bytes,
    parse_query_extraction_response,
)


class OntologyDefinition(Protocol):
    checksum: str
    entity_types: object


@dataclass(frozen=True, slots=True)
class QueryExtractorHTTPResponse:
    status: int
    body: bytes


class QueryExtractorClientError(RuntimeError):
    """Fixed failure raised at the remote extractor boundary."""

    def __init__(self, reason: QueryExtractorFailureReason) -> None:
        self.reason = reason
        super().__init__(reason.value)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args: object, **_kwargs: object) -> None:
        return None


def _stdlib_request_once(
    *,
    url: str,
    headers: dict[str, str],
    body: bytes,
    timeout_seconds: float,
    max_response_body_bytes: int,
) -> QueryExtractorHTTPResponse:
    request = Request(url, data=body, headers=headers, method="POST")
    opener = build_opener(_NoRedirect())
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            payload = response.read(max_response_body_bytes + 1)
            return QueryExtractorHTTPResponse(response.status, payload)
    except HTTPError as error:
        return QueryExtractorHTTPResponse(
            error.code, error.read(max_response_body_bytes + 1)
        )


RequestOnce = Callable[..., QueryExtractorHTTPResponse]


def reconstruct_entity_texts(
    *, query: str, response: QueryExtractionResponseV1
) -> tuple[str, ...]:
    """Reconstruct transient entity surfaces only from trusted local query text."""

    if type(query) is not str or type(response) is not QueryExtractionResponseV1:
        raise TypeError("query and response must be exact contract values")
    if (
        len(query) != response.query_code_points
        or len(query.encode()) != response.query_utf8_bytes
    ):
        raise ValueError("response query lengths do not match the local query")
    surfaces = tuple(query[span.start : span.end] for span in response.spans)
    if any(not surface for surface in surfaces):
        raise ValueError("response contains an invalid query span")
    return surfaces


class QueryExtractorClient:
    def __init__(
        self,
        settings: QueryExtractorSettings,
        *,
        request_once: RequestOnce = _stdlib_request_once,
        monotonic: Callable[[], float] = monotonic,
    ) -> None:
        if type(settings) is not QueryExtractorSettings:
            raise TypeError("settings must be exact QueryExtractorSettings")
        self._settings = settings
        self._request_once = request_once
        self._monotonic = monotonic

    def extract(
        self, *, query: str, ontology: OntologyDefinition, deadline: float
    ) -> QueryExtractionResponseV1:
        settings = self._settings
        try:
            if (
                type(ontology.checksum) is not str
                or ontology.checksum != settings.ontology_checksum
            ):
                raise ValueError("runtime ontology differs from configured ontology")
        except (AttributeError, TypeError, ValueError):
            raise QueryExtractorClientError(
                QueryExtractorFailureReason.EXTRACTOR_PROVENANCE
            ) from None
        request = QueryExtractionRequestV1(
            schema_version=QUERY_EXTRACTION_REQUEST_SCHEMA_VERSION,
            query=query,
            ontology_checksum=settings.ontology_checksum,
            max_query_utf8_bytes=settings.max_query_utf8_bytes,
            max_query_code_points=settings.max_query_code_points,
            max_spans=settings.max_spans,
        )
        body = canonical_query_extraction_request_bytes(request)
        if len(body) > settings.max_request_body_bytes:
            raise ValueError("query request exceeds the configured body cap")
        remaining = deadline - self._monotonic()
        if remaining <= 0.0:
            raise QueryExtractorClientError(
                QueryExtractorFailureReason.EXTRACTOR_TIMEOUT
            )
        timeout = min(remaining, settings.timeout_ms / 1000.0)
        try:
            wire = self._request_once(
                url=settings.url,
                headers={
                    "Authorization": (
                        "Bearer " + settings.bearer_token.get_secret_value()
                    ),
                    "Content-Type": "application/json",
                },
                body=body,
                timeout_seconds=timeout,
                max_response_body_bytes=settings.max_response_body_bytes,
            )
        except TimeoutError:
            raise QueryExtractorClientError(
                QueryExtractorFailureReason.EXTRACTOR_TIMEOUT
            ) from None
        except (OSError, URLError):
            raise QueryExtractorClientError(
                QueryExtractorFailureReason.EXTRACTOR_PROVENANCE
            ) from None
        if wire.status in {401, 403}:
            raise QueryExtractorClientError(QueryExtractorFailureReason.EXTRACTOR_AUTH)
        if wire.status != 200 or len(wire.body) > settings.max_response_body_bytes:
            raise QueryExtractorClientError(
                QueryExtractorFailureReason.EXTRACTOR_PROVENANCE
            )
        try:
            response = parse_query_extraction_response(wire.body)
            expected = (
                settings.model_identifier,
                settings.model_revision,
                settings.schema_version,
                settings.schema_checksum,
                settings.ontology_checksum,
                settings.build_hash,
            )
            actual = (
                response.provenance.model_identifier,
                response.provenance.model_revision,
                response.provenance.schema_version,
                response.provenance.schema_checksum,
                response.provenance.ontology_checksum,
                response.provenance.build_hash,
            )
            if actual != expected or len(response.spans) > settings.max_spans:
                raise ValueError("extractor provenance or span cap mismatch")
            allowed_types = set(ontology.entity_types)
            if any(span.ontology_type not in allowed_types for span in response.spans):
                raise ValueError("extractor returned a foreign ontology type")
            reconstruct_entity_texts(query=query, response=response)
        except (TypeError, ValueError):
            raise QueryExtractorClientError(
                QueryExtractorFailureReason.EXTRACTOR_PROVENANCE
            ) from None
        return response


__all__ = [
    "QueryExtractorClient",
    "QueryExtractorClientError",
    "QueryExtractorHTTPResponse",
    "reconstruct_entity_texts",
]
