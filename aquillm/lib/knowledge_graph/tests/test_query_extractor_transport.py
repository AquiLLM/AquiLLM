# ruff: noqa: E501
"""Related graph infrastructure regression scenarios."""

from lib.knowledge_graph.tests.test_query_extractor_client import (
    EMOJI,
    BytesIO,
    HTTPError,
    Ontology,
    QueryExtractorClient,
    QueryExtractorClientError,
    QueryExtractorFailureReason,
    QueryExtractorHTTPResponse,
    URLError,
    _environment,
    _response,
    client_module,
    load_query_extractor_settings,
    pytest,
)


@pytest.mark.parametrize("status", (200, 413))
def test_stdlib_transport_reads_at_most_response_cap_plus_one(
    monkeypatch, status: int
) -> None:
    maximum = 8

    class Response(BytesIO):
        def __init__(self) -> None:
            super().__init__(b"x" * 1_000)
            self.status = status
            Response.last = self

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class Opener:
        def open(self, request, *, timeout):
            del request, timeout
            if status == 200:
                return Response()
            Opener.error = HTTPError(
                "https://extractor.internal/v1/extract",
                status,
                "fixed",
                {},
                Response(),
            )
            raise Opener.error

    monkeypatch.setattr(client_module, "build_opener", lambda *_args: Opener())
    response = client_module._stdlib_request_once(
        url="https://extractor.internal/v1/extract",
        headers={},
        body=b"{}",
        timeout_seconds=0.1,
        max_response_body_bytes=maximum,
    )
    assert response.status == status
    assert response.body == b"x" * (maximum + 1)
    if status == 413:
        assert Response.last.closed



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
    client._request_once = lambda **_kwargs: (_ for _ in ()).throw(URLError(TimeoutError()))  # fmt: skip
    client._monotonic = lambda: 1.0
    with pytest.raises(QueryExtractorClientError) as exc_info:
        client.extract(query="model", ontology=Ontology(), deadline=2.0)
    assert exc_info.value.reason is QueryExtractorFailureReason.EXTRACTOR_TIMEOUT
