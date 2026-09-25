# ruff: noqa: F401
"""End-to-end client and service ontology provenance parity."""

from lib.knowledge_graph.tests.test_query_extractor_service_dynamic import (
    DIGEST,
    EMOJI,
    EntityCandidate,
    ExtractionBatchResult,
    SimpleNamespace,
    _call,
    _custom_ontology,
    _settings,
    asyncio,
    replace,
    runtime,
    service,
    time,
)


def test_real_client_and_service_agree_on_custom_schema_provenance(monkeypatch):
    from lib.knowledge_graph.query_extractor.client import (
        QueryExtractorClient,
        QueryExtractorHTTPResponse,
    )

    ontology = _custom_ontology()

    class CustomBackend:
        def extract_entities_batch(self, texts, *, ontology):
            assert set(ontology.entity_types) == {"audit_entity"}
            return (
                ExtractionBatchResult(
                    entities=(EntityCandidate("audit_entity", EMOJI, 1, 2, 0.75),),
                    relations=(),
                    diagnostics=(),
                ),
            )

    pinned = service.QueryExtractorRuntime(
        settings=_settings(),
        ontology=SimpleNamespace(checksum=DIGEST),
        backend=CustomBackend(),
    )
    monkeypatch.setattr(service, "_get_runtime", lambda *_args: pinned)

    def request_once(**kwargs):
        assert kwargs["url"] == "http://extractor:8080/v1/extract"
        sent = asyncio.run(
            _call(
                path="/v1/extract",
                body=kwargs["body"],
                authorization=kwargs["headers"]["Authorization"].encode(),
            )
        )
        return QueryExtractorHTTPResponse(sent[0]["status"], sent[1]["body"])

    client = QueryExtractorClient(
        replace(
            _settings(),
            ontology_checksum=ontology.checksum,
            url="http://extractor:8080",
        ),
        request_once=request_once,
    )
    response = client.extract(
        query=f"A{EMOJI}B",
        ontology=ontology,
        deadline=time.monotonic() + 1,
    )
    assert response.provenance.ontology_checksum == ontology.checksum
    assert response.spans[0].ontology_type == "audit_entity"
