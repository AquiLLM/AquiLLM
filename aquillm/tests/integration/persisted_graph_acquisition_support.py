"""Actual selected turn, seed SQL, gateway factory and G3."""
from dataclasses import dataclass, field
import json
from time import monotonic
from types import SimpleNamespace

from asgiref.sync import async_to_sync
from channels.db import database_sync_to_async

from apps.chat.services.rag_turn import preservation_turn
from apps.documents.models import TextChunk
from apps.documents.services.chunk_search_lifetime import enter_search
from apps.documents.services.hybrid_graph_dependencies import build_hybrid_graph_dependencies
from apps.knowledge_graph.retrieval import graph_runtime_execution_v2 as graph
from lib.knowledge_graph.query_extractor.client import QueryExtractorClient, QueryExtractorHTTPResponse
from lib.knowledge_graph.query_extractor.contracts import (
    QueryEntitySpanV1, QueryExtractionResponseV1, QueryExtractorProvenanceV1,
    canonical_query_extraction_response_bytes,
)


def install_query_transport(monkeypatch, entity=("model", "Orion")):
    """Return exact synthetic model bytes through the real validating client."""
    from apps.knowledge_graph.retrieval import production_runtime

    def client(config):
        def request(**kwargs):
            body = json.loads(kwargs["body"])
            query = body["query"]
            entity_type, label = entity
            start = query.index(label)
            provenance = QueryExtractorProvenanceV1(
                config.model_identifier, config.model_revision,
                config.schema_version, config.schema_checksum,
                config.ontology_checksum, config.build_hash,
            )
            response = QueryExtractionResponseV1(
                provenance, len(query.encode()), len(query),
                (QueryEntitySpanV1(entity_type, start, start + len(label), 1.0),),
            )
            return QueryExtractorHTTPResponse(
                200, canonical_query_extraction_response_bytes(response)
            )
        return QueryExtractorClient(config, request_once=request)

    monkeypatch.setattr(production_runtime, "QueryExtractorClient", client)


@dataclass
class AcquisitionObservation:
    references: tuple = ()
    inputs: tuple = ()
    scope: object = None
    representation: object = None
    authorization: object = None
    budget: object = None
    after_swap_remaining_ms: int | None = None
    calls: list = field(default_factory=list)


def acquire_graph(source, gateway, monkeypatch, settings, *, barrier=None,
                  documents=None, principal=None, query="What does Orion use?",
                  query_entity=("model", "Orion")):
    document = source.document
    documents = (document,) if documents is None else tuple(documents)
    consumer = SimpleNamespace(user=principal or document.ingested_by,
        col_ref=SimpleNamespace(collections=sorted({d.collection_id for d in documents})))
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source")
    settings.KG_TOPOLOGY_GATEWAY_URL = gateway._endpoint.rsplit("/v1/", 1)[0]
    install_query_transport(monkeypatch, query_entity)
    observed = AcquisitionObservation()
    original = graph.load_selection_inputs_v2

    def load(**kwargs):
        observed.calls.append("hydration")
        observed.references = kwargs["references"]
        assert observed.references, "the real frontier must yield references"
        assert kwargs["authorization"] is observed.authorization
        assert kwargs["budget"] is observed.budget
        assert kwargs["selected"].representation_snapshot is observed.representation
        if barrier is not None:
            barrier()
            observed.budget.check_active()
            observed.after_swap_remaining_ms = observed.budget.remaining_ms()
            assert observed.after_swap_remaining_ms >= 1000
            assert kwargs["deadline"] - monotonic() >= 1, "swap must leave useful original time"
        observed.inputs = original(**kwargs)
        if barrier is not None:
            observed.budget.check_active()
        return observed.inputs

    async def selected_turn():
        async with preservation_turn(consumer, max_func_calls=3) as runtime:
            assert runtime is not None
            observed.authorization = runtime.authorization
            observed.budget = runtime.budget

            def execute():
                lifetime, scope = enter_search(
                    TextChunk, documents, runtime.authorization, query, False
                )
                with lifetime:
                    observed.scope = scope
                    observed.representation = scope.representation
                    dependencies = build_hybrid_graph_dependencies(authorization=runtime.authorization)
                    assert dependencies is not None
                    result = dependencies.runtime.run_v2(
                        query=query, authorization=runtime.authorization,
                        representation_snapshot=scope.representation, budget=runtime.budget,
                        settings=dependencies.settings, deadline=monotonic() + 5,
                        requested_document_ids=scope.requested_document_ids,
                    )
                    assert scope.representation is observed.representation
                    assert scope.runtime is runtime
                    runtime.budget.check_active()
                    if barrier is not None:
                        runtime.budget.check_active()
                        assert not {"deadline", "frontier_budget"}.intersection(result.handoff.coverage.reasons)
                    return result
            return await database_sync_to_async(execute)()

    with monkeypatch.context() as probe:
        probe.setattr(graph, "load_selection_inputs_v2", load)
        result = async_to_sync(selected_turn)()
    return result, observed
