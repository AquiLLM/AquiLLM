"""The retained extended consumer freshly checks dense and full-text seeds."""

from types import SimpleNamespace

import pytest
from django.conf import settings

from apps.collections.tests import native_closed_seed_policy_fixtures as fixtures
from apps.documents.models import EmbeddingGeneration, TextChunk
from apps.documents.services import embedding_index_store as store
from apps.documents.services.chunk_search_candidates import (
    collect_hybrid_candidate_snapshot,
)
from apps.documents.services.representation_contracts import (
    DocumentRefV1,
    SourceScopeUnavailable,
)
from apps.documents.tests.test_embedding_generation import stage
from apps.documents.tests.test_lexical_publication import publish
from apps.knowledge_graph.models import CollectionEntityDocumentLink
from apps.knowledge_graph.retrieval import closed_seed_projection_batch as batch
from apps.knowledge_graph.retrieval.expansion import get_graph_expansion_config
from apps.knowledge_graph.tests.test_native_ontology_proof_postgres import (
    extractor_runtime,
)
from apps.knowledge_graph.tests.test_task16_seed_prefix import allocate
from lib.embeddings.document_contracts import digest
from lib.embeddings.document_provider import observe_query_embedding
from lib.embeddings.tests.test_document_provider import transport as transport

pytestmark = pytest.mark.django_db(transaction=True, databases="__all__")


@pytest.mark.parametrize("channel", ["full_text", "dense", "full_text_dense"])
@pytest.mark.parametrize("boundary", ["batch", "direct"])
def test_actual_extended_consumer_retains_fresh_lane_fallback_and_rejection(
    monkeypatch,
    transport,
    channel,
    boundary,
):
    leases, chunk_ids, mutations = [], [], []
    original_sources = fixtures.small_native_scope

    def sources(f):
        with monkeypatch.context() as patch:
            # Reuse the existing three-mention corpus within the provider's batch cap.
            patch.setattr(fixtures, "pressure_rows", lambda f: None)
            original_sources(f)
        if channel != "dense":
            for document in f.documents.values():
                publish((document, None))
        document = f.documents["A"]
        ref = DocumentRefV1(
            document._meta.label_lower,
            document.pkid,
            document.id,
            document.collection_id,
            document.full_text_hash,
        )
        _, lease, _, _ = stage((document, ref), transport)
        assert (
            store.activate_embedding_generation(lease, using="default") == "activated"
        )
        leases.append(lease)
        monkeypatch.setattr(
            settings, "DOCUMENT_EMBEDDING_SPACE_KEY", transport[0].space_key
        )

    def capture_dense(active):
        f, _, source, scope, snapshot, dependencies, _, _ = active
        scope.query_digest = digest(
            {"version": "unprepared-query-v1", "query": "Atlas"}
        )
        dependencies.settings.memgraph_traversal_enabled = True
        observation = observe_query_embedding(
            "Atlas", space=transport[0], budget=source.budget
        )
        return collect_hybrid_candidate_snapshot(
            TextChunk,
            "Atlas",
            20,
            snapshot.documents,
            query_embedding=observation,
            graph_config=get_graph_expansion_config(),
            authorization_context=f.authorization,
            app_config_getter=lambda _: SimpleNamespace(
                vector_top_k=45, trigram_top_k=45
            ),
            _graph_dependencies=dependencies,
        )

    def mutate():
        if channel != "dense":
            row = TextChunk.objects.get(pk=chunk_ids[0])
            assert row.search_vector_binding
            TextChunk.objects.filter(pk=row.pk).update(
                search_vector_binding={**row.search_vector_binding, "extra": "changed"}
            )
        if channel != "full_text":
            assert (
                EmbeddingGeneration.objects.filter(pk=leases[0].generation_key).update(
                    state="failed"
                )
                == 1
            )
        mutations.append((channel, boundary))

    def runtime(graph):
        graph.settings.graph_direct_enabled = True
        monkeypatch.setattr(
            settings, "RAG_FULL_TEXT_ENABLED", channel != "dense", raising=False
        )
        monkeypatch.setattr(settings, "RAG_TRIGRAM_SIMILARITY_MIN", 1.0, raising=False)
        extractor_runtime(graph)
        original = graph.extractor_factory

        def factory(**kwargs):
            extractor = original(**kwargs)

            def extract(**kwargs):
                result = extractor.extract(**kwargs)
                if boundary == "direct":
                    mutate()
                return result

            return SimpleNamespace(extract=extract)

        graph.extractor_factory = factory

    monkeypatch.setattr(fixtures, "small_native_scope", sources)
    monkeypatch.setattr(fixtures, "capture", capture_dense)
    with fixtures.native_frame(
        monkeypatch, metadata=True, prepare_runtime=runtime
    ) as frame:
        active, owner, view, _, ledger = frame
        source, scope = active[2:4]
        assert owner._native_metadata is not None and view.graph_seeds
        assert owner.snapshot.exact_chunk_ids == owner.snapshot.trigram_chunk_ids == ()
        assert dict(owner.plan.channels)["dense"]
        if channel != "dense":
            assert dict(owner.plan.channels)["full_text"]
            assert owner.snapshot.full_text_coverage.status == "ready"
        chunk_ids.append(view.seed_references[0].pk)
        before = (
            source.budget.sources_used,
            source.cache.copy(),
            scope.bodies.copy(),
            scope.dense_rows.copy(),
            scope.fusion_v2,
        )
        original = batch.seed_query_rows

        def read(query, **kwargs):
            rows = original(query, **kwargs)
            if boundary == "batch" and query.model is CollectionEntityDocumentLink:
                assert rows
                mutate()
            return rows

        monkeypatch.setattr(batch, "seed_query_rows", read)
        if channel == "full_text":
            # A lost FTS binding retains the independently live dense seed.
            result, reasons = allocate(frame, native=True)
            seeds, mass, _ = next(iter(result.values()))
            assert not reasons and seeds and mass == 1.0
        else:
            with pytest.raises(SourceScopeUnavailable):
                allocate(frame, native=True)
        assert mutations == [(channel, boundary)]
        assert (
            source.budget.sources_used,
            source.cache,
            scope.bodies,
            scope.dense_rows,
        ) == before[:4]
        assert scope.fusion_v2 is before[4]
        assert ledger.turn_budget is owner.stage.budget and ledger.queries > 0
