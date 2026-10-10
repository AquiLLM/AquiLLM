"""Dense body admission survives withdrawal of its independent FTS vote."""

from dataclasses import replace

import pytest
from django.test import override_settings

from apps.documents.models import TextChunk
from apps.documents.services import chunk_rerank, chunk_search
from apps.documents.services import embedding_index_store as store
from apps.documents.services.chunk_search_scope import current_scope
from apps.documents.services.source_loading import source_runtime_scope
from apps.documents.tests.test_embedding_generation import legacy as legacy
from apps.documents.tests.test_embedding_generation import stage
from apps.documents.tests.test_embedding_generation_query import runtime_for
from apps.documents.tests.test_lexical_publication import publish
from apps.documents.tests.test_lexical_relevance import replace_chunks
from lib.embeddings.tests.test_document_provider import transport as transport

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("capture", [False, True])
@override_settings(
    RAG_FULL_TEXT_ENABLED=True,
    RAG_RRF_ENABLED=True,
    RAG_PROTECTED_EXACT_ENABLED=False,
    RAG_TRIGRAM_SIMILARITY_MIN=1,
    KG_OVERLAY_ENABLED=False,
)
def test_title_change_withdraws_only_fts_from_verified_dense_rows(
    legacy, transport, monkeypatch, capture
):
    b, a = replace_chunks(legacy, ["assa", "assay apparatus spectrometer calibration"])
    legacy = (legacy[0], replace(legacy[1], source_hash=legacy[0].full_text_hash))
    _, lease, _, _ = stage(legacy, transport)
    assert store.activate_embedding_generation(lease, using="default") == "activated"
    publish(legacy)
    runtime, _ = runtime_for(legacy)
    original_collect = chunk_search.collect_hybrid_candidate_snapshot
    original_diagnostics = chunk_search.completed_search_diagnostics
    scopes, final, attempts = [], [], []
    monkeypatch.setenv(
        "RAG_EVIDENCE_SELECTION_MODE", "adaptive" if capture else "legacy"
    )

    def collect(*args, **kwargs):
        snapshot = original_collect(*args, **kwargs)
        assert snapshot.vector_chunk_ids == (b.pk, a.pk)
        assert snapshot.full_text_chunk_ids == (a.pk,)
        assert snapshot.exact_chunk_ids == snapshot.trigram_chunk_ids == ()
        assert not any(c.protected_exact for c in snapshot.fusion_v2.candidates)
        assert snapshot.baseline_candidates[0].pk == a.pk
        scope = current_scope()
        scopes.append((scope, scope.dense, scope.runtime.budget))
        return snapshot

    def unavailable(*args, **kwargs):
        attempts.append(True)
        type(legacy[0]).objects.filter(pk=legacy[0].pk).update(title="changed")
        raise TimeoutError("synthetic unavailable provider")

    def diagnostics(*args, **kwargs):
        final.append(args[2])
        scope, dense, budget = scopes[0]
        assert current_scope() is scope
        assert scope.dense is dense
        assert scope.runtime.budget is budget
        assert args[2].fusion_v2 == scope.fusion_v2
        return original_diagnostics(*args, **kwargs)

    monkeypatch.setattr(chunk_search, "collect_hybrid_candidate_snapshot", collect)
    monkeypatch.setattr(chunk_search, "completed_search_diagnostics", diagnostics)
    monkeypatch.setattr(chunk_rerank, "rerank_via_local_vllm", unavailable)
    monkeypatch.setattr(chunk_rerank, "rerank_via_local_vllm_scored", unavailable)
    with (
        override_settings(DOCUMENT_EMBEDDING_SPACE_KEY=transport[0].space_key),
        source_runtime_scope(runtime),
    ):
        result = chunk_search.text_chunk_search(
            TextChunk,
            "assay",
            1,
            [legacy[0]],
            authorization_context=runtime.authorization,
        )
    assert attempts == [True]
    assert [row.pk for row in result[2]] == [b.pk]
    assert final[0].vector_chunk_ids == (b.pk, a.pk)
    assert final[0].full_text_chunk_ids == ()
    assert result[3]["full_text_coverage"]["indexed_count"] == 0
    candidates = {c.chunk_id: c for c in final[0].fusion_v2.candidates}
    assert [(r.channel, r.rank) for r in candidates[a.pk].channel_ranks] == [
        ("dense", 2)
    ]
    if capture:
        assert result[3]["_score_set"]["candidate_order"] == [b.pk]
        assert result[3]["_score_set"]["status"] == "unavailable"
        assert result[3]["_score_set"]["scores"] == []
