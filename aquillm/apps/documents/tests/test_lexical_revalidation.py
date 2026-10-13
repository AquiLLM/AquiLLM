"""Channel withdrawal must change fallback evidence without changing its lifetime."""

from dataclasses import replace

import pytest
from django.test import override_settings

from apps.documents.models import TextChunk
from apps.documents.services import chunk_rerank, chunk_search
from apps.documents.services.chunk_rerank_results import rank_only_result_for_chunks
from apps.documents.services.chunk_search_scope import current_scope
from apps.documents.services.source_loading import source_runtime_scope
from .lexical_owner_support import unchunked as legacy
from apps.documents.tests.test_embedding_generation_query import runtime_for
from apps.documents.tests.test_lexical_publication import publish
from apps.documents.tests.test_lexical_relevance import replace_chunks

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("phase", ["materialization", "provider", "final"])
@pytest.mark.parametrize("mutation", ["none", "title", "kind"])
@override_settings(
    RAG_FULL_TEXT_ENABLED=True,
    RAG_RRF_ENABLED=True,
    RAG_PROTECTED_EXACT_ENABLED=False,
    RAG_TRIGRAM_SIMILARITY_MIN=0.01,
    KG_OVERLAY_ENABLED=False,
)
def test_mixed_channel_withdrawal_reorders_only_unavailable_evidence(
    legacy, monkeypatch, capture, phase, mutation
):
    a, b = replace_chunks(
        legacy,
        [
            "assay apparatus spectrometer calibration lengthy explanation",
            "assa",
        ],
    )
    publish(legacy)
    runtime, _ = runtime_for(legacy)
    original_collect = chunk_search.collect_hybrid_candidate_snapshot
    original_ranking = chunk_search.materialize_and_rerank_candidates
    original_diagnostics = chunk_search.completed_search_diagnostics
    before, final, attempts, scopes = [], [], [], []
    monkeypatch.setenv(
        "RAG_EVIDENCE_SELECTION_MODE", "adaptive" if capture else "legacy"
    )
    monkeypatch.setattr(chunk_search, "load_query_embedding", lambda _: None)

    def mutate():
        if mutation == "title":
            type(legacy[0]).objects.filter(pk=legacy[0].pk).update(title="changed")
        elif mutation == "kind":
            TextChunk.objects.filter(pk=a.pk).update(source_kind="table")

    def collect(*args, **kwargs):
        snapshot = original_collect(*args, **kwargs)
        assert snapshot.trigram_chunk_ids == (b.pk, a.pk)
        assert snapshot.full_text_chunk_ids == (a.pk,)
        assert snapshot.exact_chunk_ids == ()
        assert not any(c.protected_exact for c in snapshot.fusion_v2.candidates)
        assert snapshot.baseline_candidates[0].pk == a.pk
        before.append(snapshot)
        scope = current_scope()
        scopes.append(
            (scope, scope.representation, scope.authorization, scope.runtime.budget)
        )
        if phase == "materialization":
            mutate()
        return snapshot

    def unavailable(*args, **kwargs):
        attempts.append(True)
        if phase == "provider":
            mutate()
        raise TimeoutError("synthetic unavailable provider")

    def ranking(*args, **kwargs):
        result = original_ranking(*args, **kwargs)
        if phase == "final":
            mutate()
        return result

    def diagnostics(*args, **kwargs):
        final.append(args[2])
        scope, representation, authorization, budget = scopes[0]
        assert current_scope() is scope
        assert scope.representation is representation
        assert scope.authorization is authorization
        assert scope.runtime.budget is budget
        assert args[2].fusion_v2 == scope.fusion_v2
        return original_diagnostics(*args, **kwargs)

    monkeypatch.setattr(chunk_search, "collect_hybrid_candidate_snapshot", collect)
    monkeypatch.setattr(chunk_search, "materialize_and_rerank_candidates", ranking)
    monkeypatch.setattr(chunk_search, "completed_search_diagnostics", diagnostics)
    monkeypatch.setattr(chunk_rerank, "rerank_via_local_vllm", unavailable)
    monkeypatch.setattr(chunk_rerank, "rerank_via_local_vllm_scored", unavailable)
    with source_runtime_scope(runtime):
        result = chunk_search.text_chunk_search(
            TextChunk,
            "assay",
            1,
            [legacy[0]],
            authorization_context=runtime.authorization,
        )
    assert len(attempts) == 1
    expected = a if mutation == "none" else b
    assert [r.pk for r in result[2]] == [expected.pk]
    snapshot = final[0]
    assert snapshot.fusion_v2.scope_key == before[0].fusion_v2.scope_key
    assert (
        snapshot.fusion_v2.order_checksum == before[0].fusion_v2.order_checksum
    ) is (mutation == "none")
    assert snapshot.full_text_chunk_ids == ((a.pk,) if mutation == "none" else ())
    coverage = result[3]["full_text_coverage"]
    assert coverage["status"] == ("ready" if mutation == "none" else "incomplete")
    assert coverage["indexed_count"] == (
        2 if mutation == "none" else 0 if mutation == "title" else 1
    )
    votes = {r.channel for c in snapshot.fusion_v2.candidates for r in c.channel_ranks}
    assert ("full_text" in votes) is (mutation == "none")
    if capture:
        sidecar = result[3]["_score_set"]
        expected_score = rank_only_result_for_chunks("assay", (expected,), 1).score_set
        assert sidecar["candidate_order"] == [expected.pk]
        assert sidecar["pool_fingerprint"] == expected_score.pool_fingerprint
        assert sidecar["scores"] == []
        assert sidecar["status"] == "unavailable"


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("status", ["complete", "unknown"])
@override_settings(
    RAG_FULL_TEXT_ENABLED=True,
    RAG_RRF_ENABLED=True,
    RAG_PROTECTED_EXACT_ENABLED=False,
    RAG_TRIGRAM_SIMILARITY_MIN=0.01,
    KG_OVERLAY_ENABLED=False,
)
def test_changed_fts_retains_known_provider_order_but_not_unknown_order(
    legacy, monkeypatch, capture, status
):
    from apps.documents.services.chunk_rerank_parse import ordered_queryset_from_ids
    from apps.documents.services.chunk_rerank_results import (
        PassageScore,
        fingerprint_pair,
        fingerprint_text,
    )

    a, b = replace_chunks(
        legacy,
        [
            "assay apparatus spectrometer calibration lengthy explanation",
            "assa",
        ],
    )
    publish(legacy)
    runtime, _ = runtime_for(legacy)
    calls = []
    monkeypatch.setenv(
        "RAG_EVIDENCE_SELECTION_MODE", "adaptive" if capture else "legacy"
    )
    monkeypatch.setattr(chunk_search, "load_query_embedding", lambda _: None)

    def provider(model, query, rows, top_k, **kwargs):
        calls.append(True)
        type(legacy[0]).objects.filter(pk=legacy[0].pk).update(title="changed")
        if not capture:
            return ordered_queryset_from_ids(model, (a.pk,), status=status)
        result = rank_only_result_for_chunks(query, rows, top_k)
        scores = tuple(
            PassageScore(
                row.pk,
                row.doc_id,
                row.chunk_number,
                fingerprint_text(row.content),
                fingerprint_pair(query, row.content),
                1.0 if row.pk == a.pk else 0.5,
            )
            for row in rows
        )
        return replace(
            result,
            ranked_ids=(a.pk,),
            score_set=replace(
                result.score_set,
                status=status,
                scoring_kind="pointwise",
                scores=scores,
            ),
        )

    monkeypatch.setattr(chunk_search, "rerank_chunks", provider)
    monkeypatch.setattr(chunk_search, "rerank_chunks_scored", provider)
    with source_runtime_scope(runtime):
        result = chunk_search.text_chunk_search(
            TextChunk,
            "assay",
            1,
            [legacy[0]],
            authorization_context=runtime.authorization,
        )
    expected = a if status == "complete" else b
    assert [row.pk for row in result[2]] == [expected.pk]
    assert calls == [True]
    assert result[3]["full_text_coverage"]["indexed_count"] == 0
    if capture:
        sidecar = result[3]["_score_set"]
        assert sidecar["candidate_order"] == [expected.pk]
        assert sidecar["status"] == (
            "complete" if status == "complete" else "unavailable"
        )
        assert [score["chunk_pk"] for score in sidecar["scores"]] == (
            [a.pk] if status == "complete" else []
        )
