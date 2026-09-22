"""Scoped score handoff tests for the chunk-search tool seam."""

from types import SimpleNamespace
from unittest.mock import patch

from django.test import override_settings

from apps.documents.services.chunk_rerank_results import (
    PassageScore,
    RerankScoreSet,
    ScoredRerankResult,
    fingerprint_text,
)
from apps.documents.services.chunk_search import (
    materialize_and_rerank_candidates,
    text_chunk_search,
)
from apps.documents.services.chunk_search_score_handoff import (
    score_set_for_authorized_rows,
)
from apps.documents.tests.test_chunk_search_graph_overlay import (
    _DOC_A,
    _DOC_B,
    _model,
    _snapshot,
)


def _rows():
    return tuple(
        SimpleNamespace(pk=pk, doc_id=_DOC_A, chunk_number=pk, content=f"chunk-{pk}")
        for pk in (1, 2)
    )


def _scores(rows):
    return RerankScoreSet(
        "v2",
        "query",
        "scorer",
        "pool",
        "pointwise",
        "complete",
        (1, 2),
        tuple(
            PassageScore(
                row.pk,
                row.doc_id,
                row.chunk_number,
                fingerprint_text(row.content),
                "effective",
                float(row.pk),
            )
            for row in rows
        ),
    )


def test_scored_materializer_returns_authorized_rows_with_score_metadata(monkeypatch):
    rows = _rows()
    score_set = _scores(rows)
    monkeypatch.setattr(
        "apps.documents.services.chunk_search.rerank_chunks_scored",
        lambda *_args: ScoredRerankResult((2, 1), score_set),
    )
    model, _ = _model([])
    ranking = materialize_and_rerank_candidates(
        model,
        "query",
        2,
        rows,
        authorized_scope=None,
        force_complete_rerank=True,
        capture_scores=True,
    )
    assert [row.pk for row in ranking.ranked_results] == [2, 1]
    assert ranking.score_set is score_set


def test_invalid_scored_order_falls_back_without_scores(monkeypatch):
    rows = _rows()
    monkeypatch.setattr(
        "apps.documents.services.chunk_search.rerank_chunks_scored",
        lambda *_args: ScoredRerankResult((999,), _scores(rows)),
    )
    monkeypatch.setattr(
        "apps.documents.services.chunk_search._fallback_rerank",
        lambda _model, candidates, _top_k: list(candidates)[:1],
    )
    model, _ = _model([])
    ranking = materialize_and_rerank_candidates(
        model,
        "query",
        1,
        rows,
        authorized_scope=None,
        capture_scores=True,
    )
    assert [row.pk for row in ranking.ranked_results] == [1]
    assert ranking.score_set is None


def test_score_handoff_removes_unrelated_and_stale_score_identities():
    kept = SimpleNamespace(pk=1, doc_id=_DOC_A, chunk_number=0, content="current")
    stale = SimpleNamespace(pk=2, doc_id=_DOC_A, chunk_number=1, content="current")
    score_set = RerankScoreSet(
        "v2",
        "query",
        "scorer",
        "pool",
        "pointwise",
        "complete",
        (1, 2, 3),
        (
            PassageScore(1, _DOC_A, 0, fingerprint_text("current"), "effective", 0.9),
            PassageScore(2, _DOC_A, 1, fingerprint_text("old"), "effective", 0.8),
            PassageScore(3, _DOC_B, 2, fingerprint_text("hidden"), "effective", 0.7),
        ),
    )
    filtered = score_set_for_authorized_rows(score_set, (kept, stale))
    assert filtered.candidate_order == (1,)
    assert [score.chunk_pk for score in filtered.scores] == [1]


@override_settings(KG_OVERLAY_ENABLED=False, RAG_CACHE_ENABLED=False)
@patch("apps.documents.services.chunk_search.collect_hybrid_candidate_snapshot")
@patch("aquillm.utils.get_embedding", return_value=(0.1, 0.2))
def test_adaptive_search_transports_only_final_ranked_score(
    _mock_embed,
    mock_collect,
    monkeypatch,
):
    rows = _rows()
    mock_collect.return_value = _snapshot(baseline=rows)
    model, _ = _model([])
    monkeypatch.setenv("RAG_EVIDENCE_SELECTION_MODE", "adaptive")
    score_set = _scores(rows)
    monkeypatch.setattr(
        "apps.documents.services.chunk_search.rerank_chunks_scored",
        lambda *_args: ScoredRerankResult((2,), score_set),
    )
    returned = text_chunk_search(
        model, "query", 1, [SimpleNamespace(id=_DOC_A, collection_id=7)]
    )
    assert len(returned) == 4
    assert [row.pk for row in returned[2]] == [2]
    assert returned[3]["_score_set"]["candidate_order"] == [2]
    assert [score["chunk_pk"] for score in returned[3]["_score_set"]["scores"]] == [2]
