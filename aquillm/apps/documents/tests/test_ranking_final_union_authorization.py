"""One closed union authorizes ranking while preserving fresh final authority."""

from dataclasses import replace

import pytest
from django.db import connection
from django.test.utils import CaptureQueriesContext

from apps.collections.models import CollectionPermission
from apps.documents.models import RawTextDocument
from apps.documents.services import chunk_search_scope
from apps.documents.services.chunk_rerank_results import (
    PassageScore,
    RerankScoreSet,
    fingerprint_text,
)
from apps.documents.services.chunk_search import CandidateRankingResult
from apps.documents.services.chunk_search_authorization import (
    authorized_search_rows,
    finalize_ranking,
)
from apps.documents.services.chunk_search_candidates import HybridCandidateSnapshot
from apps.documents.services.chunk_search_scoped_rows import scoped_rows
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.source_loading import source_runtime_scope
from lib.embeddings.tests.test_document_provider import transport as transport

from .test_source_metadata_closed_operation import captured

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.fixture
def handoff(transport, monkeypatch, settings):
    case, scope, queryset, _ = captured(3, "none", transport, monkeypatch, settings)
    runtime = scope.runtime
    token = chunk_search_scope._SCOPE.set(scope)
    try:
        with source_runtime_scope(runtime):
            rows = scoped_rows(queryset, scope, lexical=True)
            assert len(rows) == 3
            scores = RerankScoreSet(
                "v2",
                "query",
                "scorer",
                "pool",
                "pointwise",
                "complete",
                tuple(row.pk for row in rows),
                tuple(
                    PassageScore(
                        row.pk,
                        row.doc_id,
                        row.chunk_number,
                        fingerprint_text(row.content),
                        "pair",
                        float(index),
                    )
                    for index, row in enumerate(rows)
                ),
            )
            snapshot = HybridCandidateSnapshot(
                documents=case.docs,
                vector_results=(),
                trigram_results=rows,
                exact_results=(),
                vector_chunk_ids=(),
                trigram_chunk_ids=tuple(row.pk for row in rows),
                exact_chunk_ids=(),
                baseline_candidates=rows,
                graph_seeds=(),
                exact_terms=(),
                vector_error=None,
                vector_ms=0.0,
                trigram_ms=0.0,
                exact_ms=0.0,
                pre_dedupe_count=3,
                graph_seed_error=False,
            )
            ranking = CandidateRankingResult(
                rows, (), (rows[2], rows[0]), 0, 0.0, 0.0, scores, "complete"
            )
            yield case, scope, rows, snapshot, ranking
    finally:
        chunk_search_scope.reset_search_scope(token)
        runtime.budget.close("final_union_test_finished")


def authorize(scope, rows):
    return authorized_search_rows(
        rows, authorization_context=scope.authorization, hybrid_requested=False
    )


@pytest.mark.parametrize("empty", [False, True])
def test_successful_ranking_uses_one_real_closed_union_without_changing_rows_or_scores(
    handoff, empty
):
    _, scope, rows, snapshot, ranking = handoff
    if empty:
        ranking = replace(ranking, ranked_results=())
    original_scope = scope.runtime._frozen_retrieval_scope
    with CaptureQueriesContext(connection) as reads:
        final, channels, scores = finalize_ranking(
            snapshot,
            ranking,
            authorize=lambda current: authorize(scope, current),
            query="query",
            top_k=2,
        )
    assert channels[0] == ()
    assert tuple(channels[1]) == rows
    assert tuple(channels[2]) == (() if empty else (rows[2], rows[0]))
    assert all(
        left is right for left, right in zip(channels[2], ranking.ranked_results)
    )
    assert final.baseline_candidates == rows
    assert scores.candidate_order == (() if empty else (rows[0].pk, rows[2].pk))
    assert tuple(score.chunk_pk for score in scores.scores) == scores.candidate_order
    assert scope.runtime._frozen_retrieval_scope is original_scope
    # Actual full source reads: entry/final for union authorization, followed by
    # the independent entry/final dense/FTS finalization checks. No rank-only pass.
    source_reads = [
        row for row in reads if "UNION ALL" in row["sql"] and "sha256" in row["sql"]
    ]
    assert len(source_reads) == 4


def test_changed_captured_body_is_withdrawn_with_its_score_at_final_union(handoff):
    _, scope, rows, snapshot, ranking = handoff
    changed = []

    def late_authorize(current):
        if not changed:
            # Represented database chunks are immutable. A changed hydrated
            # object still must fail the captured and actual metadata checks.
            rows[2].content = "Changed hydrated body"
            changed.append(rows[2].pk)
        return authorize(scope, current)

    final, channels, scores = finalize_ranking(
        snapshot, ranking, authorize=late_authorize, query="query", top_k=2
    )
    assert channels[2] == [rows[0]]
    assert final.trigram_chunk_ids == (rows[0].pk, rows[1].pk)
    assert scores.candidate_order == (rows[0].pk,)
    assert tuple(score.chunk_pk for score in scores.scores) == (rows[0].pk,)


@pytest.mark.parametrize("change", ["source", "permission", "empty_permission"])
def test_late_original_authority_change_blocks_handoff_even_without_ranked_rows(
    handoff, change
):
    case, scope, _, snapshot, ranking = handoff
    if change == "empty_permission":
        ranking = replace(ranking, ranked_results=(), combined_candidates=())
        snapshot = replace(snapshot, trigram_results=(), baseline_candidates=())
    changed, published = [], []

    def late_authorize(current):
        if not changed:
            if change == "source":
                assert (
                    RawTextDocument.objects.filter(pk=case.docs[1].pk).update(
                        full_text="Changed source with stale stored hash"
                    )
                    == 1
                )
            else:
                CollectionPermission.objects.filter(
                    user=case.owner, collection=case.collection
                ).delete()
            changed.append(change)
        return authorize(scope, current)

    with pytest.raises(SourceScopeUnavailable):
        published.append(
            finalize_ranking(
                snapshot, ranking, authorize=late_authorize, query="query", top_k=2
            )
        )
    assert changed == [change]
    assert published == []
