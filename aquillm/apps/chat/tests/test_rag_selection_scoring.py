"""Comparable score preparation, with external scoring replaced by small fakes."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID

from apps.collections.services.retrieval_authorization import (
    OpaquePrincipalReference,
    bind_retrieval_reauthorization_capability,
    freeze_retrieval_authorization_context,
)
from apps.documents.services.chunk_rerank_results import (
    PassageScore,
    RerankScoreSet,
    fingerprint_pair,
    fingerprint_pool,
    fingerprint_text,
)

DOC = UUID("11111111-1111-4111-8111-111111111111")


def _chunk(pk: int, content: str):
    return SimpleNamespace(pk=pk, doc_id=DOC, chunk_number=pk - 1, content=content)


class PointwiseScorer:
    scoring_kind = "pointwise"
    scorer_fingerprint = "scorer-v1"

    def __init__(self):
        self.calls = []

    def prepare_pair(self, query, chunk):
        return query[:12], chunk.content[:20]

    def score_pair(self, pair, timeout_seconds):
        self.calls.append((pair, timeout_seconds))
        return float(len(pair[1])), pair


def test_pointwise_scoring_uses_each_canonical_pair_independent_of_pool_order():
    from apps.documents.services.chunk_rerank_scoring import score_missing_pairs

    scorer = PointwiseScorer()
    query = "a very long question with distinguishing words"
    first = _chunk(1, "first passage is long")
    second = _chunk(2, "second")
    forward = score_missing_pairs(
        query=query,
        chunks=(first, second),
        scorer=scorer,
        deadline=100.0,
        clock=lambda: 0.0,
    )
    reversed_set = score_missing_pairs(
        query=query,
        chunks=(second, first),
        scorer=scorer,
        deadline=100.0,
        clock=lambda: 0.0,
    )
    assert forward.status == reversed_set.status == "complete"
    forward_pairs = {
        score.chunk_pk: score.effective_pair_fingerprint for score in forward.scores
    }
    reversed_pairs = {
        score.chunk_pk: score.effective_pair_fingerprint
        for score in reversed_set.scores
    }
    assert forward_pairs == reversed_pairs
    assert forward.scores[1].effective_pair_fingerprint == fingerprint_pair(
        query[:12], "second"
    )


def test_successful_retry_pair_is_provenance_not_canonical_pair():
    from apps.documents.services.chunk_rerank_scoring import score_missing_pairs

    class RetryingScorer(PointwiseScorer):
        def score_pair(self, pair, timeout_seconds):
            return -2.0, (pair[0], "successful shorter retry")

    result = score_missing_pairs(
        query="question",
        chunks=(_chunk(1, "original long content"),),
        scorer=RetryingScorer(),
        deadline=100.0,
        clock=lambda: 0.0,
    )
    assert result.scores[0].value == -2.0
    assert result.scores[0].effective_pair_fingerprint == fingerprint_pair(
        "question", "successful shorter retry"
    )
    assert result.scores[0].effective_pair_fingerprint != fingerprint_pair(
        "question", "original long content"
    )


def test_deadline_prevents_pair_submission_and_incomplete_results_are_unavailable():
    from apps.documents.services.chunk_rerank_scoring import score_missing_pairs

    scorer = PointwiseScorer()
    result = score_missing_pairs(
        query="question",
        chunks=(_chunk(1, "one"),),
        scorer=scorer,
        deadline=0.0,
        clock=lambda: 0.0,
    )
    assert result.status == "unavailable"
    assert result.scores == ()
    assert scorer.calls == []


def test_pair_cap_rejects_oversized_new_pool_without_partial_scoring():
    from apps.documents.services.chunk_rerank_scoring import score_missing_pairs

    scorer = PointwiseScorer()
    result = score_missing_pairs(
        query="question",
        chunks=tuple(_chunk(pk, "x") for pk in range(1, 47)),
        scorer=scorer,
        deadline=100.0,
        clock=lambda: 0.0,
    )
    assert result.status == "unavailable"
    assert scorer.calls == []


class Policy:
    policy_version = "v1"
    policy_checksum = "a" * 64

    def __init__(self):
        self.allowed = ((1, DOC),)

    def opaque_principal_reference(self, *, principal, database_alias):
        return OpaquePrincipalReference("b" * 64)

    def current_authorized_document_scope(
        self, *, principal, database_alias, selected_collection_ids
    ):
        return self.allowed


def _authorization(policy):
    principal = object()
    return freeze_retrieval_authorization_context(
        principal=principal,
        database_alias="default",
        policy=policy,
        selected_collection_ids=(1,),
        selected_document_ids=(DOC,),
        reauthorization_capability=bind_retrieval_reauthorization_capability(
            principal=principal, policy=policy
        ),
    )


def _pool(chunks, score_sets=()):
    from apps.chat.services.rag_retrieval import FusedRetrievalPool

    rows = tuple(
        {
            "rank": index,
            "chunk_id": chunk.pk,
            "doc_id": str(chunk.doc_id),
            "chunk": chunk.chunk_number,
            "text": chunk.content,
            "citation": f"[doc:{chunk.doc_id} chunk:{chunk.pk}]",
        }
        for index, chunk in enumerate(chunks, start=1)
    )
    return FusedRetrievalPool(
        rows,
        tuple(score_sets),
        tuple((row["citation"], 1 / (60 + index)) for index, row in enumerate(rows, 1)),
    )


def _score_set(query, chunks, values, scorer_fp="scorer-v1", kind="pointwise"):
    pairs = tuple((query[:12], chunk.content[:20]) for chunk in chunks)
    return RerankScoreSet(
        "v2",
        fingerprint_text(query),
        scorer_fp,
        fingerprint_pool(
            tuple(
                (chunk.pk, fingerprint_text(chunk.content), fingerprint_pair(*pair))
                for chunk, pair in zip(chunks, pairs)
            )
        ),
        kind,
        "complete",
        tuple(chunk.pk for chunk in chunks),
        tuple(
            PassageScore(
                chunk.pk,
                chunk.doc_id,
                chunk.chunk_number,
                fingerprint_text(chunk.content),
                fingerprint_pair(*pair),
                value,
            )
            for chunk, pair, value in zip(chunks, pairs, values)
        ),
    )


def test_primary_pointwise_scores_reuse_without_new_work_and_normalize_negative():
    from apps.chat.services.rag_selection_scoring import prepare_selection_candidates

    chunks = (_chunk(1, "first"), _chunk(2, "second"))
    scorer = PointwiseScorer()
    result = prepare_selection_candidates(
        pool=_pool(chunks, (_score_set("question", chunks, (-3.0, -1.0)),)),
        primary_query="question",
        authorization=_authorization(Policy()),
        deadline=100.0,
        allow_new_scores=True,
        scorer=scorer,
        chunk_loader=lambda _auth, _ids: chunks,
        clock=lambda: 0.0,
    )
    assert result.score_status == "model"
    assert result.reused_pairs == 2 and result.new_pairs == 0
    assert [candidate.relevance for candidate in result.candidates] == [0.0, 1.0]
    assert scorer.calls == []


def test_clause_scores_are_not_reused_and_only_missing_primary_pairs_are_scored():
    from apps.chat.services.rag_selection_scoring import prepare_selection_candidates

    chunks = (_chunk(1, "first"), _chunk(2, "second"))
    scorer = PointwiseScorer()
    result = prepare_selection_candidates(
        pool=_pool(
            chunks,
            (
                _score_set("clause query", chunks, (0.8, 0.9)),
                _score_set("question", chunks[:1], (0.3,)),
            ),
        ),
        primary_query="question",
        authorization=_authorization(Policy()),
        deadline=100.0,
        allow_new_scores=True,
        scorer=scorer,
        chunk_loader=lambda _auth, _ids: chunks,
        clock=lambda: 0.0,
    )
    assert result.score_status == "model"
    assert result.reused_pairs == 1 and result.new_pairs == 1
    assert len(scorer.calls) == 1
    assert scorer.calls[0][0] == ("question", "second")


def test_mixed_or_partial_scores_use_rank_fallback_for_whole_pool():
    from apps.chat.services.rag_selection_scoring import prepare_selection_candidates

    chunks = (_chunk(1, "first"), _chunk(2, "second"))
    result = prepare_selection_candidates(
        pool=_pool(chunks, (_score_set("question", chunks[:1], (0.9,)),)),
        primary_query="question",
        authorization=_authorization(Policy()),
        deadline=100.0,
        allow_new_scores=False,
        scorer=PointwiseScorer(),
        chunk_loader=lambda _auth, _ids: chunks,
        clock=lambda: 0.0,
    )
    assert result.score_status == "rank_fallback"
    assert [candidate.relevance for candidate in result.candidates] == [1.0, 0.0]
