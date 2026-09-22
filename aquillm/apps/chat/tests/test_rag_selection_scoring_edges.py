"""Late authorization changes and incompatible score shapes."""

from threading import Event
from time import monotonic

from apps.chat.services.rag_selection_scoring import prepare_selection_candidates
from apps.chat.tests.test_rag_selection_scoring import (
    PointwiseScorer,
    Policy,
    _authorization,
    _chunk,
    _pool,
    _score_set,
)


def test_permission_revoked_during_scoring_drops_candidates_without_refill():
    policy = Policy()
    chunks = (_chunk(1, "first"),)

    class RevokingScorer(PointwiseScorer):
        def score_pair(self, pair, timeout_seconds):
            policy.allowed = ()
            return 0.8, pair

    result = prepare_selection_candidates(
        pool=_pool(chunks),
        primary_query="question",
        authorization=_authorization(policy),
        deadline=100.0,
        allow_new_scores=True,
        scorer=RevokingScorer(),
        chunk_loader=lambda _auth, _ids: chunks,
        clock=lambda: 0.0,
    )
    assert result.candidates == ()


def test_source_changed_during_scoring_drops_stale_candidate():
    original = _chunk(1, "original")
    changed = _chunk(1, "changed")
    loads = iter(((original,), (changed,)))
    result = prepare_selection_candidates(
        pool=_pool((original,)),
        primary_query="question",
        authorization=_authorization(Policy()),
        deadline=100.0,
        allow_new_scores=True,
        scorer=PointwiseScorer(),
        chunk_loader=lambda _auth, _ids: next(loads),
        clock=lambda: 0.0,
    )
    assert result.candidates == ()


def test_model_signature_mismatch_cannot_reuse_without_new_work():
    chunks = (_chunk(1, "first"),)
    result = prepare_selection_candidates(
        pool=_pool(
            chunks, (_score_set("question", chunks, (0.8,), scorer_fp="other-model"),)
        ),
        primary_query="question",
        authorization=_authorization(Policy()),
        deadline=100.0,
        allow_new_scores=False,
        scorer=PointwiseScorer(),
        chunk_loader=lambda _auth, _ids: chunks,
        clock=lambda: 0.0,
    )
    assert result.score_status == "rank_fallback"
    assert result.reused_pairs == 0


def test_listwise_scores_entire_union_and_constant_values_tie():
    chunks = (_chunk(1, "first"), _chunk(2, "second"))

    class ListwiseScorer(PointwiseScorer):
        scoring_kind = "listwise"

        def __init__(self):
            super().__init__()
            self.pool_calls = []

        def score_pair(self, pair, timeout_seconds):
            raise AssertionError("listwise scorer must receive full ordered pool")

        def score_pool(self, pairs, timeout_seconds):
            self.pool_calls.append(pairs)
            return tuple((0.2, pair) for pair in pairs)

    scorer = ListwiseScorer()
    result = prepare_selection_candidates(
        pool=_pool(
            chunks, (_score_set("question", chunks[:1], (0.8,), kind="listwise"),)
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
    assert result.reused_pairs == 0 and result.new_pairs == 2
    assert [candidate.relevance for candidate in result.candidates] == [0.5, 0.5]
    assert scorer.pool_calls == [(("question", "first"), ("question", "second"))]


def test_preparation_honors_configured_concurrency_below_six(monkeypatch):
    from apps.chat.services import rag_selection_scoring as scoring

    gate = Event()
    calls = []

    class BlockingScorer(PointwiseScorer):
        def score_pair(self, pair, timeout_seconds):
            calls.append(pair)
            gate.wait(1.0)
            return 0.8, pair

    monkeypatch.setattr(scoring, "rerank_score_concurrency", lambda: 2)
    chunks = tuple(_chunk(pk, f"content {pk}") for pk in range(1, 8))
    try:
        result = prepare_selection_candidates(
            pool=_pool(chunks),
            primary_query="question",
            authorization=_authorization(Policy()),
            deadline=monotonic() + 0.05,
            allow_new_scores=True,
            scorer=BlockingScorer(),
            chunk_loader=lambda _auth, _ids: chunks,
        )
        assert result.score_status == "rank_fallback"
        assert len(calls) <= 2
        assert result.new_pairs <= 2
    finally:
        gate.set()


def test_fresh_retry_input_mismatch_falls_back_for_entire_pool():
    chunks = (_chunk(1, "first"), _chunk(2, "second"))

    class MixedInputScorer(PointwiseScorer):
        def score_pair(self, pair, timeout_seconds):
            if pair[1] == "second":
                return 0.9, (pair[0], "shorter retry")
            return 0.1, pair

    result = prepare_selection_candidates(
        pool=_pool(chunks),
        primary_query="question",
        authorization=_authorization(Policy()),
        deadline=100.0,
        allow_new_scores=True,
        scorer=MixedInputScorer(),
        chunk_loader=lambda _auth, _ids: chunks,
        clock=lambda: 0.0,
    )
    assert result.score_status == "rank_fallback"
    assert result.fallback_reason == "score_incompatible"
    assert [candidate.relevance for candidate in result.candidates] == [1.0, 0.0]
