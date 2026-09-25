"""Frozen source representation is the scorer, selector and synthesis input."""

from time import monotonic
from types import SimpleNamespace

from apps.chat.services.rag_selection_scoring import prepare_selection_candidates
from apps.chat.tests.test_rag_selection_scoring import (
    Policy,
    _authorization,
    _chunk,
    _pool,
)
from apps.documents.services.chunk_rerank_results import (
    fingerprint_pair,
    fingerprint_text,
)
from lib.retrieval.evidence import SourceSpan
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


class ExactScorer:
    scoring_kind = "pointwise"
    scorer_fingerprint = "exact-v1"

    def __init__(self):
        self.seen = []

    def prepare_pair(self, query, chunk):
        return query, chunk.content

    def score_pair(self, pair, timeout_seconds):
        self.seen.append(pair)
        return 1.0, pair


def test_scoring_selection_and_packet_use_full_source_tail():
    from apps.chat.services.rag_evidence import build_selected_evidence_packet
    from apps.chat.services.rag_selection import select_evidence
    from apps.chat.services.rag_selection_types import SelectionLimits, SelectionProfile

    text = "background " * 200 + "The result is 42 mK only below 1 Pa."
    chunk = _chunk(1, text)
    pool = _pool((chunk,))
    pool.rows[0]["text"] = "clipped public preview"
    scorer = ExactScorer()
    prepared = prepare_selection_candidates(
        pool=pool,
        primary_query="result?",
        authorization=_authorization(Policy()),
        deadline=monotonic() + 3,
        allow_new_scores=True,
        scorer=scorer,
        chunk_loader=lambda *_: (chunk,),
        turn_budget=TurnBudget(TurnLimits()),
        source_mode=True,
        token_ceiling=3500,
    )
    assert scorer.seen == [("result?", text)]
    assert prepared.candidates[0].text == text
    selected = select_evidence(
        prepared.candidates,
        profile=SelectionProfile("test", 1, 0, "v1"),
        limits=SelectionLimits(10, 10, 3500),
        score_status=prepared.score_status,
    )
    packet = build_selected_evidence_packet(
        selected, query="result?", search_scope="selected"
    )
    assert packet.chunks[0]["text"] == text
    assert packet.source_evidence[0].source.text == text


def test_prepared_spans_rescore_exact_rendering_and_revalidation_drops_changes():
    from apps.chat.services.rag_selection_hydration import (
        revalidate_selection_candidates,
    )

    text = "background " * 2000 + "Exception: retain only with consent."
    chunk = _chunk(1, text)
    fp = fingerprint_text(text)
    span = SourceSpan(
        1, fp, text.index("Exception:"), len(text), text[text.index("Exception:") :]
    )
    scorer = ExactScorer()
    auth = _authorization(Policy())
    prepared = prepare_selection_candidates(
        pool=_pool((chunk,)),
        primary_query="consent",
        authorization=auth,
        deadline=monotonic() + 3,
        allow_new_scores=True,
        scorer=scorer,
        chunk_loader=lambda *_: (chunk,),
        turn_budget=TurnBudget(TurnLimits()),
        source_mode=True,
        token_ceiling=100,
        source_windows={1: (span,)},
    )
    candidate = prepared.candidates[0]
    assert candidate.text == span.text
    assert scorer.seen == [("consent", span.text)]
    assert candidate.source_fingerprint == fp
    assert fingerprint_pair(*scorer.seen[0]) != fingerprint_pair("consent", text)
    changed = SimpleNamespace(**vars(chunk))
    changed.content += " changed"
    assert (
        revalidate_selection_candidates(
            (candidate,), auth, chunk_loader=lambda *_: (changed,)
        )
        == ()
    )


def test_source_mode_legacy_clipping_scorer_falls_back_without_false_coverage():
    from apps.chat.tests.test_rag_selection_scoring import PointwiseScorer

    chunk = _chunk(1, "background " * 200 + "qualified tail")
    scorer = PointwiseScorer()
    result = prepare_selection_candidates(
        pool=_pool((chunk,)),
        primary_query="the complete long primary question",
        authorization=_authorization(Policy()),
        deadline=monotonic() + 3,
        allow_new_scores=True,
        scorer=scorer,
        chunk_loader=lambda *_: (chunk,),
        turn_budget=TurnBudget(TurnLimits()),
        source_mode=True,
        token_ceiling=3500,
    )
    assert result.score_status == "rank_fallback"
    assert result.fallback_reason == "score_incompatible"
    assert result.candidates[0].text == chunk.content
    assert scorer.calls == []


def test_final_revalidation_does_not_prepare_a_second_representation(monkeypatch):
    from apps.chat.services import rag_source_hydration

    prepare = rag_source_hydration.prepare_evidence
    calls = []

    def recording(*args, **kwargs):
        calls.append(args[0])
        return prepare(*args, **kwargs)

    monkeypatch.setattr(rag_source_hydration, "prepare_evidence", recording)
    chunk = _chunk(1, "full source text")
    budget = TurnBudget(TurnLimits())
    result = prepare_selection_candidates(
        pool=_pool((chunk,)),
        primary_query="question",
        authorization=_authorization(Policy()),
        deadline=monotonic() + 3,
        allow_new_scores=False,
        chunk_loader=lambda *_: (chunk,),
        turn_budget=budget,
        source_mode=True,
        token_ceiling=3500,
    )
    assert result.candidates[0].text == chunk.content
    assert len(calls) == 1


def test_expired_source_stage_does_not_hydrate_or_tokenize():
    import pytest

    from apps.documents.services.source_loading import SourcePreparationLimited

    loaded = []
    with pytest.raises(SourcePreparationLimited):
        prepare_selection_candidates(
            pool=_pool((_chunk(1, "source"),)),
            primary_query="question",
            authorization=_authorization(Policy()),
            deadline=1,
            clock=lambda: 2,
            allow_new_scores=False,
            source_mode=True,
            turn_budget=TurnBudget(TurnLimits()),
            chunk_loader=lambda *_: loaded.append(True) or (),
        )
    assert not loaded


def test_full_source_preparation_reuses_exact_window_acquisition_scores():
    from apps.documents.services.chunk_rerank_window_adapter import (
        WindowSelectionScorer,
    )
    from apps.documents.tests.test_chunk_rerank_window_scores import Scorer

    budget = TurnBudget(TurnLimits())
    chunk = _chunk(1, "complete source evidence " * 80)
    scorer = WindowSelectionScorer(
        Scorer(),
        budget=budget,
        pair_counter=lambda q, d: len(q) + len(d) + 8,
        scorer_identity="verified-test",
    )
    scores = scorer.score_windows("q", (chunk,), phase="acquisition")
    assert scores.status == "complete"
    pairs = budget.pairs_used.copy()
    result = prepare_selection_candidates(
        pool=_pool((chunk,), (scores,)),
        primary_query="q",
        authorization=_authorization(Policy()),
        deadline=monotonic() + 3,
        allow_new_scores=True,
        scorer=scorer,
        chunk_loader=lambda *_: (chunk,),
        turn_budget=budget,
        source_mode=True,
    )
    assert result.score_status == "model"
    assert result.reused_pairs == 1 and result.new_pairs == 0
    assert budget.pairs_used == pairs
