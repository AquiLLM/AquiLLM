"""Window scoring must never turn incomplete inputs into comparable scores."""

import pytest

from apps.documents.services.chunk_rerank_window_scores import (
    aggregate_window_scores,
    score_window_plan,
)
from apps.documents.services.chunk_rerank_windows import prepare_source_windows
from lib.retrieval import SourceEvidence, TurnBudget, TurnLimits


def test_window_adapter_reuses_exact_acquisition_scores_without_more_pairs():
    from types import SimpleNamespace
    from uuid import UUID

    from apps.documents.services.chunk_rerank_score_transport import (
        deserialize_score_set,
        serialize_score_set,
    )
    from apps.documents.services.chunk_rerank_scoring import score_missing_pairs
    from apps.documents.services.chunk_rerank_window_adapter import (
        WindowSelectionScorer,
    )

    budget = TurnBudget(TurnLimits())
    row = SimpleNamespace(
        pk=1, doc_id=UUID(int=1), chunk_number=0, content="evidence " * 180
    )
    scorer = WindowSelectionScorer(
        Scorer(),
        budget=budget,
        pair_counter=lambda q, d: len(q) + len(d) + 8,
        scorer_identity="verified-test",
    )
    first = scorer.score_windows("q", (row,), phase="acquisition")
    assert first.status == "complete"
    assert deserialize_score_set(serialize_score_set(first)) == first
    pairs = budget.pairs_used.copy()
    final = score_missing_pairs(
        query="q",
        chunks=(row,),
        scorer=scorer,
        deadline=scorer.deadline,
        turn_budget=budget,
    )
    assert final == first
    assert budget.pairs_used == pairs
    row.content += " changed"
    changed = scorer.score_windows("q", (row,), phase="final")
    assert (
        changed.scores[0].effective_pair_fingerprint
        != first.scores[0].effective_pair_fingerprint
    )
    assert budget.pairs_used["final"] > 0


def test_turn_slots_and_publication_close_are_atomic():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    budget = TurnBudget(TurnLimits())
    with ThreadPoolExecutor(max_workers=12) as executor:
        accepted = list(
            executor.map(lambda _: budget.start_pair(phase="acquisition"), range(12))
        )
    assert sum(accepted) == 6
    assert budget.pairs_used["acquisition"] == 6
    for _ in range(6):
        budget.finish_pair()
    entered, release, closed = Event(), Event(), Event()
    writes = []

    def write():
        entered.set()
        assert release.wait(2)
        writes.append("before close")

    with ThreadPoolExecutor(max_workers=2) as executor:
        publication = executor.submit(budget.publish, write)
        assert entered.wait(2)
        closing = executor.submit(lambda: (budget.close("done"), closed.set()))
        assert not closed.wait(0.02)
        release.set()
        assert publication.result()
        closing.result()
    assert not budget.publish(lambda: writes.append("late"))
    assert writes == ["before close"]


def plan(text="x" * 1500):
    return prepare_source_windows(
        "q",
        SourceEvidence(1, "d", 0, "r", text),
        pair_counter=lambda q, d: len(q) + len(d) + 8,
    )


class Scorer:
    def score_pair(self, pair, timeout_seconds):
        return (float(len(pair[1])), pair)


def test_max_aggregation_includes_all_required_windows():
    result = score_window_plan(plan(), scorer=Scorer(), budget=TurnBudget(TurnLimits()))
    aggregate = aggregate_window_scores(result)
    assert aggregate.prepared_input_scoring_coverage == "complete"
    assert aggregate.value == max(item.value for item in result.successful)
    assert len(result.successful) == len(result.plan.windows)
    assert aggregate.aggregation_version == "window-max-v1"


def test_partial_timeout_never_claims_complete_score():
    budget = TurnBudget(TurnLimits())

    class Stops(Scorer):
        def score_pair(self, pair, timeout_seconds):
            budget.close("timeout")
            return super().score_pair(pair, timeout_seconds)

    result = score_window_plan(plan(), scorer=Stops(), budget=budget)
    assert aggregate_window_scores(result).prepared_input_scoring_coverage == "partial"
    assert not result.successful


def test_shorter_successful_pair_is_partial_and_has_exact_identity():
    class Shorter(Scorer):
        def score_pair(self, pair, timeout_seconds):
            return (9.0, (pair[0], pair[1][:5]))

    result = score_window_plan(
        plan("longer evidence"), scorer=Shorter(), budget=TurnBudget(TurnLimits())
    )
    assert result.successful[0].pair == ("q", "longe")
    assert aggregate_window_scores(result).prepared_input_scoring_coverage == "partial"


def test_unknown_preparation_cannot_be_scored_as_complete():
    unknown = prepare_source_windows(
        "q", SourceEvidence(1, "d", 0, "r", "tail"), pair_counter=lambda q, d: None
    )
    result = score_window_plan(
        unknown, scorer=Scorer(), budget=TurnBudget(TurnLimits())
    )
    assert aggregate_window_scores(result).prepared_input_scoring_coverage == "unknown"


def test_configured_windowed_acquisition_never_uses_legacy_or_unbudgeted_http(
    monkeypatch,
):
    from types import SimpleNamespace
    from unittest.mock import patch
    from uuid import UUID

    from apps.documents.services.chunk_rerank import rerank_chunks_scored

    monkeypatch.setenv("RAG_RERANK_TEXT_MODE", "windowed")
    row = SimpleNamespace(
        pk=1, doc_id=UUID(int=1), chunk_number=0, content="full evidence " * 200
    )
    with patch("requests.post", side_effect=AssertionError("unexpected inference")):
        result = rerank_chunks_scored(None, "q", (row,), 1)
    assert result.ranked_ids == (1,)
    assert result.score_set.status == "unavailable"
    assert result.score_set.schema_version == "v3-window"


@pytest.mark.parametrize("corrupt", [False, True])
def test_windowed_selection_reuses_acquisition_sidecar(monkeypatch, corrupt):
    from dataclasses import replace
    from time import monotonic
    from types import SimpleNamespace
    from uuid import UUID

    from apps.chat.services.rag_selection_scoring import prepare_selection_candidates
    from apps.documents.services.chunk_rerank_window_adapter import (
        WindowSelectionScorer,
    )

    doc_id = UUID(int=2)
    row = SimpleNamespace(
        pk=2, doc_id=doc_id, chunk_number=0, content="full evidence " * 130
    )
    public = {"chunk_id": 2, "citation": "ref2"}
    budget = TurnBudget(TurnLimits())

    def counter(q, d):
        return len(q) + len(d) + 8

    acquisition = WindowSelectionScorer(
        Scorer(), budget=budget, pair_counter=counter, scorer_identity="tested"
    )
    initial = acquisition.score_windows("question", (row,), phase="acquisition")
    if corrupt:
        score = initial.scores[0]
        coverage = replace(
            score.window_coverage,
            required_window_ids=tuple(
                f"wrong-{i}"
                for i in range(len(score.window_coverage.required_window_ids))
            ),
        )
        initial = replace(initial, scores=(replace(score, window_coverage=coverage),))
    final = WindowSelectionScorer(
        Scorer(), budget=budget, pair_counter=counter, scorer_identity="tested"
    )
    from apps.documents.services.chunk_rerank_results import fingerprint_text

    hydrated = SimpleNamespace(
        chunk=row,
        source_fingerprint=fingerprint_text(row.content),
        excerpt=row.content,
        row=public,
    )
    monkeypatch.setattr(
        "apps.chat.services.rag_selection_scoring.hydrate_pool_rows",
        lambda *a, **k: (hydrated,),
    )
    pool = SimpleNamespace(
        rows=(public,), source_score_sets=(initial,), fused_scores=(("ref2", 1.0),)
    )
    prepared = prepare_selection_candidates(
        pool=pool,
        primary_query="question",
        authorization=object(),
        deadline=monotonic() + 3,
        allow_new_scores=not corrupt,
        scorer=final,
        turn_budget=budget,
    )
    assert prepared.score_status == ("rank_fallback" if corrupt else "model")
    assert prepared.new_pairs == 0
    assert prepared.candidates[0].text == row.content


def test_final_allowance_cannot_restart_across_adapters():
    from types import SimpleNamespace
    from uuid import UUID

    from apps.documents.services.chunk_rerank_window_adapter import (
        WindowSelectionScorer,
    )

    now = [0.0]
    budget = TurnBudget(TurnLimits(final_scoring_ms=100), clock=lambda: now[0])
    row = SimpleNamespace(pk=1, doc_id=UUID(int=1), chunk_number=0, content="evidence")

    def adapter():
        return WindowSelectionScorer(
            Scorer(),
            budget=budget,
            pair_counter=lambda q, d: 10,
            clock=lambda: now[0],
            deadline=10,
        )

    assert adapter().score_windows("q", (row,)).status == "complete"
    now[0] = 0.11
    assert adapter().score_windows("new query", (row,)).status == "unavailable"
