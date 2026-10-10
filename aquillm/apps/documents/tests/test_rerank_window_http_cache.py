"""test rerank window http cache: synthetic regression coverage."""

from __future__ import annotations
from unittest.mock import MagicMock, patch
from django.test import override_settings


def test_window_overflow_charges_one_transport_and_disallows_retry_or_truncation():
    from time import monotonic

    from apps.documents.services.chunk_rerank_selection_provider import (
        LocalSelectionScorer,
    )
    from lib.retrieval import TurnBudget, TurnLimits

    scorer = LocalSelectionScorer(
        endpoint="http://test/score",
        shape="score_single_text_pair",
        model_name="m",
        revision="r",
        char_limit=2000,
        pair_limit=1024,
        reserve=256,
        timeout=3,
        deadline=monotonic() + 3,
    )
    budget = TurnBudget(TurnLimits())
    failed, good = MagicMock(status_code=400), MagicMock(status_code=200)
    good.json.return_value = {"score": 0.7}
    with patch(
        "apps.documents.services.chunk_rerank_selection_provider.requests.post",
        side_effect=[failed, good],
    ) as post:
        scored = scorer.score_budgeted_pair(
            ("whole question", "abcd" * 100), 2, budget=budget, phase="acquisition"
        )
    assert scored is None
    assert budget.pairs_used["acquisition"] == 1
    assert post.call_count == 1
    assert post.call_args.kwargs["json"]["text_2"] == "abcd" * 100
    assert all(
        "truncate_prompt_tokens" not in c.kwargs["json"] for c in post.call_args_list
    )
    assert all(0 < c.kwargs["timeout"] <= 2 for c in post.call_args_list)


@override_settings(RAG_CACHE_ENABLED=True)
def test_window_cache_requires_revision_and_rejects_late_writes():
    from time import monotonic
    from types import SimpleNamespace
    from uuid import UUID

    from apps.documents.services.chunk_rerank_window_adapter import (
        WindowSelectionScorer,
    )
    from lib.retrieval import TurnBudget, TurnLimits

    class Provider:
        calls = 0

        def score_pair(self, pair, timeout_seconds):
            self.calls += 1
            return (0.5, pair)

    provider = Provider()

    def counter(q, d):
        return len(q) + len(d)

    row = SimpleNamespace(
        pk=989,
        doc_id=UUID(int=989),
        chunk_number=0,
        content="cache evidence",
        source_revision="one",
    )

    def adapter(budget):
        return WindowSelectionScorer(
            provider,
            budget=budget,
            pair_counter=counter,
            scorer_identity="cache-fixture-unique",
            deadline=monotonic() + 2,
            cache_enabled=True,
        )

    first = adapter(TurnBudget(TurnLimits())).score_windows("q-cache", (row,))
    second = adapter(TurnBudget(TurnLimits())).score_windows("q-cache", (row,))
    assert first == second
    assert provider.calls == 1
    row.source_revision = "two"
    changed = adapter(TurnBudget(TurnLimits())).score_windows("q-cache", (row,))
    assert (
        changed.scores[0].effective_pair_fingerprint
        != first.scores[0].effective_pair_fingerprint
    )
    assert provider.calls == 2
    closed = TurnBudget(TurnLimits())
    closed.close("cancelled")
    assert adapter(closed).score_windows("q-cache", (row,)).status == "unavailable"
    assert provider.calls == 2


def test_verified_window_rejects_server_usage_that_hides_truncation():
    from time import monotonic

    from apps.documents.services.chunk_rerank_selection_provider import (
        LocalSelectionScorer,
    )
    from lib.retrieval import TurnBudget, TurnLimits

    scorer = LocalSelectionScorer(
        endpoint="http://test/score",
        shape="score_single_text_pair",
        model_name="m",
        revision="r",
        char_limit=2000,
        pair_limit=1024,
        reserve=256,
        timeout=3,
        deadline=monotonic() + 3,
    )
    scorer.verified_pair_counter = lambda q, d: 100
    response = MagicMock(status_code=200, url=scorer.endpoint)
    response.json.return_value = {"score": 0.8, "usage": {"prompt_tokens": 50}}
    budget = TurnBudget(TurnLimits())
    with patch(
        "apps.documents.services.chunk_rerank_selection_provider.requests.post",
        return_value=response,
    ):
        assert (
            scorer.score_budgeted_pair(("q", "source"), 1, budget=budget, phase="final")
            is None
        )
    assert budget.text_used["tokenized"] == 7
