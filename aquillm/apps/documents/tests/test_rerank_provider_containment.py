"""test rerank provider containment: synthetic regression coverage."""

from time import monotonic
from types import SimpleNamespace
import pytest
from .rerank_outage_support import server, chunks, configure


@pytest.mark.parametrize("route", ["pointwise", "scored_http", "window", "selection"])
def test_400_is_one_attempt_and_closes_request_admission(monkeypatch, route):
    from apps.documents.services.rerank_request_lifetime import TurnRerankState
    from lib.retrieval.turn_budget import TurnBudget, TurnLimits
    from apps.documents.services.chunk_rerank_selection_provider import (
        LocalSelectionScorer,
    )
    from apps.documents.services import chunk_rerank_pointwise as pointwise
    from apps.documents.services import chunk_rerank_local_scored_http as scored_http
    from apps.documents.services.chunk_rerank_window_http import score_budgeted_pair

    budget = TurnBudget(TurnLimits())
    state = TurnRerankState(budget)
    with server("400") as (endpoint, received, _):
        kwargs = dict(
            endpoint=endpoint,
            index=0,
            query="q",
            document="source " * 1000,
            headers={},
            timeout=1,
            model_name="synthetic",
            pair_token_limit=1024,
            reserve_tokens=96,
            rerank_state=state,
        )
        if route in ("pointwise", "scored_http"):
            module = pointwise if route == "pointwise" else scored_http
            assert module._score_one_document(**kwargs) is None
        else:
            provider = LocalSelectionScorer(
                endpoint=endpoint,
                shape="score_single_text_pair",
                model_name="synthetic",
                revision="",
                char_limit=900,
                pair_limit=1024,
                reserve=96,
                timeout=1,
                deadline=monotonic() + 1,
                turn_budget=budget,
            )
            if route == "window":
                import requests

                assert (
                    score_budgeted_pair(
                        provider,
                        ("q", "source " * 1000),
                        1,
                        budget=budget,
                        phase="acquisition",
                        post=requests.post,
                    )
                    is None
                )
            else:
                assert provider.score_pair(("q", "source " * 1000), 1) is None
        assert len(received) == 1
        assert not state.can_admit()


def test_auto_provider_outage_preserves_fused_order_without_cohere(monkeypatch):
    from apps.documents.services import chunk_rerank
    from apps.documents.services.chunk_rerank_results import rank_only_result_for_chunks

    monkeypatch.setattr(chunk_rerank, "rerank_provider", lambda: "auto")
    monkeypatch.setattr(
        chunk_rerank,
        "rerank_via_local_vllm_scored",
        lambda *a, **k: rank_only_result_for_chunks("q", chunks(), top_k=0),
    )
    monkeypatch.setattr(
        chunk_rerank.apps, "get_app_config", lambda *a: pytest.fail("provider retry")
    )
    result = chunk_rerank.rerank_chunks_scored(None, "q", chunks(), 5)
    assert result.ranked_ids == (1, 2, 3, 4, 5)


def test_explicit_cohere_outage_is_one_bounded_batch_with_sdk_retries_disabled(
    monkeypatch,
):
    from apps.documents.services import chunk_rerank
    from lib.retrieval.turn_budget import TurnBudget, TurnLimits

    calls = []

    def rerank(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("synthetic unavailable")

    monkeypatch.setattr(chunk_rerank, "rerank_provider", lambda: "cohere")
    monkeypatch.setattr(
        chunk_rerank.apps,
        "get_app_config",
        lambda name: SimpleNamespace(cohere_client=SimpleNamespace(rerank=rerank)),
    )
    budget = TurnBudget(TurnLimits())
    result = chunk_rerank.rerank_chunks_scored(
        None, "q", chunks(), 5, turn_budget=budget
    )
    assert result.ranked_ids == (1, 2, 3, 4, 5)
    assert result.score_set.status == "unavailable"
    assert len(calls) == 1
    assert len(calls[0]["documents"]) == 6
    assert calls[0]["request_options"]["max_retries"] == 0
    assert 0 < calls[0]["request_options"]["timeout_in_seconds"] <= 3
    assert budget.pairs_used["acquisition"] == 6


@pytest.mark.parametrize("route", ["selection", "window"])
def test_verified_selection_binding_expiration_rejects_late_scores(monkeypatch, route):
    from apps.documents.services import rerank_transport_policy as policy
    from apps.documents.services.chunk_rerank_selection_provider import (
        LocalSelectionScorer,
    )
    from lib.retrieval.turn_budget import TurnBudget, TurnLimits

    now, calls = [100.0], []
    monkeypatch.setattr(policy, "time", lambda: now[0])
    scorer = LocalSelectionScorer(
        endpoint="http://synthetic/score",
        shape="score_single_text_pair",
        model_name="synthetic",
        revision="r",
        char_limit=2000,
        pair_limit=1024,
        reserve=96,
        timeout=1,
        deadline=monotonic() + 1,
    )
    scorer.protocol_binding = policy.RerankProtocolV1(
        scorer.endpoint, scorer.shape, scorer.model_name, 99, 101
    )

    def post(endpoint, **kwargs):
        calls.append(kwargs)
        now[0] = 102
        return SimpleNamespace(
            status_code=200, url=endpoint, json=lambda: {"score": 0.9}
        )

    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_selection_provider.requests.post", post
    )
    budget = scorer.rerank_state.budget
    result = (
        scorer.score_pair(("q", "d"), 1)
        if route == "selection"
        else scorer.score_budgeted_pair(("q", "d"), 1, budget=budget, phase="final")
    )
    assert result is None
    assert len(calls) == 1
    assert not scorer.rerank_state.can_admit()
