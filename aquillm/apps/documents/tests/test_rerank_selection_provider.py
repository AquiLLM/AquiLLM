"""Bounded local HTTP scoring for the final evidence candidate union."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID


def test_known_pointwise_endpoint_sends_model_and_caps_timeout(monkeypatch):
    from apps.documents.services import chunk_rerank_selection_provider as provider

    requests = []

    def post(endpoint, *, headers, json, timeout):
        requests.append((endpoint, json, timeout))
        return SimpleNamespace(
            status_code=200, json=lambda: {"data": [{"score": -0.4}]}
        )

    monkeypatch.setattr(provider.requests, "post", post)
    scorer = provider.LocalSelectionScorer(
        endpoint="http://reranker/score",
        shape="score_single_text_pair",
        model_name="model",
        revision="revision",
        char_limit=2000,
        pair_limit=1024,
        reserve=256,
        timeout=3.0,
        deadline=2.0,
        clock=lambda: 0.0,
        headers={},
    )
    result = scorer.score_pair(("question", "document"), 1.0)
    assert result == (-0.4, ("question", "document"))
    assert requests == [
        (
            "http://reranker/score",
            {
                "model": "model",
                "text_1": "question",
                "text_2": "document",
                "truncate_prompt_tokens": 1024,
                "truncation_side": "right",
            },
            1.0,
        )
    ]


def test_retry_checks_deadline_and_records_successful_shorter_pair(monkeypatch):
    from apps.documents.services import chunk_rerank_selection_provider as provider

    times = iter((0.0, 0.0, 0.1, 0.2))
    payloads = []

    def post(endpoint, *, headers, json, timeout):
        payloads.append(json)
        if len(payloads) == 1:
            return SimpleNamespace(status_code=400)
        return SimpleNamespace(status_code=200, json=lambda: {"data": [{"score": 0.6}]})

    monkeypatch.setattr(provider.requests, "post", post)
    scorer = provider.LocalSelectionScorer(
        endpoint="http://reranker/score",
        shape="score_single_text_pair",
        model_name="model",
        revision="revision",
        char_limit=2000,
        pair_limit=64,
        reserve=0,
        timeout=3.0,
        deadline=1.0,
        clock=lambda: next(times),
        headers={},
    )
    pair = ("query " * 30, "document " * 30)
    result = scorer.score_pair(pair, 1.0)
    assert result is not None and result[0] == 0.6
    assert result[1] != pair
    assert payloads[1]["text_1"] == result[1][0]
    assert payloads[1]["text_2"] == result[1][1]


def test_unknown_capability_does_not_probe(monkeypatch):
    from apps.documents.services import chunk_rerank_selection_provider as provider

    monkeypatch.setattr(provider, "rerank_provider", lambda: "local")
    monkeypatch.setattr(provider, "rerank_base_url", lambda: "http://reranker/v1")
    monkeypatch.setattr(provider, "rerank_model", lambda: "model")
    monkeypatch.setattr(
        provider.rag_cache, "get_cached_rerank_capability", lambda *_: None
    )
    assert provider.current_selection_scorer(deadline=3.0, clock=lambda: 0.0) is None


def test_actual_long_query_pair_is_stable_when_other_candidates_change(monkeypatch):
    from apps.documents.services import chunk_rerank_selection_provider as provider
    from apps.documents.services.chunk_rerank_scoring import score_missing_pairs

    monkeypatch.setattr(
        provider.requests,
        "post",
        lambda *_args, **_kwargs: SimpleNamespace(
            status_code=200, json=lambda: {"data": [{"score": 0.5}]}
        ),
    )
    scorer = provider.LocalSelectionScorer(
        endpoint="http://reranker/score",
        shape="score_single_text_pair",
        model_name="model",
        revision="revision",
        char_limit=2000,
        pair_limit=64,
        reserve=0,
        timeout=3.0,
        deadline=100.0,
        clock=lambda: 0.0,
        headers={},
    )
    doc = UUID("11111111-1111-4111-8111-111111111111")
    other = SimpleNamespace(pk=1, doc_id=doc, chunk_number=0, content="alpha " * 200)
    target = SimpleNamespace(pk=2, doc_id=doc, chunk_number=1, content="beta " * 3)
    query = "question " * 200
    full = score_missing_pairs(
        query=query,
        chunks=(other, target),
        scorer=scorer,
        deadline=100.0,
        clock=lambda: 0.0,
    )
    alone = score_missing_pairs(
        query=query,
        chunks=(target,),
        scorer=scorer,
        deadline=100.0,
        clock=lambda: 0.0,
    )
    assert full.status == alone.status == "complete"
    assert full.scores[1].effective_pair_fingerprint == (
        alone.scores[0].effective_pair_fingerprint
    )
