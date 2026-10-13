"""Numerical rerank scores remain attached to their actual source inputs."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from uuid import UUID

import pytest
from django.test import override_settings

DOC_A = UUID("00000000-0000-0000-0000-000000000001")
DOC_B = UUID("00000000-0000-0000-0000-000000000002")


def _row(pk, content, doc_id=DOC_A):
    return SimpleNamespace(
        pk=pk,
        content=content,
        doc_id=doc_id,
        chunk_number=pk,
        modality="text",
        Modality=SimpleNamespace(IMAGE="image"),
    )


def test_score_order_preserves_values_and_original_ties():
    from apps.documents.services.chunk_rerank_results import order_scored_pairs

    assert order_scored_pairs(((1, 0.7), (0, 0.7), (2, 0.9)), (10, 20, 30)) == (
        (30, 0.9),
        (10, 0.7),
        (20, 0.7),
    )
    with pytest.raises(ValueError):
        order_scored_pairs(((0, 0.7), (0, 0.8)), (10, 20))


def test_score_set_rejects_stale_duplicate_unrelated_and_nonfinite_values():
    from apps.documents.services.chunk_rerank_results import (
        PassageScore,
        RerankScoreSet,
        validate_score_set,
    )

    identity = (10, DOC_A, 1, "source-a", "effective-a")
    base = PassageScore(10, DOC_A, 1, "source-a", "effective-a", 0.7)

    def score_set(scores, **changes):
        fields = dict(
            schema_version="v2",
            query_fingerprint="query",
            scorer_fingerprint="scorer",
            pool_fingerprint="pool",
            scoring_kind="pointwise",
            status="complete",
            candidate_order=(10,),
            scores=scores,
        )
        fields.update(changes)
        return RerankScoreSet(**fields)

    valid = validate_score_set(
        score_set((base,)),
        authorized_identities=(identity,),
        expected_query_fingerprint="query",
        expected_scorer_fingerprint="scorer",
    )
    assert valid.scores[0].value == 0.7
    with pytest.raises(FrozenInstanceError):
        valid.scores[0].value = 0.1
    bad = (
        score_set((base, base)),
        score_set((PassageScore(11, DOC_A, 1, "source-a", "effective-a", 0.7),)),
        score_set((PassageScore(10, DOC_B, 1, "source-a", "effective-a", 0.7),)),
        score_set((PassageScore(10, DOC_A, 1, "changed", "effective-a", 0.7),)),
        score_set((PassageScore(10, DOC_A, 1, "source-a", "old-retry", 0.7),)),
        score_set(
            (PassageScore(10, DOC_A, 1, "source-a", "effective-a", float("nan")),)
        ),
        score_set(
            (PassageScore(10, DOC_A, 1, "source-a", "effective-a", float("inf")),)
        ),
        score_set((PassageScore(10, DOC_A, 1, "source-a", "effective-a", True),)),
        score_set((base,), query_fingerprint="other"),
        score_set((base,), scorer_fingerprint="other"),
        score_set((), scoring_kind="rank_only", status="complete"),
    )
    for invalid in bad:
        with pytest.raises(ValueError):
            validate_score_set(
                invalid,
                authorized_identities=(identity,),
                expected_query_fingerprint="query",
                expected_scorer_fingerprint="scorer",
            )
    with pytest.raises(ValueError):
        validate_score_set(
            score_set(
                (PassageScore(1, DOC_A, 1, "source-a", "effective-a", 0.7),),
                candidate_order=(1,),
            ),
            authorized_identities=((True, DOC_A, 1, "source-a", "effective-a"),),
            expected_query_fingerprint="query",
            expected_scorer_fingerprint="scorer",
        )


@override_settings(RAG_CACHE_ENABLED=True)
def test_scored_cache_binds_complete_ordered_content_and_policy():
    from apps.documents.services.chunk_rerank_results import RerankScoreSet
    from apps.documents.services.chunk_rerank_score_cache import (
        get_scored_result,
        scored_result_cache_key,
        set_scored_result,
    )

    common = dict(
        query_fingerprint="q",
        scorer_fingerprint="s",
        scoring_kind="pointwise",
        candidate_identities=((10, "source", "pair"), (20, "source2", "pair2")),
        preparation_policy_fingerprint="template-v1-limit-1024",
    )
    key = scored_result_cache_key(**common)
    result = RerankScoreSet(
        "v2", "q", "s", "pool", "pointwise", "unavailable", (10, 20), ()
    )
    set_scored_result(key, result, timeout_seconds=45)
    assert get_scored_result(key) == result
    for change in (
        {"query_fingerprint": "q2"},
        {"scorer_fingerprint": "revision2"},
        {"scoring_kind": "listwise"},
        {"candidate_identities": ((20, "source2", "pair2"), (10, "source", "pair"))},
        {
            "candidate_identities": (
                (10, "changed-text", "pair"),
                (20, "source2", "pair2"),
            )
        },
        {
            "candidate_identities": (
                (10, "source", "retry-pair"),
                (20, "source2", "pair2"),
            )
        },
        {"preparation_policy_fingerprint": "template-v2-limit-1024"},
    ):
        assert get_scored_result(scored_result_cache_key(**(common | change))) is None


@override_settings(RAG_CACHE_ENABLED=True)
def test_old_id_only_result_cannot_be_read_as_a_scored_result():
    from apps.documents.services import rag_cache
    from apps.documents.services.chunk_rerank_score_cache import get_scored_result

    rag_cache.set_cached_rerank_result("query", (10,), 1, "model", [10])
    legacy_key = rag_cache.rerank_result_cache_key("query", (10,), 1, "model")
    assert rag_cache.get_cached_rerank_result("query", (10,), 1, "model") == [10]
    assert get_scored_result(legacy_key) is None


def test_skipped_small_pool_has_explicit_rank_only_result():
    from apps.documents.services.chunk_rerank_results import rank_only_result_for_chunks

    rows = [
        SimpleNamespace(pk=10, content="first"),
        SimpleNamespace(pk=20, content="second"),
    ]
    result = rank_only_result_for_chunks("question", rows, top_k=3)
    assert result.ranked_ids == (10, 20)
    assert result.score_set.candidate_order == (10, 20)
    assert result.score_set.status == "unavailable"
    assert result.score_set.scoring_kind == "rank_only"
    assert result.score_set.scores == ()


@override_settings(RAG_CACHE_ENABLED=True)
def test_effective_pair_cache_isolation_preserves_first_failure_latch(monkeypatch):
    from apps.documents.services import chunk_rerank, chunk_rerank_local_scored as scored
    from apps.documents.services import rag_cache, rerank_transport_policy as protocol
    from apps.documents.services.chunk_rerank_results import fingerprint_pair
    from lib.retrieval import TurnBudget, TurnLimits

    monkeypatch.setattr(chunk_rerank, "rerank_provider", lambda: "local")
    monkeypatch.setenv("RAG_RERANK_TEXT_MODE", "legacy")
    monkeypatch.setattr(scored, "rerank_base_url", lambda: "http://pair-cache/v1")
    monkeypatch.setattr(scored, "rerank_model", lambda: "synthetic")
    monkeypatch.setattr(scored, "rerank_model_revision", lambda: "pinned-revision")
    monkeypatch.setattr(scored, "rerank_doc_char_limit", lambda: 2000)
    monkeypatch.setattr(scored, "rerank_pair_token_limit", lambda: 1024)
    monkeypatch.setattr(scored, "rerank_template_reserve_tokens", lambda: 96)
    monkeypatch.setattr(protocol, "time", lambda: 100.0)
    binding = protocol.RerankProtocolV1(
        "http://pair-cache/score", "score_single_text_pair", "synthetic", 99, 160
    )
    monkeypatch.setattr(
        rag_cache, "get_cached_rerank_capability", lambda *a, **k: binding.as_cache_entry()
    )
    # Keep the real scored-cache serializer and validator over an isolated store.
    cache, payloads = {}, []
    monkeypatch.setattr(rag_cache, "cache_get", lambda key: cache.get(key))
    monkeypatch.setattr(
        rag_cache, "cache_set", lambda key, value, ttl: cache.update({key: value})
    )
    shorten = [False]
    monkeypatch.setattr(
        scored, "trim_rerank_pair",
        lambda query, text, *a: (query, text[:50] if shorten[0] else text),
    )

    def post(endpoint, *, json, **kwargs):
        assert endpoint == binding.endpoint
        assert json["model"] == binding.model_key
        assert kwargs["allow_redirects"] is False
        payloads.append(json)
        return SimpleNamespace(
            status_code=400 if len(payloads) == 1 else 200, url=endpoint,
            json=lambda: {"model": "synthetic", "score": 0.8},
        )

    monkeypatch.setattr(scored.requests, "post", post)
    row = _row(876, "evidence " * 30)
    query = "pair-sensitive"

    def score(budget=None):
        return chunk_rerank.rerank_chunks_scored(
            object, query, [row], 1, turn_budget=budget or TurnBudget(TurnLimits())
        ).score_set

    failed_budget = TurnBudget(TurnLimits())
    for _ in range(2):
        failed = score(failed_budget)
        assert failed.status == "unavailable" and failed.scores == ()
    assert len(payloads) == 1 and cache == {}
    assert failed_budget.pairs_used == {"acquisition": 1, "final": 0}

    # A new turn can score another effective pair; the failed turn cannot retry.
    shorten[0] = True
    short = score()
    assert short.status == "complete" and len(payloads) == 2
    short_pair = (payloads[1]["text_1"], payloads[1]["text_2"])
    assert short_pair == (query, row.content[:50])
    assert short.scores[0].effective_pair_fingerprint == fingerprint_pair(*short_pair)
    assert short.scores[0].effective_pair_fingerprint != fingerprint_pair(query, row.content)
    assert score() == short and len(payloads) == 2

    shorten[0] = False
    canonical = score()
    assert canonical.status == "complete" and len(payloads) == 3
    canonical_pair = (payloads[2]["text_1"], payloads[2]["text_2"])
    assert canonical_pair == (query, row.content)
    assert canonical.scores[0].effective_pair_fingerprint == fingerprint_pair(*canonical_pair)
    assert canonical.scorer_fingerprint == short.scorer_fingerprint
    assert canonical.scores[0].source_fingerprint == short.scores[0].source_fingerprint
    assert canonical.pool_fingerprint != short.pool_fingerprint
    assert score() == canonical and len(payloads) == 3
    assert len(cache) == 2
    assert all(entry["status"] == "complete" for entry in cache.values())
