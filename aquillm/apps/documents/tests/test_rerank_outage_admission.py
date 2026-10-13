"""test rerank outage admission: synthetic regression coverage."""

from threading import Event
from time import time, monotonic, sleep
import pytest
from .rerank_outage_support import server, chunks, configure


@pytest.mark.parametrize("adapter", ["scored", "legacy"])
@pytest.mark.parametrize(
    "mode", ["503", "401", "429", "malformed", "wrong_model", "reset"]
)
def test_first_failed_batch_stops_24_candidate_dispatch(monkeypatch, adapter, mode):
    configure(monkeypatch)
    from apps.documents.services.rerank_transport_policy import RerankProtocolV1
    from apps.documents.services import chunk_rerank_local_scored as scored
    from apps.documents.services import chunk_rerank_local_vllm as legacy
    from lib.retrieval.turn_budget import TurnBudget, TurnLimits

    rows = chunks()
    monkeypatch.setattr(
        legacy, "ordered_queryset_from_ids", lambda model, ids: tuple(ids)
    )
    with server(mode) as (endpoint, received, _):
        binding = RerankProtocolV1(
            endpoint, "score_batch_text_pairs", "synthetic", time() - 1, time() + 60
        )
        budget = TurnBudget(TurnLimits())
        if adapter == "scored":
            result = scored.rerank_via_local_vllm_scored(
                None, "question", rows, 5, turn_budget=budget, protocol_binding=binding
            )
            assert result.score_set.status == "unavailable"
            assert result.score_set.scores == ()
            assert result.ranked_ids == (1, 2, 3, 4, 5)
        else:
            result = legacy.rerank_via_local_vllm(
                None, "question", rows, 5, turn_budget=budget, protocol_binding=binding
            )
            assert result == (1, 2, 3, 4, 5)
        assert len(received) == 1
        assert len(received[0]["text_2"]) == 6
        assert budget.pairs_used == {"acquisition": 6, "final": 0}


@pytest.mark.parametrize("expired", [False, True])
def test_unknown_or_expired_binding_never_probes(monkeypatch, expired):
    configure(monkeypatch)
    from apps.documents.services.rerank_transport_policy import RerankProtocolV1
    from apps.documents.services.chunk_rerank_local_scored import (
        rerank_via_local_vllm_scored,
    )

    with server("success") as (endpoint, received, _):
        binding = (
            RerankProtocolV1(
                endpoint, "score_batch_text_pairs", "synthetic", time() - 60, time() - 1
            )
            if expired
            else None
        )
        result = rerank_via_local_vllm_scored(
            None, "question", chunks(), 5, protocol_binding=binding
        )
        assert result.ranked_ids == (1, 2, 3, 4, 5)
        assert result.score_set.status == "unavailable"
        assert received == []


def test_batch_deadline_returns_before_actual_io_releases_all_six_permits(monkeypatch):
    configure(monkeypatch)
    from apps.documents.services.rerank_transport_policy import RerankProtocolV1
    from apps.documents.services.chunk_rerank_local_scored import (
        rerank_via_local_vllm_scored,
    )
    from lib.retrieval.turn_budget import TurnBudget, TurnLimits
    from apps.documents.services import chunk_rerank_local_scored as scored

    cache_writes = []
    monkeypatch.setattr(
        scored, "set_scored_result", lambda *a, **k: cache_writes.append(True)
    )
    monkeypatch.setattr(scored, "rerank_model_revision", lambda: "a" * 40)
    monkeypatch.setattr(scored, "reusable_scored_result", lambda **k: None)
    release = Event()
    # Deliberately non-cooperative transport proves leases track actual completion.
    original = scored.requests.post
    monkeypatch.setattr(
        scored.requests, "post", lambda *a, **k: original(*a, **dict(k, timeout=2))
    )
    with server("success", release) as (endpoint, received, _):
        binding = RerankProtocolV1(
            endpoint, "score_batch_text_pairs", "synthetic", time() - 1, time() + 60
        )
        budget = TurnBudget(TurnLimits(retrieval_ms=50, final_scoring_ms=0))
        start = monotonic()
        result = rerank_via_local_vllm_scored(
            None, "q", chunks(), 5, turn_budget=budget, protocol_binding=binding
        )
        assert monotonic() - start < 0.4
        assert result.score_set.status == "unavailable"
        other = TurnBudget(TurnLimits())
        assert not other.start_pair(phase="acquisition")
        release.set()
        for _ in range(100):
            if other.start_pair(phase="acquisition"):
                other.finish_pair()
                break
            sleep(0.01)
        else:
            pytest.fail("actual completed I/O did not release permits")
        assert len(received) == 1
        assert cache_writes == []


def test_three_query_variants_share_first_failure_admission(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from apps.documents.services.rerank_transport_policy import RerankProtocolV1
    from apps.documents.services.rerank_request_lifetime import TurnRerankState
    from apps.documents.services.chunk_rerank_local_scored import (
        rerank_via_local_vllm_scored,
    )
    from lib.retrieval.turn_budget import TurnBudget, TurnLimits

    configure(monkeypatch)
    budget = TurnBudget(TurnLimits())
    state = TurnRerankState(budget)
    release = Event()
    with server("503", release) as (endpoint, received, started):
        binding = RerankProtocolV1(
            endpoint, "score_single_text_pair", "synthetic", time() - 1, time() + 60
        )
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(
                    rerank_via_local_vllm_scored,
                    None,
                    f"q{i}",
                    chunks(),
                    5,
                    turn_budget=budget,
                    rerank_state=state,
                    protocol_binding=binding,
                )
                for i in range(3)
            ]
            assert started.wait(1)
            sleep(0.05)
            assert len(received) <= 6
            release.set()
            assert all(f.result(2).score_set.status == "unavailable" for f in futures)
        assert len(received) <= 6
        assert budget.pairs_used["acquisition"] <= 6
        assert not state.can_admit()


def test_successful_batches_are_sequential_and_charge_every_pair(monkeypatch):
    configure(monkeypatch)
    from apps.documents.services.rerank_transport_policy import RerankProtocolV1
    from apps.documents.services.chunk_rerank_local_scored import (
        rerank_via_local_vllm_scored,
    )
    from lib.retrieval.turn_budget import TurnBudget, TurnLimits

    with server("success") as (endpoint, received, _):
        binding = RerankProtocolV1(
            endpoint, "score_batch_text_pairs", "synthetic", time() - 1, time() + 60
        )
        budget = TurnBudget(TurnLimits())
        result = rerank_via_local_vllm_scored(
            None, "q", chunks(14), 3, turn_budget=budget, protocol_binding=binding
        )
        assert result.score_set.status == "complete"
        assert [len(r["text_2"]) for r in received] == [6, 6, 2]
        assert budget.pairs_used["acquisition"] == 14
        assert result.ranked_ids == (6, 12, 5)


def test_stricter_turn_pair_limit_bounds_each_batch(monkeypatch):
    configure(monkeypatch)
    from apps.documents.services.rerank_transport_policy import RerankProtocolV1
    from apps.documents.services.chunk_rerank_local_scored import (
        rerank_via_local_vllm_scored,
    )
    from lib.retrieval.turn_budget import TurnBudget, TurnLimits

    with server("success") as (endpoint, received, _):
        binding = RerankProtocolV1(
            endpoint, "score_batch_text_pairs", "synthetic", time() - 1, time() + 60
        )
        budget = TurnBudget(TurnLimits(in_flight_pairs=2, acquisition_pairs=6))
        result = rerank_via_local_vllm_scored(
            None, "q", chunks(8), 3, turn_budget=budget, protocol_binding=binding
        )
        assert result.score_set.status == "unavailable"
        assert result.ranked_ids == (1, 2, 3)
        assert [len(r["text_2"]) for r in received] == [2, 2, 2]
        assert budget.pairs_used["acquisition"] == 6
