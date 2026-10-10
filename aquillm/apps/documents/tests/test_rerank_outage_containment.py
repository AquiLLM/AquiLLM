"""Synthetic HTTP proves request admission, never protocol search, on outages."""

import json
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread
from time import time, monotonic, sleep
from types import SimpleNamespace
from uuid import UUID

import pytest

from apps.documents.services.rerank_transport_policy import rerank_retry_action


@pytest.mark.parametrize("status", [None, 400, 401, 403, 408, 429, 500, 502, 503, 504])
def test_outage_or_invalid_request_never_discovers(status):
    assert rerank_retry_action(status, discovery=False) == "stop"
    assert rerank_retry_action(status, discovery=True) == "stop"


@pytest.mark.parametrize("status", [404, 405, 415, 422])
def test_protocol_negotiation_is_not_user_work(status):
    assert rerank_retry_action(status, discovery=False) == "stop"
    assert rerank_retry_action(status, discovery=True) == "next_protocol"


@pytest.mark.parametrize("status", [True, "503", 503.0, [], -1, 1000])
def test_invalid_status_types_are_rejected(status):
    with pytest.raises((TypeError, ValueError)):
        rerank_retry_action(status, discovery=True)


@pytest.mark.parametrize(
    "body",
    [
        None,
        {},
        {"model": "wrong", "data": [{"index": 0, "score": 1.0}]},
        {"model": "synthetic", "data": [{"index": True, "score": 1.0}]},
        {"model": "synthetic", "data": [{"index": 0, "score": float("nan")}]},
        {"model": "synthetic", "data": [{"index": 0, "score": float("inf")}]},
        {"model": "synthetic", "data": [{"index": 0, "score": True}]},
        {
            "model": "synthetic",
            "data": [{"index": 0, "score": 1.0}, {"index": 0, "score": 0.5}],
        },
    ],
)
def test_pure_finite_contract_rejects_invalid_model_scores_and_indices(body):
    from apps.documents.services.rerank_transport_policy import finite_protocol_scores

    assert (
        finite_protocol_scores(
            body,
            payload_kind="score_batch_text_pairs",
            count=1,
            model_key="synthetic",
            require_identity=True,
        )
        is None
    )


def test_pure_finite_contract_validates_exact_complete_batch():
    from apps.documents.services.rerank_transport_policy import finite_protocol_scores

    body = {
        "model": "synthetic",
        "data": [{"index": 1, "score": -0.5}, {"index": 0, "score": 0.2}],
    }
    assert finite_protocol_scores(
        body,
        payload_kind="score_batch_text_pairs",
        count=2,
        model_key="synthetic",
        require_identity=True,
    ) == (0.2, -0.5)


def test_pure_binding_requires_valid_endpoint_identity_and_finite_expiration():
    from apps.documents.services.rerank_transport_policy import RerankProtocolV1

    now = time()
    with pytest.raises(ValueError):
        RerankProtocolV1(
            "file:///invalid", "score_batch_text_pairs", "synthetic", now, now + 60
        )
    with pytest.raises(ValueError):
        RerankProtocolV1(
            "http://synthetic/score", "unknown", "synthetic", now, now + 60
        )
    with pytest.raises(ValueError):
        RerankProtocolV1(
            "http://synthetic/score",
            "score_batch_text_pairs",
            "synthetic",
            now,
            float("inf"),
        )


def test_pure_weighted_leases_limit_all_budgets_until_actual_completion():
    from lib.retrieval.turn_budget import TurnBudget, TurnLimits

    batch, pointwise = TurnBudget(TurnLimits()), TurnBudget(TurnLimits())
    assert batch.start_pairs(6, phase="acquisition")
    try:
        assert not pointwise.start_pair(phase="acquisition")
        batch.close("cancelled")
        assert not pointwise.start_pair(phase="acquisition")
    finally:
        batch.finish_pairs(6)
    assert pointwise.start_pair(phase="acquisition")
    pointwise.finish_pair()
    assert batch.pairs_used["acquisition"] == 6
    assert pointwise.pairs_used["acquisition"] == 1


def test_pure_latch_does_not_close_source_authority_and_never_admits_after_failure():
    from apps.documents.services.rerank_request_lifetime import TurnRerankState
    from lib.retrieval.turn_budget import TurnBudget, TurnLimits

    budget = TurnBudget(TurnLimits())
    state = TurnRerankState(budget)
    assert state.can_admit()
    state.fail()
    assert not state.can_admit()
    assert state.submit(1, lambda timeout: pytest.fail("closed admission")) is None
    assert budget.reserve_action("authorized lexical action")
    assert budget.can_publish()


def test_pure_worker_cancel_keeps_leases_and_cleans_on_actual_completion():
    from apps.documents.services.rerank_request_lifetime import TurnRerankState
    from lib.retrieval.turn_budget import TurnBudget, TurnLimits
    from contextvars import ContextVar

    context = ContextVar("test_worker_budget", default=None)
    started, release, done = Event(), Event(), Event()
    cleanup = []
    budget = TurnBudget(TurnLimits())

    def work(timeout):
        assert context.get() is budget
        started.set()
        release.wait(1)
        return 1.0

    token = context.set(budget)
    try:
        state = TurnRerankState(budget, worker_cleanup=lambda: cleanup.append(True))
        future = state.submit(6, work)
    finally:
        context.reset(token)
    assert started.wait(1)
    budget.close("cancelled")
    other = TurnBudget(TurnLimits())
    try:
        assert not future.cancel()
        assert cleanup == []
        assert not other.start_pair(phase="acquisition")
    finally:
        release.set()
        future.result(1)
    assert cleanup == [True]
    assert not state.publish(lambda: pytest.fail("late worker publication"))
    assert other.start_pair(phase="acquisition")
    other.finish_pair()


def test_pure_deadline_return_never_releases_a_stuck_worker_lease():
    from apps.documents.services.rerank_request_lifetime import TurnRerankState
    from lib.retrieval.turn_budget import TurnBudget, TurnLimits

    release, completed = Event(), Event()
    budget = TurnBudget(TurnLimits(retrieval_ms=50, final_scoring_ms=0))
    state = TurnRerankState(budget)

    def blocked(timeout):
        release.wait(1)
        completed.set()
        return 1.0

    start = monotonic()
    try:
        assert state.run(6, blocked) is None
        assert monotonic() - start < 0.3
        assert not completed.is_set()
        other = TurnBudget(TurnLimits())
        assert not other.start_pair(phase="acquisition")
    finally:
        release.set()
        completed.wait(1)
    for _ in range(100):
        if other.start_pair(phase="acquisition"):
            other.finish_pair()
            break
        sleep(0.001)
    else:
        pytest.fail("stuck worker completion did not release lease")
    assert not state.publish(lambda: pytest.fail("late publication"))


def test_pure_final_scoring_deadline_rejects_result_inside_longer_turn_deadline():
    from apps.documents.services.rerank_request_lifetime import TurnRerankState
    from lib.retrieval.turn_budget import TurnBudget, TurnLimits

    now = [0.0]
    budget = TurnBudget(TurnLimits(final_scoring_ms=10), clock=lambda: now[0])
    state = TurnRerankState(budget)

    def finish_late(timeout):
        now[0] = 0.02
        return 1.0

    assert state.run(1, finish_late, phase="final") is None
    assert budget.can_publish()
    assert budget.pairs_used["final"] == 1


def test_pure_cleanup_failure_releases_leases_and_rejects_scores():
    from apps.documents.services.rerank_request_lifetime import TurnRerankState
    from lib.retrieval.turn_budget import TurnBudget, TurnLimits

    budget = TurnBudget(TurnLimits())

    def cleanup():
        raise RuntimeError("synthetic cleanup failure")

    state = TurnRerankState(budget, worker_cleanup=cleanup)
    assert state.run(6, lambda timeout: 1.0) is None
    assert not state.can_admit()
    other = TurnBudget(TurnLimits())
    assert other.start_pair(phase="acquisition")
    other.finish_pair()


@contextmanager
def server(mode="503", release=None):
    requests = []
    started = Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            started.set()
            if release:
                release.wait(2)
            if mode == "reset":
                self.connection.close()
                return
            if mode.isdigit():
                self.send_response(int(mode))
                body = b"{}"
            else:
                self.send_response(200)
                count = (
                    len(payload["text_2"]) if isinstance(payload["text_2"], list) else 1
                )
                body = (
                    b"bad-json"
                    if mode == "malformed"
                    else json.dumps(
                        {
                            "model": "wrong" if mode == "wrong_model" else "synthetic",
                            "data": [
                                {"index": i, "score": float(i)} for i in range(count)
                            ],
                        }
                    ).encode()
                )
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}/score", requests, started
    finally:
        if release:
            release.set()
        httpd.shutdown()
        httpd.server_close()
        thread.join(2)


def chunks(count=24):
    return [
        SimpleNamespace(
            pk=i,
            doc_id=UUID(int=1),
            chunk_number=i - 1,
            content=f"synthetic source {i}",
        )
        for i in range(1, count + 1)
    ]


def configure(monkeypatch):
    monkeypatch.setenv("APP_RERANK_MODEL", "synthetic")
    monkeypatch.setenv("APP_RERANK_PROVIDER", "local")
    monkeypatch.setenv("RAG_RERANK_TEXT_MODE", "legacy")
    from apps.documents.services import rag_cache

    monkeypatch.setattr(rag_cache, "get_cached_rerank_capability", lambda *a, **k: None)
    monkeypatch.setattr(rag_cache, "get_cached_rerank_result", lambda *a, **k: None)


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


def test_discovery_only_binds_verified_model_and_finite_scores(monkeypatch):
    from apps.documents.services.rerank_transport_policy import discover_rerank_protocol
    from apps.documents.services import rag_cache

    monkeypatch.setattr(rag_cache, "set_cached_rerank_capability", lambda *a, **k: None)
    with server("success") as (endpoint, received, _):
        binding = discover_rerank_protocol(
            endpoint.removesuffix("/score"),
            "synthetic",
            deadline=monotonic() + 1,
            max_attempts=3,
        )
        assert binding.endpoint == endpoint
        assert binding.model_key == "synthetic"
        assert len(received) == 1
    with server("wrong_model") as (endpoint, received, _):
        assert (
            discover_rerank_protocol(
                endpoint.removesuffix("/score"),
                "synthetic",
                deadline=monotonic() + 1,
                max_attempts=3,
            )
            is None
        )
        assert len(received) == 1


def test_request_context_cannot_trigger_discovery():
    from apps.documents.services.rerank_transport_policy import discover_rerank_protocol
    from lib.llm.turn_context import TurnContext, bind_turn
    from lib.retrieval.turn_budget import TurnBudget, TurnLimits

    with bind_turn(TurnContext(TurnBudget(TurnLimits()))):
        with pytest.raises(RuntimeError):
            discover_rerank_protocol(
                "http://127.0.0.1:1", "synthetic", deadline=monotonic() + 1
            )


def test_cache_preserves_binding_identity_and_rejects_expiration(monkeypatch):
    from apps.documents.services.rerank_transport_policy import (
        RerankProtocolV1,
        resolve_rerank_protocol,
    )
    from apps.documents.services import rag_cache
    from django.test import override_settings

    with override_settings(RAG_CACHE_ENABLED=True):
        # Cache calls are isolated to a public synthetic mapping, not shared Redis.
        values = {}
        monkeypatch.setattr(
            rag_cache, "cache_set", lambda key, value, ttl: values.update({key: value})
        )
        monkeypatch.setattr(rag_cache, "cache_get", lambda key: values.get(key))
        now = time()
        binding = RerankProtocolV1(
            "http://synthetic/score",
            "score_batch_text_pairs",
            "synthetic",
            now - 1,
            now + 60,
        )
        rag_cache.set_cached_rerank_capability(
            "http://synthetic", "synthetic", binding.as_cache_entry()
        )
        assert resolve_rerank_protocol("http://synthetic", "synthetic") == binding
        assert resolve_rerank_protocol("http://synthetic", "different-model") is None
        values[next(iter(values))]["expires_at"] = now - 1
        assert resolve_rerank_protocol("http://synthetic", "synthetic") is None


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


@pytest.mark.asyncio
async def test_real_lifespan_warms_protocol_outside_request_and_revokes_on_shutdown(
    monkeypatch,
):
    import asyncio
    from apps.documents.services import pair_worker_lifecycle as worker
    from apps.documents.services import rag_cache
    from apps.documents.services import chunk_rerank_config as config
    from apps.documents.services.rerank_transport_policy import resolve_rerank_protocol
    from django.test import override_settings

    monkeypatch.setattr(worker, "enabled", lambda: False)
    monkeypatch.setenv("RAG_RERANK_PROTOCOL_WARM_ENABLED", "1")
    values = {}
    monkeypatch.setattr(
        rag_cache, "cache_set", lambda key, value, ttl: values.update({key: value})
    )
    monkeypatch.setattr(rag_cache, "cache_get", lambda key: values.get(key))
    monkeypatch.setattr(
        rag_cache,
        "delete_cached_rerank_capability",
        lambda base, model: values.pop(
            rag_cache.rerank_capability_cache_key(base, model), None
        ),
    )
    with (
        override_settings(RAG_CACHE_ENABLED=True),
        server("success") as (endpoint, received, started),
    ):
        base = endpoint.removesuffix("/score")
        monkeypatch.setattr(config, "rerank_base_url", lambda: base)
        monkeypatch.setattr(config, "rerank_model", lambda: "synthetic")
        monkeypatch.setattr(config, "rerank_provider", lambda: "local")
        wrapper = worker.PairCapabilityLifespan(None)
        queue = asyncio.Queue()

        async def send(message):
            pass

        task = asyncio.create_task(wrapper({"type": "lifespan"}, queue.get, send))
        await queue.put({"type": "lifespan.startup"})
        for _ in range(100):
            if resolve_rerank_protocol(base, "synthetic"):
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("startup did not bind a verified synthetic protocol")
        assert len(received) == 1
        second = worker.RerankProtocolController()
        assert await second.attempt() == resolve_rerank_protocol(base, "synthetic")
        assert len(received) == 1
        await queue.put({"type": "lifespan.shutdown"})
        await asyncio.wait_for(task, 0.5)
        assert resolve_rerank_protocol(base, "synthetic") is None
        await second.stop()


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
