"""test chunk rerank window fences: synthetic regression coverage."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from time import monotonic
from types import SimpleNamespace
import pytest
from apps.documents.services.chunk_rerank_selection_provider import LocalSelectionScorer
from lib.retrieval import TurnBudget, TurnLimits


@pytest.fixture
def verified_window_http():
    """Actual loopback HTTP, complete model identity and token-usage proof."""
    import json
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(payload)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {"model": "synthetic", "score": 0.7, "usage": {"prompt_tokens": 7}}
                ).encode()
            )

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}/score", calls
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(2)


def _verified_window_provider(endpoint, budget):
    from time import time

    from apps.documents.services.rerank_transport_policy import RerankProtocolV1

    scorer = LocalSelectionScorer(
        endpoint=endpoint,
        shape="score_single_text_pair",
        model_name="synthetic",
        revision="r",
        char_limit=2000,
        pair_limit=1024,
        reserve=96,
        timeout=3,
        deadline=monotonic() + 3,
        turn_budget=budget,
        protocol_binding=RerankProtocolV1(
            endpoint, "score_single_text_pair", "synthetic", time() - 1, time() + 60
        ),
    )
    scorer.verified_pair_counter = lambda q, d: 7
    return scorer


@pytest.mark.parametrize("phase", ["acquisition", "final"])
@pytest.mark.parametrize("route", ["plan", "adapter"])
def test_production_window_attempt_adapter_keeps_original_ledger_and_usage_proof(
    verified_window_http,
    phase,
    route,
):
    from uuid import UUID

    from apps.documents.services.chunk_rerank_window_adapter import (
        WindowSelectionScorer,
    )
    from apps.documents.services.chunk_rerank_window_scores import score_window_plan
    from apps.documents.services.chunk_rerank_windows import prepare_source_windows
    from lib.retrieval import SourceEvidence

    endpoint, calls = verified_window_http
    budget = TurnBudget(TurnLimits())
    scorer = _verified_window_provider(endpoint, budget)
    if route == "plan":
        plan = prepare_source_windows(
            "q",
            SourceEvidence(1, str(UUID(int=1)), 0, "source-proof", "source"),
            pair_counter=scorer.verified_pair_counter,
            budget=budget,
        )
        result = score_window_plan(plan, scorer=scorer, budget=budget, phase=phase)
        assert result.attempted_pairs == 1
        assert len(result.successful) == 1
        assert result.successful[0].pair == ("q", "source")
        assert result.successful[0].window_id == plan.required_window_ids[0]
    else:
        row = SimpleNamespace(
            pk=1, doc_id=UUID(int=1), chunk_number=0, content="source"
        )
        adapter = WindowSelectionScorer(
            scorer, budget=budget, pair_counter=scorer.verified_pair_counter
        )
        attempts = []
        result = adapter.score_windows(
            "q", (row,), phase=phase, on_submit=attempts.append
        )
        assert result.status == "complete"
        assert attempts == [1]
        assert result.scores[0].window_coverage.is_complete()
    assert scorer.rerank_state.budget is budget
    assert budget.pairs_used[phase] == 1
    assert len(calls) == 1
    assert calls[0]["text_1"] == "q" and calls[0]["text_2"] == "source"
    assert calls[0]["max_tokens_per_query"] == calls[0]["max_tokens_per_doc"] == 0


@pytest.mark.parametrize("cache_kind", ["memory", "persistent"])
def test_pending_authorized_window_cache_hit_obeys_shared_scoring_latch(
    monkeypatch, cache_kind
):
    from uuid import UUID, uuid4

    from django.test import override_settings

    from apps.documents.services import bounded_rag_cache as cache
    from apps.documents.services.chunk_rerank_window_adapter import (
        WindowSelectionScorer,
    )
    from apps.documents.services.rerank_request_lifetime import TurnRerankState

    row = SimpleNamespace(pk=1, doc_id=UUID(int=1), chunk_number=0, content="source")
    budget = TurnBudget(TurnLimits())
    state = TurnRerankState(budget)
    identity = "fix1-cache-" + uuid4().hex

    def adapter():
        return WindowSelectionScorer(
            SimpleNamespace(score_pair=lambda p, t: (0.7, p)),
            budget=budget,
            pair_counter=lambda q, d: 7,
            scorer_identity=identity,
            cache_enabled=True,
        )

    with override_settings(RAG_CACHE_ENABLED=True):
        primed = adapter()
        assert (
            primed.score_windows("q", (row,), phase="acquisition").status == "complete"
        )
        reader = primed if cache_kind == "memory" else adapter()
        entered, release = Event(), Event()
        if cache_kind == "memory":

            class Pending(dict):
                def get(self, key, default=None):
                    entered.set()
                    assert release.wait(2)
                    return super().get(key, default)

            reader._scores = Pending(reader._scores)
        else:
            transport = cache._transport

            def pending(config, operation, key, args, kwargs):
                if operation == "get":
                    entered.set()
                    assert release.wait(2)
                return transport(config, operation, key, args, kwargs)

            monkeypatch.setattr(cache, "_transport", pending)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(reader.score_windows, "q", (row,), phase="final")
            try:
                assert entered.wait(1)
                state.fail()  # A second variant fails while this cache hit is pending.
                assert budget.can_publish()
                assert budget.reserve_action("authorized-other-source")
            finally:
                release.set()
            result = future.result(2)
        assert result.status == "unavailable"
        assert result.scores == ()
        assert budget.pairs_used == {"acquisition": 1, "final": 0}


def test_window_success_cannot_cache_or_publish_after_other_variant_failure(
    monkeypatch,
    verified_window_http,
):
    from uuid import UUID, uuid4

    from django.test import override_settings

    from apps.documents.services import bounded_rag_cache as cache
    from apps.documents.services import chunk_rerank_window_adapter as windows
    from apps.documents.services.chunk_rerank_score_cache import (
        get_window_result,
        window_cache_key,
    )
    from apps.documents.services.rerank_request_lifetime import TurnRerankState

    endpoint, calls = verified_window_http
    budget = TurnBudget(TurnLimits())
    state = TurnRerankState(budget)
    scorer = _verified_window_provider(endpoint, budget)
    adapter = windows.WindowSelectionScorer(
        scorer,
        budget=budget,
        pair_counter=scorer.verified_pair_counter,
        cache_enabled=True,
        scorer_identity="fix1-after-http-" + uuid4().hex,
    )
    row = SimpleNamespace(pk=1, doc_id=UUID(int=1), chunk_number=0, content="source")
    plan = adapter.plan("q", row)
    key = window_cache_key(adapter.scorer_fingerprint, plan.fingerprint)
    entered, release, writes = Event(), Event(), []
    aggregate = windows.aggregate_window_scores

    def pending(result):
        value = aggregate(result)
        assert value.prepared_input_scoring_coverage == "complete"
        entered.set()  # The actual HTTP and window proof have completed.
        assert release.wait(2)
        return value

    monkeypatch.setattr(windows, "aggregate_window_scores", pending)
    transport = cache._transport

    def observe(config, operation, key, args, kwargs):
        if operation == "set":
            writes.append(key)
        return transport(config, operation, key, args, kwargs)

    monkeypatch.setattr(cache, "_transport", observe)
    with (
        override_settings(RAG_CACHE_ENABLED=True),
        ThreadPoolExecutor(max_workers=1) as executor,
    ):
        future = executor.submit(
            adapter.score_windows, "q", (row,), phase="acquisition"
        )
        try:
            assert entered.wait(1)
            state.fail()
            assert budget.can_publish()
            assert budget.reserve_action("authorized-other-source")
        finally:
            release.set()
        result = future.result(2)
        assert len(calls) == 1
        assert budget.pairs_used["acquisition"] == 1
        assert result.status == "unavailable" and result.scores == ()
        assert adapter._scores == {}
        assert writes == []
        assert get_window_result(key, budget=budget) is None
