"""Pause real rerank workers after admission, before their original target runs."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event, Thread
from time import time
from types import SimpleNamespace

from .rerank_outage_support import server


@contextmanager
def paused_http_score(monkeypatch, kind, phase):
    from apps.documents.services import chunk_rerank_local_scored_http as transport
    from apps.documents.services import rerank_request_lifetime as lifetime
    from apps.documents.services.rerank_transport_policy import RerankProtocolV1
    from lib.retrieval.turn_budget import TurnBudget, TurnLimits

    entered, release = Event(), Event()
    threads, dispatches, cleanup, expired_barriers = [], [], [], []
    clock = [0.0]
    limits = TurnLimits()
    budget = TurnBudget(limits, clock=lambda: clock[0])
    state = lifetime.TurnRerankState(budget, worker_cleanup=lambda: cleanup.append(1))
    original_post = transport.requests.post

    def observed_post(*args, **kwargs):
        dispatches.append((kwargs["timeout"], budget.scoring_remaining_ms(phase)))
        return original_post(*args, **kwargs)

    def deferred_thread(*, target, args, **kwargs):
        def delegated():
            entered.set()
            if not release.wait(5):
                expired_barriers.append(True)
            target(*args)

        thread = Thread(target=delegated, **kwargs)
        threads.append(thread)
        return thread

    with monkeypatch.context() as patch, server("success") as (endpoint, received, _):
        patch.setattr(lifetime, "Thread", deferred_thread)
        patch.setattr(transport.requests, "post", observed_post)
        binding = RerankProtocolV1(
            endpoint, kind, "synthetic", time() - 1, time() + 60
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                transport.score_bound_pairs,
                binding,
                (("question", "original source"),),
                state=state,
                headers={},
                timeout=10,
                phase=phase,
            )

            def finish():
                release.set()
                for thread in threads:
                    thread.join(5)
                    assert not thread.is_alive()
                result = future.result(5)
                assert not expired_barriers
                return result

            try:
                assert entered.wait(2), "real admitted worker never reached barrier"
                assert state.budget is budget and budget.limits is limits
                assert budget._inflight == 1
                assert budget.pairs_used[phase] == 1
                assert dispatches == [] and received == []
                yield SimpleNamespace(
                    budget=budget, state=state, clock=clock, finish=finish,
                    dispatches=dispatches, received=received, cleanup=cleanup,
                    deadline=budget._deadline, final_deadline=budget._final_deadline,
                )
            finally:
                finish()


def assert_original_lease_finished(case, phase):
    from lib.retrieval.turn_budget import TurnBudget, TurnLimits

    assert case.cleanup == [1]
    assert case.budget._inflight == 0
    assert case.budget._deadline == case.deadline
    assert case.budget._final_deadline == case.final_deadline
    assert case.budget.pairs_used == {
        "acquisition": int(phase == "acquisition"),
        "final": int(phase == "final"),
    }
    probe = TurnBudget(TurnLimits())
    admitted = probe.start_pairs(6, phase="acquisition")
    try:
        assert admitted, "completed worker leaked a process pair lease"
    finally:
        if admitted:
            probe.finish_pairs(6)
