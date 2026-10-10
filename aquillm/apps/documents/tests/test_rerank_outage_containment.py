"""test rerank outage containment: synthetic regression coverage."""

from threading import Event
from time import time, monotonic, sleep
import pytest
from apps.documents.services.rerank_transport_policy import rerank_retry_action
from .rerank_outage_support import server, chunks, configure


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
