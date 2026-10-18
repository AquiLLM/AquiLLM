"""Malformed observations never establish a graph limit or source authority."""

from dataclasses import replace

import pytest

from apps.knowledge_graph.tests.test_closed_graph_attempt_v2 import attempt_for
from apps.knowledge_graph.tests.test_closed_graph_limits_v2 import (
    finishing_spies,
    returned_envelope,
)
from apps.knowledge_graph.tests.test_graph_candidate_operation_work import graph_frame

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize(
    "mutation",
    [
        "coverage",
        "reason",
        "frontier_type",
        "frontier_count",
        "frontier_string",
        "invocation",
        "selection",
        "envelope",
        "timeout",
        "value",
    ],
)
def test_malformed_or_unrelated_errors_never_become_proven_limits(
    monkeypatch, mutation
):
    from apps.knowledge_graph.retrieval import graph_runtime_v2 as runtime

    with graph_frame(monkeypatch) as active:

        def corrupt(kw, envelope):
            if mutation == "coverage":
                object.__setattr__(envelope.handoff.coverage, "ready_documents", True)
            elif mutation == "reason":
                object.__setattr__(envelope.handoff.coverage, "reasons", ("unknown",))
            elif mutation == "frontier_type":
                object.__setattr__(
                    envelope.frontier_invocation, "frontier_budget", object()
                )
            elif mutation == "frontier_count":
                object.__setattr__(envelope.frontier_budget, "queries_used", True)
            elif mutation == "frontier_string":
                object.__setattr__(
                    envelope.frontier_budget, "policy_signature", "private body"
                )
            elif mutation == "invocation":
                object.__setattr__(
                    envelope.frontier_invocation, "invocation_id", object()
                )
            elif mutation == "selection":
                object.__setattr__(envelope, "selection", replace(envelope.selection))

        returned_envelope(monkeypatch, empty=True, reasons=("deadline",), after=corrupt)
        if mutation in ("envelope", "timeout", "value"):

            def fail(**kwargs):
                if mutation == "envelope":
                    return object()
                raise (
                    TimeoutError("provider")
                    if mutation == "timeout"
                    else ValueError("provider")
                )

            monkeypatch.setattr(runtime, "_run_graph_v2", fail)
        calls = finishing_spies(monkeypatch)
        outcome = attempt_for(active, active[4], "What is A?").run_native_closure(
            closure_reserve_ms=1250
        )
        assert outcome.kind == "failure" and outcome.pool is None and calls == []
        assert outcome.diagnostics == {
            "graph_status": "error",
            "graph_candidate_count": 0,
        }


@pytest.mark.parametrize("forgery", ["subclass", "uninitialized", "overrides"])
def test_private_exception_forgery_remains_generic(monkeypatch, forgery):
    from apps.knowledge_graph.retrieval import graph_runtime_v2 as runtime
    from apps.knowledge_graph.retrieval.closed_graph_attempt_v2 import (
        _ClosedGraphLimited,
    )

    class ForgedLimit(_ClosedGraphLimited):
        pass

    if forgery == "subclass":
        error = ForgedLimit("deadline")
    elif forgery == "uninitialized":
        error = _ClosedGraphLimited.__new__(_ClosedGraphLimited)
    else:
        error = _ClosedGraphLimited("deadline")
        error.observations = {
            "graph_status": "hit",
            "graph_candidate_count": 45,
            "source_admission": {"fallback_top_up_count": 45},
        }

    def fail(**kwargs):
        raise error

    with graph_frame(monkeypatch) as active:
        monkeypatch.setattr(runtime, "_run_graph_v2", fail)
        calls = finishing_spies(monkeypatch)
        outcome = attempt_for(active, active[4], "What is A?").run_native_closure(
            closure_reserve_ms=1250
        )
        assert outcome.kind == "failure" and outcome.pool is None and calls == []
        assert outcome.diagnostics == {
            "graph_status": "error",
            "graph_candidate_count": 0,
        }


@pytest.mark.parametrize("reason", ["readiness_mismatch", "unknown", object()])
def test_private_limit_constructor_rejects_unrecognized_reason(reason):
    from apps.knowledge_graph.retrieval.closed_graph_attempt_v2 import (
        _ClosedGraphLimited,
    )

    with pytest.raises(ValueError):
        _ClosedGraphLimited(reason)


def test_failed_observation_snapshot_copies_scalars_and_retains_over_cap_charges(
    monkeypatch,
):
    from apps.knowledge_graph.retrieval.closed_graph_attempt_v2 import (
        _ClosedGraphLimited,
    )

    with graph_frame(monkeypatch) as active:
        seen = returned_envelope(monkeypatch, empty=True, reasons=("deadline",))
        outcome = attempt_for(active, active[4], "What is A?").run_native_closure(
            closure_reserve_ms=1250
        )
        assert outcome.diagnostics["graph_failure_reason"] == "deadline"
        envelope = seen[0]
        object.__setattr__(envelope.frontier_budget, "bytes_used", 1_048_577)
        error = _ClosedGraphLimited(
            "deadline",
            coverage=envelope.handoff.coverage,
            invocation=envelope.frontier_invocation,
        )
        object.__setattr__(envelope.frontier_budget, "bytes_used", 0)
        object.__setattr__(envelope.handoff.coverage, "reasons", ("not_ready",))
        assert error.observations["frontier_budget"]["bytes_used"] == 1_048_577
        assert error.observations["graph_coverage_v2"]["reasons"] == ("deadline",)
        assert outcome.diagnostics["graph_coverage_v2"]["reasons"] == ("deadline",)

        def scalar_snapshot(value):
            if type(value) is dict:
                return all(
                    type(k) is str and scalar_snapshot(v) for k, v in value.items()
                )
            if type(value) in (list, tuple):
                return all(scalar_snapshot(v) for v in value)
            return value is None or type(value) in (str, int, bool)

        assert scalar_snapshot(error.observations)
        assert set(vars(error)) == {"reason", "observations"}


@pytest.mark.parametrize(
    "mutation", ["authority", "callback", "count", "invocation_count", "reason"]
)
def test_allowed_snapshot_keys_cannot_carry_unvalidated_values(monkeypatch, mutation):
    from apps.knowledge_graph.retrieval import graph_runtime_v2 as runtime
    from apps.knowledge_graph.retrieval.closed_graph_attempt_v2 import (
        _ClosedGraphLimited,
    )

    with graph_frame(monkeypatch) as active:
        original, callbacks = runtime._run_graph_v2, []

        def callback():
            callbacks.append(True)
            raise AssertionError("observation callbacks must never execute")

        def failed(**kwargs):
            envelope = original(**kwargs)
            error = _ClosedGraphLimited(
                "deadline",
                coverage=envelope.handoff.coverage,
                invocation=envelope.frontier_invocation,
            )
            snapshot = error.observations
            if mutation == "authority":
                snapshot["graph_coverage_v2"] = active[3]
            elif mutation == "callback":
                snapshot["frontier_invocations"] = callback
            elif mutation == "count":
                snapshot["frontier_budget"]["queries_used"] = True
            elif mutation == "invocation_count":
                snapshot["frontier_invocations"][0]["frontier_budget"]["rows_used"] = (
                    object()
                )
            else:
                snapshot["graph_coverage_v2"]["reasons"] = ("unknown",)
            error.observations = dict(snapshot)
            raise error

        monkeypatch.setattr(runtime, "_run_graph_v2", failed)
        calls = finishing_spies(monkeypatch)
        outcome = attempt_for(active, active[4], "What is A?").run_native_closure(
            closure_reserve_ms=1250
        )
        assert outcome.kind == "failure" and outcome.pool is None
        assert outcome.diagnostics == {
            "graph_status": "error",
            "graph_candidate_count": 0,
        }
        assert calls == [] and callbacks == []
        assert active[2].cache == {} and active[3].bodies == {}


def test_consumed_limit_snapshot_does_not_alias_later_exception_mutations(monkeypatch):
    from apps.knowledge_graph.retrieval import graph_runtime_v2 as runtime
    from apps.knowledge_graph.retrieval.closed_graph_attempt_v2 import (
        _ClosedGraphLimited,
    )

    with graph_frame(monkeypatch) as active:
        original, errors = runtime._run_graph_v2, []

        def failed(**kwargs):
            envelope = original(**kwargs)
            error = _ClosedGraphLimited(
                "deadline",
                coverage=envelope.handoff.coverage,
                invocation=envelope.frontier_invocation,
            )
            errors.append(error)
            raise error

        monkeypatch.setattr(runtime, "_run_graph_v2", failed)
        outcome = attempt_for(active, active[4], "What is A?").run_native_closure(
            closure_reserve_ms=1250
        )
        before = outcome.diagnostics["frontier_budget"]["queries_used"]
        assert before > 0 and outcome.diagnostics["graph_failure_reason"] == "deadline"
        errors[0].observations["frontier_budget"]["queries_used"] = object()
        errors[0].observations["frontier_invocations"][0]["query_sha256"] = object()
        assert outcome.diagnostics["frontier_budget"]["queries_used"] == before
        assert (
            type(outcome.diagnostics["frontier_invocations"][0]["query_sha256"]) is str
        )
