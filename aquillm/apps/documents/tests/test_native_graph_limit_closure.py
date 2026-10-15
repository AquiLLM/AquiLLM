"""Limit observations cannot change native final authority or compatibility fallback."""

from time import monotonic

import pytest

from apps.documents.services.hybrid_graph_v2 import hybrid_graph_candidate_pool_v2
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.tests._native_graph_search_closure_support import (
    native_case,
    run_current_or_native,
)
from apps.knowledge_graph.tests.test_closed_graph_limits_v2 import returned_envelope

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("reason", ["deadline", "frontier_budget"])
@pytest.mark.parametrize("expired_owner", [False, True])
def test_native_limit_retains_only_real_finalized_baseline(
    monkeypatch, reason, expired_owner
):
    from apps.documents.services import native_graph_search_closure as closure

    epoch = monotonic()
    with native_case(
        monkeypatch, fail=False, graph_epoch=epoch, controlled_graph_clock=True
    ) as case:
        parent_deadline = case.operation.deadline

        def expire(kwargs, envelope):
            if expired_owner:
                case.now[0] = kwargs["deadline"] - epoch

        returned_envelope(monkeypatch, empty=True, reasons=(reason,), after=expire)
        original, finalized = closure._finalize_closed_ranking, []

        def final(*args, **kwargs):
            result = original(*args, **kwargs)
            finalized.append(result)
            return result

        monkeypatch.setattr(closure, "_finalize_closed_ranking", final)
        before = (
            case.source.cache.copy(),
            case.scope.bodies.copy(),
            case.scope.fusion_v2,
        )
        result = run_current_or_native(case)
        assert len(finalized) == 1
        assert tuple(result.final_rows[2]) == case.snapshot.baseline_candidates[:15]
        assert result.graph_diagnostics["graph_status"] == "error"
        assert result.graph_diagnostics["graph_failure_reason"] == (
            "deadline" if expired_owner else reason
        )
        assert result.graph_diagnostics["source_admission"]["fallback_top_up_skipped"]
        assert case.source.budget.sources_used == 27 and case.inference == []
        assert (case.source.cache, case.scope.bodies, case.scope.fusion_v2) == before
        assert case.operation.deadline == parent_deadline and case.operation.active()


@pytest.mark.parametrize("permission", ["live", "revoked"])
def test_limit_history_never_substitutes_for_fresh_permission(monkeypatch, permission):
    with native_case(monkeypatch, fail=False) as case:

        def revoke(kw, envelope):
            if permission == "revoked":
                case.f.policy.rows = ()

        returned_envelope(monkeypatch, empty=True, reasons=("deadline",), after=revoke)
        if permission == "revoked":
            with pytest.raises(SourceScopeUnavailable):
                run_current_or_native(case)
        else:
            assert run_current_or_native(case).final_rows[2]
        assert case.source.budget.sources_used == 27


@pytest.mark.parametrize("mode", ["early", "timeout", "owner", "inner", "invalid"])
def test_public_compatibility_fallback_keeps_calls_rows_and_admission(
    monkeypatch, mode
):
    from apps.knowledge_graph.retrieval import graph_candidate_operation_v2 as owner
    from apps.knowledge_graph.retrieval import graph_runtime_v2 as runtime

    epoch = monotonic()
    with native_case(
        monkeypatch, fail=False, graph_epoch=epoch, controlled_graph_clock=True
    ) as case:
        before_deadline = case.operation.deadline

        def after(kwargs, envelope):
            if mode == "owner":
                case.now[0] = kwargs["deadline"] - epoch
            elif mode == "invalid":
                case.operation.live = False

        seen = returned_envelope(
            monkeypatch, empty=True, reasons=("deadline",), after=after
        )
        if mode in ("early", "timeout"):

            def fail(**kwargs):
                raise (
                    TimeoutError("provider")
                    if mode == "timeout"
                    else ValueError("provider")
                )

            monkeypatch.setattr(runtime, "_run_graph_v2", fail)
        original, calls, fallback_results = owner._try_closed_graph_pool, [], []

        def capture(*args, fallback, **kwargs):
            def fallback_spy(**options):
                calls.append(options)
                result = fallback(**options)
                fallback_results.append((result[0], dict(result[1])))
                return result

            return original(*args, fallback=fallback_spy, **kwargs)

        monkeypatch.setattr(owner, "_try_closed_graph_pool", capture)
        rows, observation = hybrid_graph_candidate_pool_v2(
            case.snapshot, "Atlas", case.f.authorization, case.dependencies
        )
        if mode == "invalid":
            assert calls == [] and rows == ()
            assert observation["graph_status"] == "error"
            assert "graph_failure_reason" not in observation
        elif mode == "inner":
            assert calls == [] and observation["graph_status"] == "miss"
            assert "graph_failure_reason" not in observation
            assert len(rows) == 43 and case.source.budget.sources_used == 45
        else:
            expired = mode == "owner"
            assert calls == [{"admit_deferred": not expired}]
            assert rows is fallback_results[0][0]
            if expired:
                allowed = {
                    "graph_failure_reason",
                    "graph_coverage_v2",
                    "frontier_budget",
                    "frontier_invocations",
                }
                assert {
                    key: value
                    for key, value in observation.items()
                    if key not in allowed
                } == fallback_results[0][1]
                assert observation["graph_failure_reason"] == "deadline" and seen
                assert observation["graph_coverage_v2"]["reasons"] == ("deadline",)
            else:
                assert observation == fallback_results[0][1]
                assert "graph_failure_reason" not in observation
            assert len(rows) == (25 if expired else 45)
            assert case.source.budget.sources_used == (27 if expired else 45)
            assert observation["source_admission"]["fallback_top_up_count"] == (
                0 if expired else 20
            )
            assert observation["source_admission"]["fallback_top_up_skipped"] is expired
        assert case.operation.deadline == before_deadline
        assert case.operation.active() is (mode != "invalid")


def test_committed_database_permission_revocation_rejects_historical_limit(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from django.db import connection

    from apps.collections.models import Collection, CollectionPermission
    from apps.documents.models import RawTextDocument
    from apps.knowledge_graph.retrieval.closed_graph_attempt_v2 import (
        _ClosedGraphAttempt,
    )

    with native_case(monkeypatch, fail=False) as case:
        assert connection.vendor == "postgresql"
        user = case.f.documents["A"].ingested_by
        for document in case.f.documents.values():
            CollectionPermission.objects.create(
                user=user, collection_id=document.collection_id, permission="VIEW"
            )

        def current(**kwargs):
            collections = Collection.objects.filter_by_user_perm(user, "VIEW")
            read_budget = kwargs.get("read_budget")
            if read_budget is not None:
                read_budget.reserve_read(2)
            rows = tuple(
                RawTextDocument.objects.filter(
                    collection_id__in=collections
                ).values_list("collection_id", "id")
            )
            if read_budget is not None:
                read_budget.charge_read(reserved_rows=2, scanned_rows=len(rows))
            return rows

        monkeypatch.setattr(case.f.policy, "current_authorized_document_scope", current)
        monkeypatch.setattr(
            case.f.policy, "current_authorized_selected_document_scope", current
        )

        def revoke(kw, envelope):
            def committed():
                try:
                    CollectionPermission.objects.filter(user_id=user.pk).delete()
                finally:
                    connection.close()

            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(committed).result(timeout=10)

        returned_envelope(monkeypatch, empty=True, reasons=("deadline",), after=revoke)
        run, outcomes = _ClosedGraphAttempt.run_native_closure, []

        def observed(attempt, **kwargs):
            result = run(attempt, **kwargs)
            outcomes.append(result)
            return result

        monkeypatch.setattr(_ClosedGraphAttempt, "run_native_closure", observed)
        with pytest.raises(SourceScopeUnavailable):
            run_current_or_native(case)
        assert len(outcomes) == 1 and outcomes[0].kind == "failure"
        assert outcomes[0].diagnostics["graph_failure_reason"] == "deadline"
        assert not CollectionPermission.objects.filter(user=user).exists()
        assert case.operation.active() and case.inference == []
