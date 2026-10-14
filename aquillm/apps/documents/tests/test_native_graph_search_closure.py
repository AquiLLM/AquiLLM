"""Native closure removes redundant failed-graph work without weakening finality."""

import pytest

from ._native_graph_search_closure_support import (
    native_case,
    owner_queries,
    run_current_or_native,
)

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("graph_on", [True, False])
def test_expired_graph_uses_one_fresh_closure_and_no_acquisition_inference(
    monkeypatch, graph_on
):
    with native_case(monkeypatch, graph_on=graph_on) as case:
        before = case.source.budget.pairs_used.copy()
        result = run_current_or_native(case)
        assert result.final_rows[2]
        assert len(case.full_checks) == 2
        assert case.fetched == []
        assert case.inference == []
        assert case.source.budget.pairs_used == before
        assert result.ranking.rerank_status == "unavailable"
        assert result.score_set is None or result.score_set.scores == ()
        assert len(owner_queries(case)) == 2


def test_failed_graph_never_enters_acquisition_adapter(monkeypatch):
    with native_case(monkeypatch) as case:
        run_current_or_native(case)
        assert case.inference == []


@pytest.mark.parametrize("expired", [False, True])
def test_failure_keeps_only_initial_prefix_and_finalized_replay_rows(
    monkeypatch, expired
):
    with native_case(monkeypatch, expired=expired) as case:
        baseline = case.snapshot.baseline_candidates
        before = case.source.budget.text_used.copy()
        result = run_current_or_native(case)
        assert case.source.budget.sources_used == 25
        assert case.source.budget.text_used == before and case.fetched == []
        assert tuple(result.final_rows[2]) == baseline[:15]
        assert result.ranking.combined_candidates == tuple(result.final_rows[2])
        assert result.hybrid_pool == tuple(result.final_rows[2])
        assert result.preclosure_pool_count == 25
        assert result.materialized_union_status == "unavailable"
        assert result.graph_diagnostics["source_admission"]["fallback_top_up_skipped"]
        assert not case.events and result.score_set is None


def test_success_keeps_native_graph_proofs_and_single_source_finalizer(monkeypatch):
    from apps.documents.services import native_graph_search_closure as closure
    from apps.knowledge_graph.tests.test_graph_candidate_operation_work import (
        count_full_authorities,
    )
    from lib import replay_observation

    with native_case(monkeypatch, fail=False) as case:
        proofs, events, finals = count_full_authorities(monkeypatch), [], []
        finalizer = closure._finalize_closed_ranking

        def final(*args, **kwargs):
            case.after_graph = True
            case.after_graph_query_index = len(
                __import__("django.db").db.connection.queries
            )
            finals.append(args[1])
            return finalizer(*args, **kwargs)

        monkeypatch.setattr(closure, "_finalize_closed_ranking", final)
        monkeypatch.setattr(
            replay_observation, "graph_selected_sources_v2", events.append
        )
        result = run_current_or_native(case)
        assert result.graph_diagnostics["graph_status"] == "hit"
        assert len(proofs) == 2 and case.driver.calls and events
        assert len(finals) == 1 and len(case.full_checks) == 2
        assert len(owner_queries(case)) == 2
        assert case.source.budget.sources_used <= 45 and case.inference == []
        assert case.f.chunks["B"].pk in {
            row.pk for row in finals[0].combined_candidates
        }
        assert result.ranking.graph_candidates == ()


@pytest.mark.parametrize(
    "kind", ["scored", "fts", "wrapped", "no_admission", "deferred"]
)
def test_initial_ineligibility_has_no_reads_and_records_scored(monkeypatch, kind):
    from dataclasses import replace

    from django.conf import settings
    from django.db import connection
    from django.test.utils import CaptureQueriesContext

    from apps.documents.services.native_graph_search_closure import (
        _try_native_search_closure,
    )

    with native_case(monkeypatch) as case:
        if kind == "scored":
            monkeypatch.delenv("RAG_ACQUISITION_SCHEDULE")
        elif kind == "fts":
            monkeypatch.setattr(settings, "RAG_FULL_TEXT_ENABLED", True)
        elif kind == "wrapped":
            case.authorize = replace(case.authorize, delegate=lambda rows: rows)
        elif kind == "no_admission":
            case.snapshot = replace(case.snapshot, source_admission=None)
        else:
            case.snapshot.baseline_candidates[0].__dict__.pop("content")
        observation = {}
        with CaptureQueriesContext(connection) as queries:
            result = _try_native_search_closure(
                case.snapshot,
                "Atlas",
                15,
                authorize=case.authorize,
                dependencies=case.dependencies,
                overlay_enabled=True,
                observation=observation,
            )
        assert result is None and len(queries) == 0
        assert observation == {"acquisition_schedule": "scored"}
        assert case.driver.calls == [] and case.inference == []


@pytest.mark.parametrize(
    "value,want",
    [
        (None, "scored"),
        ("native_closure", "native_closure"),
        (" SCORED ", "scored"),
        ("", None),
        ("other", None),
    ],
)
def test_schedule_configuration(monkeypatch, value, want):
    from django.core.exceptions import ImproperlyConfigured

    from apps.documents.services.native_search_schedule import acquisition_schedule

    if value is None:
        monkeypatch.delenv("RAG_ACQUISITION_SCHEDULE", raising=False)
    else:
        monkeypatch.setenv("RAG_ACQUISITION_SCHEDULE", value)
    if want is None:
        with pytest.raises(ImproperlyConfigured):
            acquisition_schedule()
    else:
        assert acquisition_schedule() == want


@pytest.mark.parametrize("graph_on", [True, False])
def test_fresh_requested_check_is_not_duplicated_after_graph(monkeypatch, graph_on):
    from apps.documents.services import native_graph_search_closure as closure
    from apps.knowledge_graph.retrieval import graph_candidate_operation_v2 as graph

    with native_case(monkeypatch, graph_on=graph_on) as case:
        owner = graph if graph_on else closure
        original, checks = owner.validate_requested_authorization, []

        def validate(*args):
            result = original(*args)
            checks.append(case.after_graph)
            return result

        monkeypatch.setattr(owner, "validate_requested_authorization", validate)
        result = run_current_or_native(case)
        assert result.final_rows[2]
        assert checks == ([False, False] if graph_on else [True])


@pytest.mark.parametrize("elapsed,want", [(6.0, 2750), (9.0, 0)])
def test_clock_translation_keeps_original_tail_and_never_tops_up(
    monkeypatch, elapsed, want
):
    from apps.knowledge_graph.retrieval import graph_candidate_operation_v2 as graph

    with native_case(monkeypatch, expired=False, start=1000, graph_epoch=-990) as case:
        original, seen = graph._run_closed_graph, []
        deadline = case.operation.deadline
        case.now[0] += elapsed

        def run(*args):
            seen.append((args[6] - case.graph.clock()) * 1000)
            return original(*args)

        monkeypatch.setattr(graph, "_run_closed_graph", run)
        result = run_current_or_native(case)
        assert result.final_rows[2] and case.operation.deadline == deadline == 1010
        assert case.source.budget.sources_used == 25 and case.fetched == []
        if want:
            assert len(seen) == 1 and want - 2 <= seen[0] <= want
        else:
            assert seen == [] and case.driver.calls == []


@pytest.mark.parametrize("graph_on", [True, False])
def test_rrf_keeps_exact_captured_order_without_manufactured_scores(
    monkeypatch, graph_on
):
    with native_case(monkeypatch, graph_on=graph_on, rrf=True) as case:
        # Existing captured fusion independently specifies the protected/RRF prefix.
        expected = tuple(c.chunk_id for c in case.scope.fusion_v2.candidates)[:15]
        result = run_current_or_native(case)
        assert tuple(row.pk for row in result.final_rows[2]) == expected
        assert result.ranking.score_set is None and result.score_set is None
        assert case.source.budget.pairs_used["acquisition"] == 0


def test_final_window_scoring_still_admits_under_same_reserved_budget(monkeypatch):
    from apps.documents.services.chunk_rerank_window_adapter import (
        WindowSelectionScorer,
    )
    from apps.documents.tests.test_chunk_rerank_window_scores import Scorer

    with native_case(monkeypatch, graph_on=False) as case:
        result = run_current_or_native(case)
        budget = case.source.budget
        scorer = WindowSelectionScorer(
            Scorer(),
            budget=budget,
            pair_counter=lambda q, d: len(q) + len(d) + 8,
            clock=lambda: case.now[0],
            deadline=case.now[0] + 3,
        )
        scores = scorer.score_windows(
            "Atlas", tuple(result.final_rows[2]), phase="final"
        )
        assert scores.status == "complete" and scores.scores
        assert budget.pairs_used["acquisition"] == 0
        assert 0 < budget.pairs_used["final"] <= 45
        assert budget.limits.final_scoring_ms == 3000


def test_off_custom_dependencies_remain_ineligible_without_reads(monkeypatch):
    from apps.documents.services.native_graph_search_closure import (
        _try_native_search_closure,
    )

    with native_case(monkeypatch, graph_on=False) as case:
        result = _try_native_search_closure(
            case.snapshot,
            "Atlas",
            15,
            authorize=case.authorize,
            dependencies=object(),
            overlay_enabled=False,
        )
        assert result is None and case.full_checks == [] and case.inference == []


@pytest.mark.parametrize(
    "mutation",
    [
        "budget_revoke",
        "parent_revoke",
        "parent_scope",
        "parent_operation",
        "budget_control",
        "parent_control",
    ],
)
def test_off_final_clock_callback_cannot_escape_original_fence(monkeypatch, mutation):
    from apps.documents.services.representation_contracts import SourceScopeUnavailable

    from ._native_graph_search_closure_support import completion_clock

    with native_case(monkeypatch, graph_on=False) as case:
        before = (case.source.budget.pairs_used, case.operation.deadline)
        returned = []
        with completion_clock(case, monkeypatch, mutation) as (armed, fired):
            if mutation.endswith("control"):
                returned.append(run_current_or_native(case))
            else:
                with pytest.raises(SourceScopeUnavailable):
                    returned.append(run_current_or_native(case))
            assert armed == [True] and fired == [mutation]
        assert len(returned) == int(mutation.endswith("control"))
        assert case.inference == [] and len(case.full_checks) == 2
        assert (case.source.budget.pairs_used, case.operation.deadline) == before
