"""An optional baseline read cannot publish unfinished evidence or erase an on-time graph."""

from time import monotonic
from types import SimpleNamespace

import pytest
from django.db import OperationalError

from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.source_loading import SourcePreparationLimited

from ._native_graph_search_closure_support import native_case, run_current_or_native


pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize(
    "expire", [
        "optional", "graph", "operation", "permission", "early_limit", "unrelated_limit",
        "early_statement", "foreign_cancel", "psycopg2_timeout", "psycopg3_timeout",
    ],
)
def test_optional_baseline_cutoff_preserves_only_a_live_verified_graph(monkeypatch, expire):
    from apps.documents.services.graph_source_admission import NativeSourceAdmission
    from apps.documents.services.graph_source_admission_policy import charged_sources
    from apps.documents.services.source_deadline import _DEADLINE, check_source_deadline
    from apps.knowledge_graph.retrieval.graph_candidate_stage_v2 import _GraphSourceStage
    from lib import replay_observation

    with native_case(
        monkeypatch, fail=False, controlled_graph_clock=True, graph_epoch=monotonic(),
    ) as case:
        ids = NativeSourceAdmission.top_up_ids
        read = _GraphSourceStage.read_captured
        state = {"pending": False}
        observed, extras = [], set()
        initial_bodies = case.scope.bodies.copy()
        initial_cache = case.source.cache.copy()

        def top_up(plan):
            selected = ids(plan)
            if selected:
                state["pending"] = True
            return selected

        def late_read(stage, refs):
            rows = read(stage, refs)
            if state.pop("pending", False) and not extras:
                extras.update(row.pk for row in rows)
                assert extras
                assert case.scope.bodies == initial_bodies and case.source.cache == initial_cache
                bound, clock, _ = _DEADLINE.get()
                if expire == "operation":
                    case.operation.live = False
                elif expire == "early_limit":
                    raise SourcePreparationLimited("unrelated source limit")
                else:
                    if expire == "permission":
                        case.f.policy.rows = ()
                    if expire != "early_statement":
                        case.now[0] += bound - clock() + (1.5 if expire == "graph" else 0.001)
                    if expire == "unrelated_limit":
                        raise SourcePreparationLimited("source_runtime_unavailable")
                    if expire in {"early_statement", "foreign_cancel", "psycopg2_timeout", "psycopg3_timeout"}:
                        cause = Exception()
                        setattr(cause, "sqlstate" if expire == "psycopg3_timeout" else "pgcode", "57014")
                        cause.diag = SimpleNamespace(message_primary=(
                            "canceling statement due to user request" if expire == "foreign_cancel"
                            else "canceling statement due to statement timeout"
                        ))
                        raise OperationalError("query canceled") from cause
                check_source_deadline()
            return rows

        monkeypatch.setattr(NativeSourceAdmission, "top_up_ids", top_up)
        monkeypatch.setattr(_GraphSourceStage, "read_captured", late_read)
        monkeypatch.setattr(replay_observation, "graph_selected_sources_v2", observed.append)
        if expire in {"operation", "permission"}:
            with pytest.raises((SourceScopeUnavailable, SourcePreparationLimited)):
                run_current_or_native(case)
            assert not observed
            return

        result = run_current_or_native(case)
        assert extras
        if expire in {"graph", "early_limit", "unrelated_limit", "early_statement", "foreign_cancel"}:
            assert not observed
            assert result.graph_diagnostics["graph_status"] != "hit"
            return

        assert result.graph_diagnostics["graph_status"] == "hit"
        graph_id = case.f.chunks["B"].pk
        assert any(graph_id in group for group in observed)
        rows = {row.pk for row in result.final_rows[2]}
        initial = {row.pk for row in case.snapshot.baseline_candidates}
        authoritative = initial.union(pk for group in observed for pk in group)
        assert graph_id in rows and rows <= authoritative
        discarded = extras - authoritative
        assert discarded and not discarded & set(case.scope.bodies)
        assert not discarded & {row.pk for row in case.source.cache.values()}
        assert discarded <= {pk for pk, _ in charged_sources(case.source.budget)}
        assert result.graph_diagnostics["source_admission"]["baseline_top_up"] == 0


def test_empty_graph_keeps_the_full_baseline_read_window(monkeypatch):
    from apps.documents.services.graph_source_admission import NativeSourceAdmission
    from apps.documents.services.source_deadline import _DEADLINE
    from apps.knowledge_graph.tests.test_closed_graph_limits_v2 import returned_envelope

    with native_case(
        monkeypatch, fail=False, controlled_graph_clock=True, graph_epoch=monotonic(),
    ) as case:
        bounds = []
        returned_envelope(
            monkeypatch, empty=True, after=lambda *args: bounds.append(_DEADLINE.get()),
        )
        top_up_ids = NativeSourceAdmission.top_up_ids

        def fill(plan):
            assert bounds
            bounds.append(_DEADLINE.get())
            return top_up_ids(plan)

        monkeypatch.setattr(NativeSourceAdmission, "top_up_ids", fill)
        result = run_current_or_native(case)
        assert result.graph_diagnostics["source_admission"]["baseline_top_up"] > 0
        assert len(bounds) == 2 and bounds[1] is bounds[0]

@pytest.mark.parametrize("offset", [0.001, -0.0005])
def test_late_materialization_stops_before_the_next_provenance_read(monkeypatch, offset):
    from apps.documents.services.source_deadline import _DEADLINE
    from apps.knowledge_graph.retrieval import ready_materialization_v2 as materializer
    from lib import replay_observation

    with native_case(
        monkeypatch, fail=False, controlled_graph_clock=True, graph_epoch=monotonic(),
    ) as case:
        reads, events = [], []
        materializing = False
        original = materializer._materialize_closed_rows
        provenance = materializer.provenance_for_chunk

        def materialize(**kwargs):
            nonlocal materializing
            materializing = True
            try:
                return original(**kwargs)
            finally:
                materializing = False

        def delayed(**kwargs):
            value = provenance(**kwargs)
            if materializing:
                reads.append(kwargs["chunk"].pk)
                if len(reads) == 1:
                    deadline, clock, _ = _DEADLINE.get()
                    case.now[0] += deadline - clock() + offset
            return value

        monkeypatch.setattr(materializer, "_materialize_closed_rows", materialize)
        monkeypatch.setattr(materializer, "provenance_for_chunk", delayed)
        monkeypatch.setattr(replay_observation, "graph_selected_sources_v2", events.append)
        result = run_current_or_native(case)
        assert len(reads) == 1
        assert not events
        failure = result.graph_diagnostics["graph_failure"]
        assert (failure["phase"], failure["reason"]) == ("materialization", "deadline")
        assert result.graph_diagnostics["source_admission"]["baseline_completion_attempted"]
