"""Private closure never adopts replacement authority or emits stale pool bodies."""

from dataclasses import replace

import pytest

from apps.documents.models import TextChunk
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.source_loading import SourcePreparationLimited

from ._native_graph_search_closure_support import native_case, run_current_or_native

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("change", ["persisted", "hydrated"])
@pytest.mark.parametrize("position", [0, 24])
def test_changed_body_is_withdrawn_from_every_completed_payload(
    monkeypatch, change, position
):
    from apps.documents.services import native_graph_search_closure as closure

    with native_case(monkeypatch) as case:
        row = case.snapshot.baseline_candidates[position]
        original = closure._finalize_closed_ranking

        def final(*args, **kwargs):
            if change == "persisted":
                TextChunk.objects.filter(pk=row.pk).update(
                    content="committed body edit"
                )
            else:
                row.content = "independently edited hydrated body"
            return original(*args, **kwargs)

        monkeypatch.setattr(closure, "_finalize_closed_ranking", final)
        result = run_current_or_native(case)
        groups = (
            *result.final_rows,
            result.hybrid_pool,
            result.ranking.combined_candidates,
            result.snapshot.baseline_candidates,
        )
        assert all(row.pk not in {r.pk for r in group} for group in groups)
        assert len(result.final_rows[2]) == 24 and result.score_set is None


@pytest.mark.parametrize("mutation", ["permission", "source", "operation"])
def test_failed_graph_authority_loss_never_returns_packet(monkeypatch, mutation):
    with native_case(monkeypatch, mutation=mutation) as case:
        with pytest.raises(SourceScopeUnavailable):
            run_current_or_native(case)
        assert case.events == [] and case.fetched == [] and case.inference == []


@pytest.mark.parametrize(
    "mutation",
    [
        "settings",
        "query",
        "admission",
        "witness",
        "fts",
        "schedule",
        "deadline",
        "authorization",
        "overlay",
    ],
)
def test_material_replacement_during_graph_never_switches_route(monkeypatch, mutation):
    from django.conf import settings

    from apps.knowledge_graph.retrieval import ready_scope_repository_v2 as ready

    with native_case(monkeypatch) as case:
        original = ready._select_closed_ready_scope_v2

        def select(**kwargs):
            result = original(**kwargs)
            if mutation == "settings":
                case.dependencies.settings.graph_overall_timeout_ms = 4999
            elif mutation == "query":
                case.scope.query_digest = "a" * 64
            elif mutation == "admission":
                object.__setattr__(case.snapshot, "source_admission", None)
            elif mutation == "witness":
                case.source._frozen_retrieval_scope = replace(
                    case.source._frozen_retrieval_scope
                )
            elif mutation == "fts":
                settings.RAG_FULL_TEXT_ENABLED = True
            elif mutation == "schedule":
                monkeypatch.setenv("RAG_ACQUISITION_SCHEDULE", "scored")
            elif mutation == "deadline":
                case.operation.deadline += 1
            elif mutation == "overlay":
                settings.KG_OVERLAY_ENABLED = False
            else:
                object.__setattr__(
                    case.authorize, "authorization", replace(case.f.authorization)
                )
            return result

        monkeypatch.setattr(ready, "_select_closed_ready_scope_v2", select)
        with pytest.raises(SourceScopeUnavailable):
            run_current_or_native(case)
        assert case.inference == [] and case.fetched == [] and case.events == []


@pytest.mark.parametrize(
    "when",
    [
        "raise",
        "none",
        "after",
        "finish",
        "cleanup",
        "after_settings",
        "after_authority",
    ],
)
def test_finalizer_is_once_and_cannot_escape_revoked_original(monkeypatch, when):
    from contextlib import contextmanager

    from apps.documents.services import chunk_search_authorization as authorization
    from apps.documents.services import closed_search_finalizer as owner
    from apps.documents.services import native_graph_search_closure as closure

    with native_case(monkeypatch) as case:
        original, calls = closure._finalize_closed_ranking, []

        def final(*args, **kwargs):
            calls.append(True)
            if when == "raise":
                raise SourceScopeUnavailable("synthetic finalizer failure")
            if when == "none":
                return None
            result = original(*args, **kwargs)
            if when == "after":
                case.operation.live = False
            elif when == "after_settings":
                case.dependencies.settings.graph_overall_timeout_ms = 4999
            elif when == "after_authority":
                case.source.authorization = replace(case.source.authorization)
            return result

        if when == "finish":
            finish = authorization._finish_search_rows

            def revoke(*args, **kwargs):
                result = finish(*args, **kwargs)
                case.operation.live = False
                return result

            monkeypatch.setattr(authorization, "_finish_search_rows", revoke)
        if when == "cleanup":
            database = owner.bounded_source_database

            @contextmanager
            def cleanup(*args):
                with database(*args):
                    yield
                case.operation.live = False

            monkeypatch.setattr(owner, "bounded_source_database", cleanup)
        monkeypatch.setattr(closure, "_finalize_closed_ranking", final)
        with pytest.raises((SourceScopeUnavailable, SourcePreparationLimited)):
            run_current_or_native(case)
        assert calls == [True] and case.inference == [] and case.events == []


def test_committed_graph_survives_later_closure_failure(monkeypatch):
    from apps.documents.services import native_graph_search_closure as closure
    from lib import replay_observation

    with native_case(monkeypatch, fail=False) as case:
        events = []
        monkeypatch.setattr(
            replay_observation, "graph_selected_sources_v2", events.append
        )

        def fail(*args, **kwargs):
            assert case.f.chunks["B"].pk in case.scope.bodies and events
            raise SourceScopeUnavailable("later closure failure")

        monkeypatch.setattr(closure, "_finalize_closed_ranking", fail)
        with pytest.raises(SourceScopeUnavailable):
            run_current_or_native(case)
        assert events and case.source.cache
        assert case.inference == []


def test_novel_hydration_failure_keeps_charges_without_graph_positives(monkeypatch):
    from apps.knowledge_graph.retrieval import ready_materialization_v2 as materializer
    from lib import replay_observation

    with native_case(monkeypatch, fail=False) as case:
        initial = {r.pk for r in case.snapshot.baseline_candidates}
        before, events = case.source.budget.sources_used, []
        monkeypatch.setattr(
            replay_observation, "graph_selected_sources_v2", events.append
        )

        def fail(**kwargs):
            assert case.source.budget.sources_used > before
            raise ValueError("after actual graph hydration")

        monkeypatch.setattr(materializer, "_materialize_rows", fail)
        result = run_current_or_native(case)
        assert result.graph_diagnostics["graph_status"] == "error"
        assert before < case.source.budget.sources_used <= 45 and events == []
        assert set(case.scope.bodies) == initial
        assert all(row.pk in initial for row in case.source.cache.values())
        assert all(row.pk in initial for row in result.hybrid_pool)
        assert result.preclosure_pool_count == 25


@pytest.mark.parametrize("change", ["permission", "source", "empty_permission"])
def test_final_owner_rechecks_after_incoming_authorization(monkeypatch, change):
    from apps.documents.services import chunk_search_authorization as owner

    with native_case(monkeypatch) as case:
        original = owner.reauthorized_baseline

        def incoming(*args, **kwargs):
            result = original(*args, **kwargs)
            if change == "source":
                from hashlib import sha256

                document = case.f.documents["A"]
                document.full_text += " changed"
                document.full_text_hash = sha256(
                    document.full_text.encode()
                ).hexdigest()
                document.save(dont_rechunk=True)
            else:
                case.f.policy.rows = ()
            return ((), False) if change == "empty_permission" else result

        monkeypatch.setattr(owner, "reauthorized_baseline", incoming)
        with pytest.raises(SourceScopeUnavailable):
            run_current_or_native(case)
        assert case.inference == [] and case.events == []


def test_equal_scalar_query_and_header_values_are_not_identity_guards(monkeypatch):
    from apps.knowledge_graph.retrieval import ready_scope_repository_v2 as owner

    with native_case(monkeypatch) as case:
        original, digest = owner._select_closed_ready_scope_v2, case.scope.query_digest

        def select(**kwargs):
            result = original(**kwargs)
            case.scope.query_digest = (" " + digest)[1:]
            case.scope.using = (" " + case.scope.using)[1:]
            header = case.authorize.header
            object.__setattr__(case.authorize, "header", ((*header[0],), (*header[1],)))
            assert (
                case.scope.query_digest == digest
                and case.scope.query_digest is not digest
            )
            return result

        monkeypatch.setattr(owner, "_select_closed_ready_scope_v2", select)
        assert run_current_or_native(case).final_rows[2]
