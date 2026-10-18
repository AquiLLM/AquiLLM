"""Production dispatch with real metadata capture, native authority and closure."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from django.conf import settings

from apps.documents.models import TextChunk
from apps.documents.services import chunk_search, native_graph_search_closure
from apps.documents.tests.test_graph_source_admission import pressure_rows
from apps.knowledge_graph.tests.test_graph_candidate_operation_work import (
    count_full_authorities,
    graph_frame,
)
from lib import replay_observation
from lib.evidence_observation import observe
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

pytestmark = pytest.mark.django_db(transaction=True)


@contextmanager
def dispatch_case(monkeypatch, *, graph_on=True, fail=False, schedule="native_closure"):
    from apps.documents.services import chunk_rerank
    from apps.documents.services.chunk_rerank_results import rank_only_result_for_chunks
    from apps.documents.tests.test_graph_expired_fallback import install_failure

    monkeypatch.setenv("RAG_ACQUISITION_SCHEDULE", schedule)
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source")
    monkeypatch.setattr(settings, "KG_OVERLAY_ENABLED", graph_on, raising=False)
    monkeypatch.setattr(settings, "RAG_CANDIDATE_MULTIPLIER", 3.0, raising=False)
    monkeypatch.setattr(settings, "RAG_SHORT_QUERY_CANDIDATE_SCALE", 1.0, raising=False)
    monkeypatch.setattr(
        chunk_search.apps.get_app_config("aquillm"), "trigram_top_k", 45)
    monkeypatch.setattr(chunk_search.apps.get_app_config("aquillm"), "vector_top_k", 45)
    monkeypatch.setattr(chunk_search, "load_query_embedding", lambda query: None)
    now = [0.0]
    budget = TurnBudget(TurnLimits(), clock=lambda: now[0])
    with graph_frame(monkeypatch, prepare_sources=pressure_rows, dataset=True,
                     budget=budget, timeout_ms=10_000) as active:
        f, graph, source, scope, _, dependencies, operation, driver = active
        dependencies.settings.memgraph_traversal_enabled = True
        case = SimpleNamespace(
            f=f, graph=graph, source=source, operation=operation, driver=driver,
            dependencies=dependencies, stages=[], events=[], inference=[],
            finals=[], graph_events=[], emitted=[], now=now, scope=scope,
        )
        if fail:
            graph.clock = lambda: now[0]
            install_failure(monkeypatch, active, now, expired=True)
        finalizer = native_graph_search_closure._finalize_closed_ranking
        stage = chunk_search.retrieval_stages
        graph_event = replay_observation.graph_selected_sources_v2

        def final(*args, **kwargs):
            result = finalizer(*args, **kwargs)
            case.finals.append(result)
            case.events.append("finalized")
            return result

        def stages(*args, **kwargs):
            case.stages.append((args, kwargs))
            case.events.append("stage")
            return stage(*args, **kwargs)

        def inference(model, query, rows, top_k, **kwargs):
            case.inference.append(tuple(row.pk for row in rows))
            return rank_only_result_for_chunks(query, rows, top_k)

        def graph_selected(ids):
            case.graph_events.append(ids)
            return graph_event(ids)

        monkeypatch.setattr(
            native_graph_search_closure, "_finalize_closed_ranking", final)
        monkeypatch.setattr(chunk_search, "retrieval_stages", stages)
        monkeypatch.setattr(
            replay_observation, "graph_selected_sources_v2", graph_selected)
        monkeypatch.setattr(chunk_rerank, "rerank_provider", lambda: "local")
        monkeypatch.setattr(chunk_rerank, "rerank_via_local_vllm_scored", inference)
        case.run = lambda: chunk_search.text_chunk_search(
            TextChunk, "Atlas", 15, list(f.documents.values()),
            authorization_context=f.authorization,
            hybrid_graph_dependencies=dependencies if graph_on else None,
        )
        with observe(
            lambda event, data: case.emitted.append((event, data))
        ) as observation:
            yield case
            assert not observation.failed.is_set()


@pytest.mark.parametrize("graph_on,fail", [(True, False), (True, True), (False, False)])
def test_actual_dispatch_closes_once_before_stage_without_acquisition_scoring(
    monkeypatch, graph_on, fail
):
    with dispatch_case(monkeypatch, graph_on=graph_on, fail=fail) as case:
        for name in ("hybrid_graph_candidate_pool", "materialize_and_rerank_candidates",
                     "finalize_ranking"):
            monkeypatch.setattr(chunk_search, name, lambda *a, **k: pytest.fail(
                "native route reentered public scored flow"))
        proofs = count_full_authorities(monkeypatch)
        result = case.run()
        assert case.inference == []
        assert case.events == ["finalized", "stage"]
        # Native acquisition publishes its bounded, freshly authorized pool;
        # The passage cutoff belongs to final relevance selection.
        assert len(case.finals) == 1
        # Graph consumes two additional source identities but publishes one
        # novel passage: 25 initial +18 top-up +1 graph, with all45 charges kept.
        assert len(result[2]) == (25 if fail else 44 if graph_on else 45)
        assert case.source.budget.pairs_used["acquisition"] == 0
        assert case.operation.deadline == 10.0 and case.operation.active()
        assert case.source.budget.sources_used <= 45
        args, observed = case.stages[0]
        assert observed["acquisition_schedule_requested"] == "native_closure"
        assert observed["acquisition_schedule_observed"] == "native_closure"
        assert observed["acquisition_initial_ineligibility_reason"] is None
        assert observed["acquisition_score_status"] == "not_requested"
        assert observed["materialized_union_status"] == "unavailable"
        assert observed["preclosure_pool_count"] >= 25
        assert tuple(args[3].combined_candidates) == tuple(result[2])
        assert tuple(observed["hybrid_pool"]) == tuple(result[2])
        emitted = [data for event, data in case.emitted if event == "retrieval_stages"]
        assert len(emitted) == 1
        assert emitted[0]["acquisition_score_status"] == "not_requested"
        assert emitted[0]["materialized_union"] is None
        assert [row["chunk_id"] for row in emitted[0]["post_rerank"]] == [
            row.pk for row in result[2]]
        if graph_on and not fail:
            assert args[5]["graph_status"] == "hit"
            assert len(proofs) == 2 and case.driver.calls and case.graph_events
            assert case.f.chunks["B"].pk in case.graph_events[0]
        else:
            assert not case.graph_events
            if fail:
                assert args[5]["graph_status"] == "error"
                assert case.source.budget.sources_used == 25


@pytest.mark.parametrize("kind", [
    "default_on", "default_off", "fts", "wrapped", "authorizer", "legacy", "preflight",
    "off_custom"])
def test_ineligible_and_default_dispatch_keep_scored_path(monkeypatch, kind):
    graph_on = kind not in ("default_off", "off_custom")
    with dispatch_case(monkeypatch, graph_on=graph_on) as case:
        if kind.startswith("default"):
            monkeypatch.delenv("RAG_ACQUISITION_SCHEDULE")
        elif kind == "fts":
            monkeypatch.setattr(settings, "RAG_FULL_TEXT_ENABLED", True)
        elif kind == "wrapped":
            run = case.graph.run_v2
            case.graph.run_v2 = lambda **kwargs: run(**kwargs)
        elif kind == "authorizer":
            authorize = chunk_search.authorized_search_rows
            monkeypatch.setattr(chunk_search, "authorized_search_rows",
                                lambda *a, **k: authorize(*a, **k))
        elif kind == "legacy":
            case.dependencies.settings.graph_remediation_mode = "legacy"
            case.dependencies.settings.memgraph_traversal_enabled = False
        elif kind == "preflight":
            # Genuine mismatch: requested docs omit part of the exact authorization.
            case.run = lambda: chunk_search.text_chunk_search(
                TextChunk, "Atlas", 15, [case.f.documents["A"]],
                authorization_context=case.f.authorization,
                hybrid_graph_dependencies=case.dependencies,
            )
        elif kind == "off_custom":
            case.run = lambda: chunk_search.text_chunk_search(
                TextChunk, "Atlas", 15, list(case.f.documents.values()),
                authorization_context=case.f.authorization,
                hybrid_graph_dependencies=case.dependencies,
            )
        result = case.run()
        assert result[2] and case.inference and case.finals == []
        assert case.events == ["stage"]
        args, observed = case.stages[0]
        assert observed["acquisition_schedule_observed"] == "scored"
        assert observed["acquisition_schedule_requested"] == (
            "scored" if kind.startswith("default") else "native_closure")
        assert observed["acquisition_initial_ineligibility_reason"] == (
            None if kind.startswith("default") else "native_ineligible")
        assert observed["acquisition_score_status"] == "unavailable"
        assert observed["materialized_union_status"] == "observed"
        assert observed["preclosure_pool_count"] is None


@pytest.mark.parametrize("mutation", ["permission", "operation"])
def test_failed_native_finalization_never_falls_back_or_emits_stage(
    monkeypatch, mutation
):
    from apps.documents.services.chunk_search_scope import current_scope
    from apps.documents.services.representation_contracts import SourceScopeUnavailable
    from apps.documents.services.source_loading import SourcePreparationLimited

    with dispatch_case(monkeypatch) as case:
        original = native_graph_search_closure._finalize_closed_ranking

        def mutate(*args, **kwargs):
            if mutation == "permission":
                case.f.policy.rows = ()
            else:
                case.operation.live = False
            return original(*args, **kwargs)

        monkeypatch.setattr(
            native_graph_search_closure, "_finalize_closed_ranking", mutate)
        with pytest.raises(
            (SourceScopeUnavailable, SourcePreparationLimited, TimeoutError)
        ):
            case.run()
        assert case.stages == [] and case.inference == []
        assert current_scope() is case.scope  # Search lifetime restored caller scope.


@pytest.mark.parametrize("candidate", ["baseline", "graph"])
def test_current_source_mutation_is_removed_by_single_native_finalizer(
    monkeypatch, candidate
):
    with dispatch_case(monkeypatch) as case:
        original, changed = native_graph_search_closure._finalize_closed_ranking, []

        def mutate(*args, **kwargs):
            changed.append(
                args[1].ranked_results[0].pk if candidate == "baseline"
                else case.f.chunks["B"].pk)
            TextChunk.objects.filter(pk=changed[0]).update(content="changed")
            return original(*args, **kwargs)

        monkeypatch.setattr(
            native_graph_search_closure, "_finalize_closed_ranking", mutate)
        result = case.run()
        assert changed[0] not in {row.pk for row in result[2]}
        assert len(result[2]) == 43 and case.events == ["finalized", "stage"]
        assert case.inference == []
