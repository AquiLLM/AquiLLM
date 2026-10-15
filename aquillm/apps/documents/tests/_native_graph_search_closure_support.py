"""Real captured source fixtures and the preimplementation behavioral adapter."""

from contextlib import contextmanager
from importlib.util import find_spec
from types import SimpleNamespace

from django.conf import settings
from django.db import connection
from django.test.utils import CaptureQueriesContext

from apps.documents.models import TextChunk
from apps.documents.services import chunk_search, chunk_search_scope
from apps.documents.services.closed_search_finalizer import _bind_search_authorizer
from apps.documents.services.hybrid_graph_v2 import hybrid_graph_candidate_pool_v2
from apps.documents.tests.test_graph_expired_fallback import install_failure
from apps.documents.tests.test_graph_source_admission import capture, pressure_rows
from apps.knowledge_graph.tests.test_graph_candidate_operation_work import graph_frame
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


def run_current_or_native(case):
    module = "apps.documents.services.native_graph_search_closure"
    if find_spec(module):
        from apps.documents.services.native_graph_search_closure import (
            _try_native_search_closure,
        )

        return _try_native_search_closure(
            case.snapshot,
            "Atlas",
            15,
            authorize=case.authorize,
            dependencies=case.dependencies if case.graph_on else None,
            overlay_enabled=case.graph_on,
        )
    pool, diagnostics = (
        hybrid_graph_candidate_pool_v2(
            case.snapshot, "Atlas", case.f.authorization, case.dependencies
        )
        if case.graph_on
        else (case.snapshot.baseline_candidates, {})
    )
    snapshot = case.snapshot
    if snapshot.source_admission:
        snapshot = snapshot.source_admission.refresh_snapshot(snapshot, pool)
    ranking = chunk_search.materialize_and_rerank_candidates(
        TextChunk,
        "Atlas",
        15,
        case.authorize(pool),
        authorized_scope=None,
        capture_scores=True,
    )
    snapshot, rows, scores = chunk_search.finalize_ranking(
        snapshot, ranking, authorize=case.authorize, query="Atlas", top_k=15
    )
    return SimpleNamespace(
        snapshot=snapshot,
        final_rows=rows,
        ranking=ranking,
        score_set=scores,
        graph_diagnostics=diagnostics,
    )


@contextmanager
def native_case(
    monkeypatch,
    *,
    graph_on=True,
    fail=True,
    expired=True,
    mutation=None,
    graph_epoch=0.0,
    start=0.0,
    rrf=False,
    controlled_graph_clock=False,
):
    monkeypatch.setenv("RAG_ACQUISITION_SCHEDULE", "native_closure")
    monkeypatch.setattr(settings, "KG_OVERLAY_ENABLED", graph_on, raising=False)
    monkeypatch.setattr(settings, "RAG_CANDIDATE_MULTIPLIER", 3.0, raising=False)
    monkeypatch.setattr(settings, "RAG_SHORT_QUERY_CANDIDATE_SCALE", 1.0, raising=False)
    now = [start]
    budget = TurnBudget(TurnLimits(), clock=lambda: now[0])
    with graph_frame(
        monkeypatch,
        prepare_sources=pressure_rows,
        dataset=True,
        budget=budget,
        timeout_ms=10_000,
    ) as active:
        f, graph, source, scope, _, dependencies, operation, driver = active
        if fail or controlled_graph_clock:
            graph.clock = lambda: graph_epoch + now[0]
        monkeypatch.setattr(settings, "RAG_RRF_ENABLED", rrf)
        monkeypatch.setattr(settings, "RAG_PROTECTED_EXACT_ENABLED", rrf, raising=False)
        scope.graph_enabled = graph_on
        snapshot = capture(active, native=graph_on)
        authorize = _bind_search_authorizer(
            chunk_search.authorized_search_rows, f.authorization, graph_on
        )
        case = SimpleNamespace(
            active=active,
            f=f,
            graph=graph,
            source=source,
            scope=scope,
            snapshot=snapshot,
            dependencies=dependencies,
            operation=operation,
            driver=driver,
            authorize=authorize,
            now=now,
            graph_on=graph_on,
            full_checks=[],
            inference=[],
            fetched=[],
            events=[],
            after_graph=not graph_on,
            after_graph_query_index=None,
        )
        if fail and graph_on:
            from apps.knowledge_graph.retrieval import (
                ready_scope_repository_v2 as ready,
            )

            _, case.fetched, case.events = install_failure(
                monkeypatch, active, now, expired=expired, mutation=mutation
            )
            select = ready._select_closed_ready_scope_v2

            def failed(**kwargs):
                try:
                    return select(**kwargs)
                finally:
                    case.after_graph = True
                    case.after_graph_query_index = len(connection.queries)

            monkeypatch.setattr(ready, "_select_closed_ready_scope_v2", failed)
        validate = chunk_search_scope.FrozenRetrievalScope.validate

        def checked(current):
            result = validate(current)
            if case.after_graph:
                case.full_checks.append(current)
            return result

        monkeypatch.setattr(
            chunk_search_scope.FrozenRetrievalScope, "validate", checked
        )
        from apps.documents.services import chunk_rerank
        from apps.documents.services.chunk_rerank_results import (
            rank_only_result_for_chunks,
        )

        def inference(model, query, rows, top_k, **kwargs):
            case.inference.append(tuple(row.pk for row in rows))
            return rank_only_result_for_chunks(query, rows, top_k)

        # Replace only the external scoring adapter, retaining actual materializer,
        # scoring dispatch, authorizer, and finalizer in the old-path RED adapter.
        monkeypatch.setattr(chunk_rerank, "rerank_provider", lambda: "local")
        monkeypatch.setattr(chunk_rerank, "rerank_via_local_vllm_scored", inference)
        with CaptureQueriesContext(connection) as statements:
            case.statements = statements
            if not graph_on:
                case.after_graph_query_index = len(connection.queries)
            yield case


def owner_queries(case):
    start = case.after_graph_query_index - case.statements.initial_queries
    return [
        s
        for s in case.statements.captured_queries[start:]
        if "UNION ALL" in s["sql"] and "sha256" in s["sql"]
    ]


@contextmanager
def dense_case(transport, monkeypatch, settings):
    from apps.documents.services.chunk_search_candidates import HybridCandidateSnapshot
    from apps.documents.services.chunk_search_scoped_rows import dense_rows
    from apps.documents.services.source_loading import source_runtime_scope
    from apps.documents.tests.test_source_metadata_closed_operation import captured
    from lib.embeddings.document_provider import observe_query_embedding
    from lib.retrieval.operation import operation_scope

    case, scope, _, _ = captured(1, "ready", transport, monkeypatch, settings)
    settings.RAG_FULL_TEXT_ENABLED = False
    settings.RAG_RRF_ENABLED = False
    settings.KG_OVERLAY_ENABLED = False
    monkeypatch.setenv("RAG_ACQUISITION_SCHEDULE", "native_closure")
    token = chunk_search_scope.bind_search_scope(scope, "source", graph_enabled=False)
    try:
        with (
            source_runtime_scope(scope.runtime),
            operation_scope(scope.runtime.budget, timeout_ms=10_000),
        ):
            observation = observe_query_embedding(
                "source", space=scope.space, budget=scope.runtime.budget
            )
            rows = dense_rows(TextChunk, observation, scope, 5)
            assert len(rows) == 1
            snapshot = HybridCandidateSnapshot(
                case.docs,
                rows,
                (),
                (),
                tuple(r.pk for r in rows),
                (),
                (),
                rows,
                (),
                (),
                None,
                0.0,
                0.0,
                0.0,
                1,
                False,
            )
            authorize = _bind_search_authorizer(
                chunk_search.authorized_search_rows, scope.authorization, False
            )
            yield scope, snapshot, authorize
    finally:
        chunk_search_scope.reset_search_scope(token)
        scope.runtime.budget.close("native_dense_test_finished")


@contextmanager
def completion_clock(case, monkeypatch, mutation):
    from dataclasses import replace

    from apps.documents.services import native_graph_search_closure as closure
    from apps.documents.services.source_deadline import _DEADLINE
    from lib.retrieval.operation import _OPERATION

    armed, fired, tokens = [], [], []
    original = closure._NativeSearchClosureResult

    def completed(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result.final_rows[2]  # Real unchanged finalizer has completed.
        armed.append(True)
        return result

    def clock():
        if armed and not fired:
            fired.append(mutation)
            if mutation.endswith("revoke"):
                case.operation.live = False
            elif mutation == "parent_scope":
                tokens.append(
                    (
                        chunk_search_scope._SCOPE,
                        chunk_search_scope._SCOPE.set(replace(case.scope)),
                    )
                )
            elif mutation == "parent_operation":
                tokens.append((_OPERATION, _OPERATION.set(replace(case.operation))))
        return case.now[0]

    monkeypatch.setattr(closure, "_NativeSearchClosureResult", completed)
    if mutation.startswith("budget"):
        monkeypatch.setattr(case.source.budget, "_clock", clock)
    else:
        tokens.append((_DEADLINE, _DEADLINE.set((100.0, clock, None))))
    try:
        yield armed, fired
    finally:
        for variable, token in reversed(tokens):
            variable.reset(token)
