"""Successful actual graph capture/frontier/body/materialization work."""

from contextlib import contextmanager
from time import monotonic

import pytest

from apps.documents.services.chunk_search_candidates import HybridCandidateSnapshot
from apps.documents.services.chunk_search_scope import (
    bind_search_scope,
    freeze_search_scope,
    reset_search_scope,
)
from apps.documents.services.hybrid_graph_authorization import (
    HybridGraphRetrievalDependencies,
)
from apps.documents.services.hybrid_graph_v2 import hybrid_graph_candidate_pool_v2
from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from apps.knowledge_graph.tests.graph_runtime_fixture_v2 import persisted_runtime
from lib.retrieval.operation import operation_scope
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

pytestmark = pytest.mark.django_db(transaction=True)


@contextmanager
def graph_frame(
    monkeypatch,
    *,
    requested=None,
    limits=None,
    budget=None,
    timeout_ms=None,
    prepare_sources=None,
    dataset=False,
):
    from django.conf import settings as application_settings

    f, graph, settings, driver = persisted_runtime(monkeypatch, dataset=dataset)
    if prepare_sources:
        prepare_sources(f)
    monkeypatch.setattr(
        application_settings, "RAG_FULL_TEXT_ENABLED", False, raising=False
    )
    monkeypatch.setattr(application_settings, "RAG_RRF_ENABLED", False, raising=False)
    settings.graph_overall_timeout_ms = 5000
    settings.graph_direct_timeout_ms = 5000
    graph.settings = settings
    source = SourceRuntime(
        budget or TurnBudget(limits or TurnLimits()), f.authorization
    )
    with (
        source_runtime_scope(source),
        operation_scope(source.budget, timeout_ms=timeout_ms) as operation,
    ):
        docs = tuple(
            d for d in f.documents.values() if requested is None or d.id in requested
        )
        scope = freeze_search_scope(
            docs, using="default", authorization=f.authorization
        )
        token = bind_search_scope(scope, "What is A?", graph_enabled=True)
        baseline = tuple(
            f.chunks[label]
            for label in ("A",)
            if f.chunks[label].doc_id in scope.requested_document_ids
        )
        snapshot = HybridCandidateSnapshot(
            docs,
            baseline,
            (),
            (),
            tuple(row.pk for row in baseline),
            (),
            (),
            baseline,
            (),
            (),
            None,
            0.0,
            0.0,
            0.0,
            len(baseline),
            False,
        )
        dependencies = HybridGraphRetrievalDependencies(
            graph, settings, graph.materialize
        )
        try:
            yield f, graph, source, scope, snapshot, dependencies, operation, driver
        finally:
            reset_search_scope(token)


def run_graph(active):
    f, _, _, _, snapshot, dependencies, _, _ = active
    return hybrid_graph_candidate_pool_v2(
        snapshot, "What is A?", f.authorization, dependencies
    )


def count_full_authorities(monkeypatch):
    from apps.knowledge_graph.retrieval import ready_materialization_v2 as materializer
    from apps.knowledge_graph.retrieval import ready_scope_repository_v2 as repository

    original = repository.load_authorities_v2
    calls = []

    def load(**kwargs):
        result = original(**kwargs)
        calls.append(
            (kwargs["authorization"], kwargs["representation_snapshot"], result)
        )
        return result

    monkeypatch.setattr(repository, "load_authorities_v2", load)
    monkeypatch.setattr(materializer, "load_authorities_v2", load)
    return calls


@pytest.mark.parametrize("repeated", [False, True])
def test_successful_nonempty_native_graph_pool_has_two_fresh_full_membership_loads(
    monkeypatch, repeated
):
    with graph_frame(monkeypatch) as active:
        f, _, source, scope, _, _, _, driver = active
        if repeated:
            warm_rows, warm_diagnostics = run_graph(active)
            assert warm_rows and warm_diagnostics["graph_status"] == "hit"
        charged = source.budget.text_used.copy()
        calls = count_full_authorities(monkeypatch)
        rows, diagnostics = run_graph(active)
        assert diagnostics["graph_status"] == "hit"
        assert {row.pk for row in rows} == {f.chunks["A"].pk, f.chunks["B"].pk}
        assert all(
            row.content == ("A" if row.pk == f.chunks["A"].pk else "B") for row in rows
        )
        assert len(driver.calls) > 0
        assert source.budget.sources_used == 2
        assert source.budget.text_used["materialized"] == 2
        if repeated:
            assert source.budget.text_used == charged
        assert set(scope.bodies) == {row.pk for row in rows}
        assert all(
            auth is source.authorization and snapshot is scope.representation
            for auth, snapshot, _ in calls
        )
        assert all(len(authorities) == 2 for _, _, authorities in calls)
        assert len(calls) == 2


def test_standalone_public_runtime_and_materializer_keep_independent_membership_fences(
    monkeypatch,
):
    from apps.knowledge_graph.retrieval.ready_materialization_v2 import (
        materialize_graph_candidates_v2,
    )

    with graph_frame(monkeypatch) as active:
        _, graph, source, scope, snapshot, dependencies, _, _ = active
        calls = count_full_authorities(monkeypatch)
        result = graph.run_v2(
            query="What is A?",
            authorization=source.authorization,
            representation_snapshot=scope.representation,
            budget=source.budget,
            settings=dependencies.settings,
            deadline=monotonic() + 5,
            baseline=snapshot,
            requested_document_ids=scope.requested_document_ids,
        )
        assert result.handoff.candidates
        rows = materialize_graph_candidates_v2(
            selected=result.selection,
            refs=result.handoff.candidates,
            authorization=source.authorization,
        )
        assert len(rows) == 2
        assert len(calls) == 5


def test_actual_final_publication_retains_captured_runtime_lock_after_database_cleanup(
    monkeypatch,
):
    with graph_frame(monkeypatch) as active:
        _, _, source, _, _, _, _, _ = active
        calls = count_full_authorities(monkeypatch)
        inner, depth, observed = source.lock, [], []

        class Lock:
            def __enter__(self):
                depth.append(True)
                return inner.__enter__()

            def __exit__(self, *args):
                depth.pop()
                return inner.__exit__(*args)

        source.lock = Lock()
        original = source.budget.publish

        def publish(callback):
            if len(calls) == 2:
                assert depth, "outer staged commit must retain original runtime lock"
                observed.append(True)
            return original(callback)

        monkeypatch.setattr(source.budget, "publish", publish)
        rows, diagnostics = run_graph(active)
        assert observed and len(rows) == 2 and diagnostics["graph_status"] == "hit"


def test_45_actual_source_admissions_prevent_novel_graph_body_without_budget_increase(
    monkeypatch,
):
    from apps.documents.models import TextChunk
    from apps.documents.services.source_loading import source_query_rows

    pks = []

    def prepare(f):
        first = f.chunks["A"]
        pks.append(first.pk)
        text = f.documents["A"].full_text
        for number in range(44):
            pks.append(
                TextChunk.objects.create(
                    doc_id=first.doc_id,
                    content=text[number : number + 1],
                    chunk_number=100 + number,
                    start_position=number,
                    end_position=number + 1,
                ).pk
            )

    with graph_frame(monkeypatch, prepare_sources=prepare, dataset=True) as active:
        f, _, source, _, _, _, _, _ = active
        baseline = source_query_rows(
            TextChunk.objects.filter(pk__in=pks).order_by("pk"), runtime=source
        )
        assert len(baseline) == 45 and source.budget.sources_used == 45
        charged = source.budget.text_used.copy()
        rows, diagnostics = run_graph(active)
        assert diagnostics["graph_status"] == "miss"
        assert f.chunks["B"].pk not in {row.pk for row in rows}
        assert f.chunks["B"].pk not in {row.pk for row in source.cache.values()}
        assert source.budget.sources_used == source.budget.limits.unique_sources == 45
        assert source.budget.text_used == charged
        assert diagnostics["graph_coverage_v2"]["status"] == "partial"


def test_custom_runtime_callback_retains_independently_guarded_six_load_path(
    monkeypatch,
):
    with graph_frame(monkeypatch) as active:
        _, graph, _, _, _, _, _, _ = active
        original = graph.run_v2
        graph.run_v2 = lambda **kwargs: original(**kwargs)
        calls = count_full_authorities(monkeypatch)
        rows, diagnostics = run_graph(active)
        assert len(rows) == 2 and diagnostics["graph_status"] == "hit"
        assert len(calls) == 6


def test_class_wrapped_runtime_callback_retains_public_independent_fences(monkeypatch):
    from apps.knowledge_graph.retrieval.production_runtime import (
        ProductionHybridBranchRuntime,
    )

    original = ProductionHybridBranchRuntime.run_v2

    def wrapped(self, **kwargs):
        return original(self, **kwargs)

    monkeypatch.setattr(ProductionHybridBranchRuntime, "run_v2", wrapped)
    with graph_frame(monkeypatch) as active:
        calls = count_full_authorities(monkeypatch)
        rows, diagnostics = run_graph(active)
        assert len(rows) == 2 and diagnostics["graph_status"] == "hit"
        assert len(calls) == 6
