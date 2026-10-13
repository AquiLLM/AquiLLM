"""Actual body/authority changes abort the unpublished outer native graph operation."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from uuid import uuid4

import pytest

from apps.documents.services.chunk_search_scoped_rows import fingerprint
from apps.knowledge_graph.tests.test_graph_candidate_operation_work import (
    graph_frame,
    run_graph,
)
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

pytestmark = pytest.mark.django_db(transaction=True)


def after_hydration(monkeypatch, callback):
    from apps.knowledge_graph.retrieval import graph_runtime_execution_v2 as execution

    original = execution.select_graph_evidence_v2

    def select(**kwargs):
        result = original(**kwargs)
        assert result.candidates, (
            "fault must follow actual successful private hydration"
        )
        callback()
        return result

    monkeypatch.setattr(execution, "select_graph_evidence_v2", select)


def test_actual_hydrated_bodies_and_fingerprints_remain_private_until_outer_publication(
    monkeypatch,
):
    with graph_frame(monkeypatch) as active:
        _, _, source, scope, _, _, _, _ = active
        observed = []

        def inspect():
            assert source.budget.sources_used == 2
            assert not source.cache
            assert not scope.bodies
            observed.append(True)

        after_hydration(monkeypatch, inspect)
        rows, diagnostics = run_graph(active)
        assert observed == [True]
        assert len(rows) == 2 and diagnostics["graph_status"] == "hit"
        assert len(source.cache) == 2
        assert set(scope.bodies) == {row.pk for row in rows}


@pytest.mark.parametrize(
    "change",
    [
        "membership",
        "generation",
        "permission",
        "source",
        "budget",
        "cache",
        "authorization",
        "operation",
        "vocabulary_header",
        "ranking",
        "nested",
    ],
)
def test_actual_post_body_change_never_publishes_graph_staging_or_adopts_new_authority(
    monkeypatch, change
):
    from django.conf import settings

    from apps.knowledge_graph.models import CollectionEntity, CollectionGraphProjection

    with graph_frame(monkeypatch) as active:
        f, _, source, scope, _, _, operation, _ = active
        original_budget, original_cache = source.budget, source.cache
        before_generation = f.projections["B"].generation_key

        def mutate():
            assert original_budget.sources_used == 2
            assert not original_cache and not scope.bodies

            def committed():
                from django.db import connection, transaction

                try:
                    with transaction.atomic():
                        if change == "membership":
                            assert (
                                CollectionEntity.objects.filter(
                                    artifact=f.projections["B"].artifact
                                ).update(status="superseded")
                                > 0
                            )
                        elif change == "generation":
                            assert (
                                CollectionGraphProjection.objects.filter(
                                    pk=f.projections["B"].pk
                                ).update(generation_key=uuid4())
                                == 1
                            )
                        else:
                            f.documents["B"].full_text = "Changed original owner"
                            f.documents["B"].save(dont_rechunk=True)
                finally:
                    connection.close()

            if change in {"membership", "generation", "source"}:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    pool.submit(committed).result(timeout=10)
            elif change == "permission":
                f.policy.rows = ()
            elif change == "budget":
                source.budget = TurnBudget(TurnLimits())
            elif change == "cache":
                source.cache = {}
            elif change == "authorization":
                source.authorization = replace(source.authorization)
            elif change == "operation":
                operation.live = False
            elif change == "vocabulary_header":
                scope.vocabulary_digest = "replaced-vocabulary"
            elif change == "ranking":
                monkeypatch.setattr(settings, "RAG_RRF_ENABLED", True)
            else:
                scope.bodies[f.chunks["A"].pk] = fingerprint(f.chunks["A"])

        after_hydration(monkeypatch, mutate)
        rows, diagnostics = run_graph(active)
        assert diagnostics["graph_status"] == "error"
        assert f.chunks["B"].pk not in {row.pk for row in rows}
        assert f.chunks["B"].pk not in scope.bodies
        assert not original_cache
        assert original_budget.sources_used == 2
        assert original_budget.text_used["materialized"] == 2
        if change == "budget":
            assert source.budget.sources_used == 0 and not source.cache
        if change == "nested":
            assert scope.bodies[f.chunks["A"].pk] == fingerprint(f.chunks["A"])
        if change == "generation":
            f.projections["B"].refresh_from_db()
            assert f.projections["B"].generation_key != before_generation


@pytest.mark.parametrize("text", ["Z", "Longer changed chunk"])
def test_late_actual_same_length_or_growing_body_cannot_replace_captured_fingerprint(
    monkeypatch, text
):
    from apps.documents.models import TextChunk

    with graph_frame(monkeypatch) as active:
        f, _, source, scope, _, _, _, _ = active

        def mutate():
            assert (
                TextChunk.objects.filter(pk=f.chunks["B"].pk).update(content=text) == 1
            )

        after_hydration(monkeypatch, mutate)
        rows, diagnostics = run_graph(active)
        assert diagnostics["graph_status"] == "error"
        assert f.chunks["B"].pk not in {row.pk for row in rows}
        assert f.chunks["B"].pk not in scope.bodies and not source.cache
        assert source.budget.text_used["materialized"] >= 2


def test_preparation_context_cleanup_precedes_positive_graph_publication(monkeypatch):
    from apps.knowledge_graph.retrieval import graph_candidate_operation_v2 as owner

    with graph_frame(monkeypatch) as active:
        _, _, source, scope, _, _, _, _ = active
        original, original_cache, changed = (
            owner.source_preparation_scope,
            source.cache,
            [],
        )

        @contextmanager
        def preparation(*args):
            with original(*args):
                yield
            changed.append(True)
            source.cache = {}

        monkeypatch.setattr(owner, "source_preparation_scope", preparation)
        rows, diagnostics = run_graph(active)
        assert changed == [True]
        assert rows == () and diagnostics["graph_status"] == "error"
        assert not original_cache and not source.cache and not scope.bodies
        assert source.budget.sources_used == 2
