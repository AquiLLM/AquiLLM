"""Focused native closed-seed policy contracts."""

import pytest
from django.db import connection

from apps.documents.models import RawTextDocument, TextChunk
from apps.documents.tests.test_graph_source_admission import capture
from apps.knowledge_graph.tests.test_graph_candidate_operation_work import graph_frame

from .native_closed_seed_policy_fixtures import (
    small_native_scope,
)

pytestmark = pytest.mark.django_db(transaction=True, databases="__all__")


@pytest.mark.parametrize(
    "change", ["body", "source_hash", "representation", "generation"]
)
def test_native_aggregate_keeps_late_source_and_projection_publication_fences(
    monkeypatch, change
):
    from uuid import uuid4

    from apps.documents.models import DocumentRepresentationState
    from apps.documents.services.hybrid_graph_v2 import hybrid_graph_candidate_pool_v2
    from apps.knowledge_graph.models import CollectionGraphProjection
    from apps.knowledge_graph.tests.test_graph_candidate_operation_races import (
        after_hydration,
    )

    with graph_frame(
        monkeypatch, prepare_sources=small_native_scope, dataset=True
    ) as active:
        f, graph, source, scope, _, dependencies, _, _ = active
        graph.authorization = scope.authorization
        dependencies.settings.graph_direct_enabled = False
        dependencies.settings.graph_extended_enabled = True
        dependencies.settings.graph_extended_timeout_ms = 5000
        snapshot = capture(active)
        proofs, observed = [], []

        def track(execute, sql, params, many, context):
            result = execute(sql, params, many, context)
            if sql.startswith("WITH native_seed_policy_permissions"):
                proofs.append(context["cursor"].rowcount)
            return result

        def mutate():
            assert proofs == [1, 1]
            observed.append(True)
            b = f.documents["B"]
            if change == "body":
                assert (
                    TextChunk.objects.filter(pk=f.chunks["B"].pk).update(
                        content="changed"
                    )
                    == 1
                )
            elif change == "source_hash":
                assert (
                    RawTextDocument.objects.filter(pk=b.pk).update(
                        full_text_hash="f" * 64
                    )
                    == 1
                )
            elif change == "representation":
                assert (
                    DocumentRepresentationState.objects.filter(document_id=b.id).update(
                        legacy_closed=True
                    )
                    == 1
                )
            else:
                assert (
                    CollectionGraphProjection.objects.filter(
                        pk=f.projections["B"].pk
                    ).update(generation_key=uuid4())
                    == 1
                )

        after_hydration(monkeypatch, mutate)
        with connection.execute_wrapper(track):
            rows, diagnostics = hybrid_graph_candidate_pool_v2(
                snapshot, "Atlas", f.authorization, dependencies
            )
        assert observed == [True]
        assert diagnostics["graph_status"] == ("miss" if change == "body" else "error")
        assert f.chunks["B"].pk not in {row.pk for row in rows}
        assert f.chunks["B"].pk not in {row.pk for row in source.cache.values()}
        assert f.chunks["B"].pk not in scope.bodies
