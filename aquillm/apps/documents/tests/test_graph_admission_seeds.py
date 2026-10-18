"""Deferred native V2 seeds reach real mapping without spending body slots."""

from copy import copy
from time import monotonic

import pytest

from apps.documents.tests.test_graph_source_admission import capture, pressure_rows
from apps.knowledge_graph.tests.test_graph_candidate_operation_work import graph_frame

pytestmark = pytest.mark.django_db(transaction=True)


def test_extended_only_deferred_seed_reaches_actual_mapping_without_body_admission(monkeypatch):
    from apps.knowledge_graph.retrieval.graph_seed_allocation_v2 import validated_seed_baseline, group_baseline_view
    from apps.knowledge_graph.retrieval.extended_seed_repository import ExtendedSeedRepository
    from apps.knowledge_graph.retrieval.production_extended import prepare_extended_branch
    from apps.knowledge_graph.retrieval.production_runtime_support import ProductionSharedScopeV1
    from apps.knowledge_graph.retrieval.ready_scope_repository_v2 import select_ready_scope_v2

    with graph_frame(monkeypatch, prepare_sources=pressure_rows, dataset=True) as active:
        f, graph, source, scope, _, dependencies, _, _ = active
        dependencies.settings.graph_direct_enabled = False
        dependencies.settings.graph_extended_enabled = True
        snapshot = capture(active)
        initial = {r.pk for r in snapshot.baseline_candidates}
        deferred = {s.chunk_id for s in snapshot.graph_seeds} - initial
        assert deferred
        before = (source.budget.sources_used, source.cache.copy(), scope.bodies.copy(), scope.dense_rows.copy(), scope.fusion_v2)
        view, reasons = validated_seed_baseline(snapshot, authorization=scope.authorization, settings=dependencies.settings)
        assert view is not None and "not_ready" not in reasons
        selected = select_ready_scope_v2(
            authorization=scope.authorization, settings=dependencies.settings,
            representation_snapshot=scope.representation,
            requested_document_ids=scope.requested_document_ids,
        )
        actual = ExtendedSeedRepository(using="default")
        load, mapped = actual.load_seed_identities, []

        def observe(**kwargs):
            mapped.extend(pk for pk, _ in kwargs["chunks"])
            return load(**kwargs)

        monkeypatch.setattr(actual, "load_seed_identities", observe)
        graph.projection_repository_factory = lambda: actual
        for group in selected.groups:
            local = copy(graph)
            local._shared = ProductionSharedScopeV1(group.scope)
            prepare_extended_branch(
                local, baseline=group_baseline_view(view, group), shared=local._shared,
                authorization=scope.authorization, settings=dependencies.settings,
                deadline=monotonic() + 5,
            )
        assert set(mapped) & deferred
        assert source.budget.sources_used == before[0] == 25
        assert (source.cache, scope.bodies, scope.dense_rows) == before[1:4]
        assert scope.fusion_v2 is before[4]


@pytest.mark.parametrize("mutation", ["source", "permission"])
def test_seed_view_consumption_rechecks_exact_owner_and_permission(monkeypatch, mutation):
    from apps.documents.services.graph_admission_seeds import native_seed_rows
    from apps.documents.services.representation_contracts import SourceScopeUnavailable
    from apps.documents.tests.test_representation_schema import capture as capture_legacy_snapshot
    from apps.knowledge_graph.retrieval.graph_seed_allocation_v2 import validated_seed_baseline

    def prepare(f):
        pressure_rows(f)
        capture_legacy_snapshot(f.documents["A"])

    with graph_frame(monkeypatch, prepare_sources=prepare, dataset=True) as active:
        f, _, source, scope, _, dependencies, _, _ = active
        dependencies.settings.graph_direct_enabled = False
        dependencies.settings.graph_extended_enabled = True
        snapshot = capture(active)
        view, _ = validated_seed_baseline(snapshot, authorization=scope.authorization, settings=dependencies.settings)
        assert view is not None and native_seed_rows(view)
        before = (source.budget.sources_used, source.cache.copy(), scope.bodies.copy(), scope.dense_rows.copy(), scope.fusion_v2)
        if mutation == "permission":
            f.policy.rows = ()
        else:
            doc = f.documents["A"]
            type(doc).objects.filter(pk=doc.pk).update(full_text=doc.full_text.replace("Atlas", "Cedar"))
        with pytest.raises(SourceScopeUnavailable):
            native_seed_rows(view)
        assert source.budget.sources_used == before[0]
        assert (source.cache, scope.bodies, scope.dense_rows) == before[1:4]
        assert scope.fusion_v2 is before[4]
