"""Real selected-scope SQL cost of unpublished native seed preparation."""

from types import SimpleNamespace
from dataclasses import replace
from hashlib import sha256

import pytest
from django.db import connection
from django.test.utils import CaptureQueriesContext

from apps.collections.models import Collection, CollectionPermission
from apps.collections.services.django_retrieval_authorization import build_selected_scope_authorization_context
from apps.collections.services.bounded_retrieval_authorization import bounded_reauthorization_reads
from apps.documents.models import RawTextDocument, TextChunk
from apps.documents.services.chunk_search_source_operation import _closed_header
from apps.documents.services.graph_admission_seeds import native_seed_baseline, native_seed_rows, group_native_seed_view
from apps.documents.tests.test_graph_source_admission import capture, pressure_rows
from apps.knowledge_graph.retrieval.graph_candidate_stage_v2 import _GraphSourceStage
from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import FrontierBudgetV2
from apps.knowledge_graph.tests.test_graph_candidate_operation_work import graph_frame

pytestmark = pytest.mark.django_db(transaction=True)


def large_selected_scope(f):
    pressure_rows(f)
    user = f.documents["A"].ingested_by
    collections = list(Collection.objects.filter(pk__in=(7, 9)))
    collections += Collection.objects.bulk_create([Collection(pk=100 + i, name=f"Selected {i}") for i in range(29)])
    CollectionPermission.objects.bulk_create([
        CollectionPermission(user=user, collection=c, permission="VIEW") for c in collections
    ])
    docs = RawTextDocument.objects.bulk_create([
        RawTextDocument(title=f"Owner {i}", full_text=f"Unrelated original owner {i}.",
                        full_text_hash=sha256(f"Unrelated original owner {i}.".encode()).hexdigest(),
                        collection=collections[i % 31], ingested_by=user)
        for i in range(528)
    ])
    f.documents.update({f"extra-{i}": doc for i, doc in enumerate(docs)})
    f.authorization = build_selected_scope_authorization_context(
        principal=user, selected_collection_ids=tuple(c.pk for c in collections),
        selected_documents=tuple(f.documents.values()),
    )


def test_31_collections_530_documents_do_not_repeat_whole_scope_seed_proofs(monkeypatch):
    with graph_frame(monkeypatch, prepare_sources=large_selected_scope, dataset=True) as active:
        _, graph, source, scope, _, dependencies, _, _ = active
        graph.authorization = scope.authorization
        dependencies.settings.graph_direct_enabled = False
        dependencies.settings.graph_extended_enabled = True
        snapshot = capture(active)
        assert len(scope.requested_document_ids) == 530
        assert len(scope.authorization.selected_collection_ids) == 31
        base = TextChunk.objects.all()
        stage = _GraphSourceStage(scope, base, _closed_header(base, scope))
        stage.initial_keys = set(scope.validate())
        frontier = FrontierBudgetV2(turn_budget=source.budget, deadline=graph.clock() + 5,
            clock=graph.clock, max_identities=200, max_edges=1000,
            max_queries=64, max_source_rows=5000, max_wire_bytes=1048576)
        before = (source.budget.sources_used, source.cache.copy(), scope.bodies.copy())
        owner = stage.seed_validation(snapshot, stage.check) if hasattr(stage, "seed_validation") else None
        try:
            with bounded_reauthorization_reads(scope.authorization, frontier), CaptureQueriesContext(connection) as queries:
                view, _ = (owner.baseline(snapshot, authorization=scope.authorization, settings=dependencies.settings)
                    if owner else native_seed_baseline(snapshot, authorization=scope.authorization, settings=dependencies.settings))
                group = SimpleNamespace(scope=SimpleNamespace(selected_document_ids=scope.requested_document_ids))
                view = owner.group(view, group) if owner else group_native_seed_view(view, group)
                refs = owner.rows(view) if owner else native_seed_rows(view)
            assert refs and all(not hasattr(r, "content") for r in refs)
            assert before == (source.budget.sources_used, source.cache, scope.bodies)
            print(f"seed authority SQL: queries={frontier.queries}, rows={frontier.source_rows}; all SQL={len(queries)}")
            assert frontier.source_rows <= 1122, "six repeated 561-row proofs exhaust the 5000-row frontier"
            assert frontier.queries <= 4
        finally:
            stage.withdraw()


@pytest.mark.parametrize("mutation", ["forged_view", "seed_value", "reference_value", "keys", "containers", "settings", "dependency", "expired", "operation"])
def test_private_seed_views_reject_substitution_and_lifetime_loss(monkeypatch, mutation):
    from apps.documents.services.representation_contracts import SourceScopeUnavailable

    with graph_frame(monkeypatch, prepare_sources=pressure_rows, dataset=True) as active:
        _, _, source, scope, _, dependencies, operation, _ = active
        dependencies.settings.graph_extended_enabled = True
        snapshot = capture(active)
        base = TextChunk.objects.all()
        stage = _GraphSourceStage(scope, base, _closed_header(base, scope))
        stage.initial_keys = set(scope.validate())
        owner = stage.seed_validation(snapshot, stage.check)
        view, _ = owner.baseline(snapshot, authorization=scope.authorization, settings=dependencies.settings)
        before = source.cache.copy()
        if mutation == "forged_view":
            view = replace(view)
        elif mutation == "seed_value":
            object.__setattr__(snapshot.graph_seeds[0], "chunk_id", 999999)
        elif mutation == "reference_value":
            object.__setattr__(view.seed_references[0], "revision", "0" * 32)
        elif mutation == "keys":
            stage.initial_keys.add("replacement-generation")
        elif mutation == "containers":
            scope.bodies = dict(scope.bodies)
        elif mutation == "settings":
            dependencies.settings.graph_extended_max_seeds += 1
        elif mutation == "dependency":
            object.__setattr__(dependencies, "runtime", SimpleNamespace())
        elif mutation == "expired":
            stage.withdraw()
        else:
            operation.live = False
        with pytest.raises((SourceScopeUnavailable, TimeoutError)):
            owner.rows(view)
        assert source.cache == before


def test_private_view_is_not_a_public_seed_or_body_authority(monkeypatch):
    from apps.documents.services.representation_contracts import SourceScopeUnavailable

    with graph_frame(monkeypatch, prepare_sources=pressure_rows, dataset=True) as active:
        f, _, source, scope, _, dependencies, _, _ = active
        dependencies.settings.graph_extended_enabled = True
        snapshot = capture(active)
        base = TextChunk.objects.all()
        stage = _GraphSourceStage(scope, base, _closed_header(base, scope))
        stage.initial_keys = set(scope.validate())
        owner = stage.seed_validation(snapshot, stage.check)
        view, _ = owner.baseline(snapshot, authorization=scope.authorization, settings=dependencies.settings)
        assert native_seed_rows(view) is None
        f.policy.rows = ()
        before = source.cache.copy()
        charged = source.budget.sources_used
        assert stage.read_captured(view.seed_references) == ()
        assert source.budget.sources_used == charged
        with pytest.raises(SourceScopeUnavailable):
            scope.validate()
        assert source.cache == before
        stage.withdraw()


def test_private_seed_rows_recheck_live_chunk_fingerprint(monkeypatch):
    from apps.documents.services.representation_contracts import SourceScopeUnavailable

    with graph_frame(monkeypatch, prepare_sources=pressure_rows, dataset=True) as active:
        _, _, source, scope, _, dependencies, _, _ = active
        dependencies.settings.graph_extended_enabled = True
        snapshot = capture(active)
        base = TextChunk.objects.all()
        stage = _GraphSourceStage(scope, base, _closed_header(base, scope))
        stage.initial_keys = set(scope.validate())
        owner = stage.seed_validation(snapshot, stage.check)
        view, _ = owner.baseline(snapshot, authorization=scope.authorization, settings=dependencies.settings)
        TextChunk.objects.filter(pk=view.seed_references[0].pk).update(content="changed")
        before = source.cache.copy()
        with pytest.raises(SourceScopeUnavailable):
            owner.rows(view)
        assert source.cache == before
        stage.withdraw()


def test_actual_closed_graph_uses_private_views_and_revokes_them_on_exit(monkeypatch):
    from apps.documents.services.hybrid_graph_v2 import hybrid_graph_candidate_pool_v2
    from apps.documents.services.representation_contracts import SourceScopeUnavailable
    from apps.knowledge_graph.retrieval.extended_seed_repository import ExtendedSeedRepository

    with graph_frame(monkeypatch, prepare_sources=pressure_rows, dataset=True) as active:
        f, _, _, _, _, dependencies, _, _ = active
        dependencies.settings.graph_direct_enabled = False
        dependencies.settings.graph_extended_enabled = True
        dependencies.settings.graph_extended_timeout_ms = 5000
        snapshot = capture(active)
        original, owners, batches = _GraphSourceStage.seed_validation, [], []
        batch = ExtendedSeedRepository._load_seed_identities_batch

        def owner(stage, *args, **kwargs):
            result = original(stage, *args, **kwargs)
            owners.append(result)
            return result

        def read(repository, **kwargs):
            # Same PostgreSQL repository, using the fixture's test alias.
            repository.using = "default"
            result = batch(repository, **kwargs)
            batches.append(result)
            return result

        monkeypatch.setattr(_GraphSourceStage, "seed_validation", owner)
        monkeypatch.setattr(ExtendedSeedRepository, "_load_seed_identities_batch", read)
        rows, diagnostics = hybrid_graph_candidate_pool_v2(snapshot, "Atlas", f.authorization, dependencies)
        assert owners and batches and any(batches[0].values())
        assert rows and diagnostics["graph_status"] in {"hit", "miss"}
        assert owners[0].views
        with pytest.raises(SourceScopeUnavailable):
            owners[0].rows(owners[0].views[0][0])


def test_private_seed_batch_rejects_replaced_execution_runtime_before_reads(monkeypatch):
    from copy import copy
    from apps.documents.services.representation_contracts import SourceScopeUnavailable

    with graph_frame(monkeypatch, prepare_sources=pressure_rows, dataset=True) as active:
        _, graph, _, scope, _, dependencies, _, _ = active
        dependencies.settings.graph_extended_enabled = True
        snapshot = capture(active)
        base = TextChunk.objects.all()
        stage = _GraphSourceStage(scope, base, _closed_header(base, scope))
        stage.initial_keys = set(scope.validate())
        owner = stage.seed_validation(snapshot, stage.check)
        view, _ = owner.baseline(snapshot, authorization=scope.authorization, settings=dependencies.settings)
        with CaptureQueriesContext(connection) as queries, pytest.raises(SourceScopeUnavailable):
            owner.prepare_projections(view, (), runtime=copy(graph),
                authorization=scope.authorization, settings=dependencies.settings, frontier=None)
        assert not queries.captured_queries
        stage.withdraw()
