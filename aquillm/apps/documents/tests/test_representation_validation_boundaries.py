"""A fresh persisted-source sweep remains before every dense metadata boundary."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from django.db import connection

from apps.collections.services.django_retrieval_authorization import build_selected_scope_authorization_context
from apps.documents.models.embedding_index import DocumentEmbeddingState
from apps.documents.services import chunk_search_scope as scope_module, embedding_index_query as query
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from lib.embeddings.tests.test_document_provider import transport as transport
from lib.retrieval.turn_budget import TurnBudget, TurnLimits
from .representation_cutover_support import configure_cutover
from .representation_rollback_support import build_dense
from .representation_support import retained as retained, upload as upload, activate

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.fixture
def ready(retained, transport, monkeypatch, settings):
    doc, output, _ = retained
    assert activate(doc, output) == "activated"
    doc.refresh_from_db()
    space = transport[0]
    snapshot, dense = build_dense(doc, space)
    assert snapshot.selections[0].generation_key is not None and len(dense.generation_keys) == 1
    configure_cutover(monkeypatch, settings, space)
    return SimpleNamespace(doc=doc, space=space, keys=dense.generation_keys)


def original_scope(ready, settings, target):
    if target == "no_dense":
        settings.DOCUMENT_EMBEDDING_SPACE_KEY = None
    doc = ready.doc
    authority = build_selected_scope_authorization_context(principal=doc.ingested_by,
        selected_collection_ids=(doc.collection_id,), selected_documents=(doc,))
    assert authority is not None
    runtime = SourceRuntime(TurnBudget(TurnLimits()), authority)
    with source_runtime_scope(runtime):
        scope = scope_module.freeze_search_scope((doc,), using="default")
    assert scope.authorization is runtime.authorization and scope.runtime is runtime
    assert (scope.dense is None) is (target == "no_dense")
    return scope


def invoke(scope, target):
    if target == "freeze":
        return query.freeze_dense_generations(scope.representation,
            space_key=scope.space.space_key, using=scope.using).generation_keys
    if target == "dense":
        return query.validate_dense_snapshot(scope.representation, scope.dense, using=scope.using)
    return scope.validate()


@contextmanager
def observe_boundary(monkeypatch, *, before_source=None):
    """Delegate real validation and SQL; observing never supplies authority."""
    events = []
    original = query.validate_representation_snapshot
    table = connection.ops.quote_name(DocumentEmbeddingState._meta.db_table)

    def validate(snapshot, *, using):
        events.append("source-start")
        if before_source is not None:
            before_source()
        result = original(snapshot, using=using)
        events.append("source-valid")
        return result

    def execute(execute, sql, params, many, context):
        if sql.lstrip().upper().startswith("SELECT") and f"FROM {table}" in sql:
            events.append("dense-read")
        return execute(sql, params, many, context)

    with monkeypatch.context() as probe:
        probe.setattr(query, "validate_representation_snapshot", validate)
        probe.setattr(scope_module, "validate_representation_snapshot", validate)
        with connection.execute_wrapper(execute):
            yield events


@pytest.mark.parametrize("target", ["freeze", "dense", "scope", "no_dense"])
def test_each_real_boundary_rechecks_source_once_before_dense_metadata(ready, settings, monkeypatch, target):
    scope = original_scope(ready, settings, target)
    original = scope.runtime._frozen_retrieval_scope
    budget = scope.runtime.budget
    expected_events = ["source-start", "source-valid"] + ([] if target == "no_dense" else ["dense-read"])
    try:
        with observe_boundary(monkeypatch) as events:
            for _ in range(2):
                events.clear()
                with scope.database():
                    assert invoke(scope, target) == (() if target == "no_dense" else ready.keys)
                assert events == expected_events
        assert scope.runtime._frozen_retrieval_scope is original
        assert scope.representation is original.representation
        assert scope.runtime.budget is budget and budget.can_publish()
    finally:
        budget.close("boundary_test_finished")


@pytest.mark.parametrize("target", ["freeze", "dense", "scope", "no_dense"])
def test_actual_owner_edit_immediately_before_retained_sweep_prevents_dense_read(
    ready, settings, monkeypatch, target,
):
    scope = original_scope(ready, settings, target)
    original = scope.runtime._frozen_retrieval_scope
    old_hash = scope.representation.selections[0].document.source_hash
    mutations = []

    def edit_source():
        assert not mutations
        mutations.append(ready.doc.pk)
        ready.doc.full_text += " A supported ordinary source edit."
        ready.doc.save(dont_rechunk=True)

    try:
        with scope.database(), observe_boundary(monkeypatch, before_source=edit_source) as events:
            # Catch inside the real bounded transaction so the supported edit
            # commits; the failed validation never refreshes the captured scope.
            with pytest.raises(SourceScopeUnavailable, match="source_scope_unavailable"):
                invoke(scope, target)
        assert mutations == [ready.doc.pk]
        assert events == ["source-start"]  # No dense-owner metadata after stale source.
        ready.doc.refresh_from_db()
        assert ready.doc.full_text_hash != old_hash
        assert scope.runtime._frozen_retrieval_scope is original
        assert scope.representation is original.representation
        assert scope.representation.selections[0].document.source_hash == old_hash
        assert scope.runtime.budget.can_publish()  # Rejected by identity, not timeout.
    finally:
        scope.runtime.budget.close("boundary_test_finished")
