from __future__ import annotations

import inspect
import json
import os
import socket
import uuid
from types import SimpleNamespace

import pytest
from django.conf import settings
from django.db import models, transaction


def _database_is_reachable() -> bool:
    database = settings.DATABASES["default"]
    try:
        with socket.create_connection(
            (database["HOST"], int(database.get("PORT") or 5432)), timeout=0.2
        ):
            return True
    except OSError:
        return False


database_required = pytest.mark.skipif(
    not _database_is_reachable() and os.environ.get("KG_REQUIRE_POSTGRES_TESTS") != "1",
    reason="configured PostgreSQL database is not reachable",
)


def _capture_on_commit(monkeypatch, invalidation):
    callbacks = []
    robust_flags = []
    database_aliases = []

    def capture(callback, *, using=None, robust=False):
        callbacks.append(callback)
        robust_flags.append(robust)
        database_aliases.append(using)

    monkeypatch.setattr(invalidation.transaction, "on_commit", capture)
    return callbacks, robust_flags, database_aliases


def _cleanup_result(
    invalidation,
    affected,
    *,
    current,
    source_hash,
    ingestion_complete=False,
    has_active_document_artifact=False,
):
    return invalidation.DocumentGraphCleanupResult(
        affected_collection_ids=affected,
        current_collection_id=current,
        source_hash=source_hash,
        ingestion_complete=ingestion_complete,
        has_active_document_artifact=has_active_document_artifact,
    )


def _persist_successful_document_rebuild_audit():
    from django.utils import timezone

    from apps.documents.models import RawTextDocument
    from apps.knowledge_graph.models import (
        GraphArtifact,
        GraphBuildRun,
        GraphRebuildRequest,
    )
    from apps.knowledge_graph.models.artifacts import _activation_audit_values
    from apps.knowledge_graph.tests.test_models import DOCUMENT_ID

    document = RawTextDocument.objects.get(id=DOCUMENT_ID)
    now = timezone.now()
    request = GraphRebuildRequest.objects.create(
        id=uuid.uuid4(),
        scope_type=GraphRebuildRequest.ScopeType.DOCUMENT,
        scope_id=str(document.id),
        requested_documents=[
            {
                "document_id": str(document.id),
                "document_pkid": document.pkid,
                "model_label": RawTextDocument._meta.label_lower,
                "collection_id": document.collection_id,
                "source_hash": document.full_text_hash,
            }
        ],
        document_count=1,
        status=GraphRebuildRequest.Status.RUNNING,
        started_at=now,
        document_publication_state=GraphRebuildRequest.PublicationState.PUBLISHED,
    )
    artifact = GraphArtifact.objects.create(
        scope_type=GraphArtifact.ScopeType.DOCUMENT,
        scope_id=str(document.id),
        status=GraphArtifact.Status.SUPERSEDED,
        source_hash=document.full_text_hash,
        ontology_version="ontology-v1",
        extractor_version="extractor-v1",
        resolver_version="resolver-v1",
        filter_policy_version="filter-v1",
        build_key="d" * 64,
        build_generation=1,
        orchestration_version=GraphArtifact.OrchestrationVersion.SCOPED_V1,
        rebuild_request=request,
        activated_at=now,
        completed_at=now,
        superseded_at=now,
    )
    run = GraphBuildRun.objects.create(
        artifact=artifact,
        rebuild_request=request,
        stage=GraphBuildRun.Stage.SUPERSEDED,
        status=GraphBuildRun.Status.CANCELLED,
        attempt=1,
        stage_marker={"stage_sequence": ["active", "superseded"]},
        finished_at=now,
    )
    request.status = GraphRebuildRequest.Status.SUCCEEDED
    request.completed_document_count = 1
    request.completed_at = now
    for field, value in _activation_audit_values(artifact, run).items():
        setattr(request, field, value)
    request.save()
    return request, artifact


def test_content_change_invalidation_is_deferred_and_refreshes_collection(monkeypatch):
    from apps.knowledge_graph.graph import invalidation
    from apps.knowledge_graph.services import builds

    callbacks, robust_flags, database_aliases = _capture_on_commit(
        monkeypatch, invalidation
    )
    actions = []
    monkeypatch.setattr(
        invalidation,
        "cleanup_document_graph_state",
        lambda document_ref, collection_ids, *, reason, using, expected_source_hash: (
            actions.append(
                (
                    "cleanup",
                    document_ref,
                    collection_ids,
                    reason,
                    using,
                    expected_source_hash,
                )
            )
            or _cleanup_result(
                invalidation,
                (17,),
                current=17,
                source_hash="b" * 64,
            )
        ),
    )
    monkeypatch.setattr(
        invalidation,
        "load_current_document_state",
        lambda _document_ref, *, using: ("b" * 64, 17),
    )
    monkeypatch.setattr(
        builds,
        "enqueue_current_collection_refresh",
        lambda collection_id: actions.append(("refresh", collection_id)),
    )

    document_id = uuid.uuid4()
    event = invalidation.DocumentLifecycleEvent(
        document=invalidation.DocumentLifecycleRef(
            concrete_model_label="apps_documents.rawtextdocument",
            document_pkid=41,
            document_id=document_id,
        ),
        old_source_hash="a" * 64,
        committed_source_hash="b" * 64,
        old_collection_id=17,
        committed_collection_id=17,
    )
    invalidation.schedule_document_content_invalidation(
        event,
        using="default",
        after_cleanup=lambda: actions.append(("publish_chunks",)),
    )

    assert actions == []
    assert robust_flags == [True]
    assert database_aliases == ["default"]
    callbacks[0]()
    assert actions == [
        (
            "cleanup",
            event.document,
            (17,),
            "document_content_changed",
            "default",
            "b" * 64,
        ),
        ("publish_chunks",),
    ]


def test_move_reuses_document_graph_and_refreshes_old_and_new_collections(monkeypatch):
    from apps.knowledge_graph.graph import invalidation
    from apps.knowledge_graph.services import builds

    callbacks, robust_flags, database_aliases = _capture_on_commit(
        monkeypatch, invalidation
    )
    actions = []
    monkeypatch.setattr(
        invalidation,
        "cleanup_document_collection_graph_state",
        lambda document_ref, collection_ids, *, reason, using, expected_source_hash: (
            actions.append(
                (
                    "collections",
                    document_ref,
                    collection_ids,
                    reason,
                    using,
                    expected_source_hash,
                )
            )
            or _cleanup_result(
                invalidation,
                (19, 23),
                current=19,
                source_hash="a" * 64,
                ingestion_complete=True,
                has_active_document_artifact=True,
            )
        ),
    )
    monkeypatch.setattr(
        builds,
        "enqueue_current_collection_refresh",
        lambda collection_id: actions.append(("refresh", collection_id)),
    )
    monkeypatch.setattr(
        invalidation,
        "load_current_document_state",
        lambda _document_ref, *, using: ("a" * 64, 19),
    )

    document_id = uuid.uuid4()
    event = invalidation.DocumentLifecycleEvent(
        document=invalidation.DocumentLifecycleRef(
            concrete_model_label="apps_documents.rawtextdocument",
            document_pkid=43,
            document_id=document_id,
        ),
        old_source_hash="a" * 64,
        committed_source_hash="a" * 64,
        old_collection_id=23,
        committed_collection_id=19,
    )
    invalidation.schedule_document_move_invalidation(event, using="default")

    assert actions == []
    assert robust_flags == [True]
    assert database_aliases == ["default"]
    callbacks[0]()
    assert actions == [
        (
            "collections",
            event.document,
            (19, 23),
            "document_moved",
            "default",
            "a" * 64,
        ),
        ("refresh", 19),
        ("refresh", 23),
    ]


def test_delete_signal_defers_chunk_and_graph_cleanup(monkeypatch):
    from apps.knowledge_graph.graph import invalidation
    from apps.knowledge_graph.services import builds

    callbacks, robust_flags, database_aliases = _capture_on_commit(
        monkeypatch, invalidation
    )
    actions = []
    document_id = uuid.uuid4()
    sender = type("ConcreteDocument", (), {})
    sender._meta = SimpleNamespace(label_lower="apps_documents.rawtextdocument")
    instance = SimpleNamespace(
        id=document_id,
        pkid=47,
        collection_id=31,
        full_text_hash="a" * 64,
    )
    context = SimpleNamespace(snapshot=SimpleNamespace(locked_collection_ids=(31,)))
    monkeypatch.setattr(
        invalidation,
        "_ensure_origin_delete_scope_locked",
        lambda origin, *, using: context,
    )
    monkeypatch.setattr(
        invalidation,
        "_assert_document_delete_scope",
        lambda *_args: None,
    )

    monkeypatch.setattr(
        invalidation,
        "cleanup_document_graph_state",
        lambda document_ref, collection_ids, *, reason, using, **guards: (
            actions.append(
                ("cleanup", document_ref, collection_ids, reason, using, guards)
            )
            or _cleanup_result(
                invalidation,
                (31,),
                current=31,
                source_hash="a" * 64,
            )
        ),
    )
    monkeypatch.setattr(
        builds,
        "enqueue_current_collection_refresh",
        lambda collection_id: actions.append(("refresh", collection_id)),
    )

    invalidation.document_pre_delete(
        sender=sender,
        instance=instance,
        using="default",
        origin=instance,
    )
    document_ref = invalidation.DocumentLifecycleRef(
        concrete_model_label="apps_documents.rawtextdocument",
        document_pkid=47,
        document_id=document_id,
    )
    assert actions == [
        (
            "cleanup",
            document_ref,
            (31,),
            "document_deleted",
            "default",
            {
                "expected_source_hash": "a" * 64,
                "expected_collection_id": 31,
                "fail_on_stale": True,
                "_origin_context": context,
            },
        )
    ]
    invalidation.document_post_delete(sender=sender, instance=instance, using="default")

    assert actions == [
        (
            "cleanup",
            document_ref,
            (31,),
            "document_deleted",
            "default",
            {
                "expected_source_hash": "a" * 64,
                "expected_collection_id": 31,
                "fail_on_stale": True,
                "_origin_context": context,
            },
        )
    ]
    assert robust_flags == [True]
    assert database_aliases == ["default"]
    callbacks[0]()
    assert actions == [
        (
            "cleanup",
            document_ref,
            (31,),
            "document_deleted",
            "default",
            {
                "expected_source_hash": "a" * 64,
                "expected_collection_id": 31,
                "fail_on_stale": True,
                "_origin_context": context,
            },
        ),
        ("refresh", 31),
    ]


def test_stale_content_callback_cannot_cleanup_a_newer_document_version(monkeypatch):
    from apps.knowledge_graph.graph import invalidation
    from apps.knowledge_graph.services import builds

    callbacks, _robust_flags, _database_aliases = _capture_on_commit(
        monkeypatch, invalidation
    )
    actions = []
    event = invalidation.DocumentLifecycleEvent(
        document=invalidation.DocumentLifecycleRef(
            concrete_model_label="apps_documents.rawtextdocument",
            document_pkid=53,
            document_id=uuid.uuid4(),
        ),
        old_source_hash="a" * 64,
        committed_source_hash="b" * 64,
        old_collection_id=17,
        committed_collection_id=17,
    )
    monkeypatch.setattr(
        invalidation,
        "load_current_document_state",
        lambda _document_ref, *, using: ("c" * 64, 17),
    )
    monkeypatch.setattr(
        invalidation,
        "cleanup_document_graph_state",
        lambda *_args, **kwargs: (
            actions.append(("cleanup", kwargs["expected_source_hash"])) or ()
        ),
    )
    monkeypatch.setattr(
        builds,
        "enqueue_current_collection_refresh",
        lambda *_args, **_kwargs: actions.append("refresh"),
    )

    invalidation.schedule_document_content_invalidation(event, using="default")
    callbacks[0]()

    assert actions == [("cleanup", "b" * 64)]


def test_content_then_move_callbacks_still_remove_obsolete_document_graph(monkeypatch):
    from apps.knowledge_graph.graph import invalidation
    from apps.knowledge_graph.services import builds

    callbacks, _robust_flags, _database_aliases = _capture_on_commit(
        monkeypatch, invalidation
    )
    actions = []
    document = invalidation.DocumentLifecycleRef(
        concrete_model_label="apps_documents.rawtextdocument",
        document_pkid=59,
        document_id=uuid.uuid4(),
    )
    content_event = invalidation.DocumentLifecycleEvent(
        document=document,
        old_source_hash="a" * 64,
        committed_source_hash="b" * 64,
        old_collection_id=17,
        committed_collection_id=17,
    )
    move_event = invalidation.DocumentLifecycleEvent(
        document=document,
        old_source_hash="b" * 64,
        committed_source_hash="b" * 64,
        old_collection_id=17,
        committed_collection_id=19,
    )
    monkeypatch.setattr(
        invalidation,
        "load_current_document_lifecycle_state",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("routing must use the locked cleanup snapshot")
        ),
    )
    monkeypatch.setattr(
        invalidation,
        "cleanup_document_graph_state",
        lambda _document_ref, collection_ids, *, reason, using, expected_source_hash: (
            actions.append(
                ("document", collection_ids, reason, using, expected_source_hash)
            )
            or _cleanup_result(
                invalidation,
                (17, 19, 23),
                current=19,
                source_hash="b" * 64,
            )
        ),
    )
    monkeypatch.setattr(
        invalidation,
        "cleanup_document_collection_graph_state",
        lambda _document_ref, collection_ids, *, reason, using, expected_source_hash: (
            actions.append(
                ("collections", collection_ids, reason, using, expected_source_hash)
            )
            or _cleanup_result(
                invalidation,
                (17, 19),
                current=19,
                source_hash="b" * 64,
            )
        ),
    )
    monkeypatch.setattr(
        builds,
        "enqueue_current_collection_refresh",
        lambda collection_id: actions.append(("refresh", collection_id)),
    )

    invalidation.schedule_document_content_invalidation(content_event)
    invalidation.schedule_document_move_invalidation(move_event)
    for callback in callbacks:
        callback()

    assert actions == [
        ("document", (17,), "document_content_changed", "default", "b" * 64),
        ("refresh", 17),
        ("refresh", 23),
        ("collections", (17, 19), "document_moved", "default", "b" * 64),
        ("refresh", 17),
    ]


def test_move_then_content_refreshes_dependent_original_collection(monkeypatch):
    from apps.knowledge_graph.graph import invalidation
    from apps.knowledge_graph.services import builds

    callbacks, _robust_flags, _database_aliases = _capture_on_commit(
        monkeypatch, invalidation
    )
    actions = []
    document = invalidation.DocumentLifecycleRef(
        concrete_model_label="apps_documents.rawtextdocument",
        document_pkid=61,
        document_id=uuid.uuid4(),
    )
    move_event = invalidation.DocumentLifecycleEvent(
        document=document,
        old_source_hash="a" * 64,
        committed_source_hash="a" * 64,
        old_collection_id=17,
        committed_collection_id=19,
    )
    content_event = invalidation.DocumentLifecycleEvent(
        document=document,
        old_source_hash="a" * 64,
        committed_source_hash="b" * 64,
        old_collection_id=19,
        committed_collection_id=19,
    )
    monkeypatch.setattr(
        invalidation,
        "load_current_document_state",
        lambda _document_ref, *, using: ("b" * 64, 19),
    )
    monkeypatch.setattr(
        invalidation,
        "cleanup_document_graph_state",
        lambda _document_ref, collection_ids, *, reason, using, expected_source_hash: (
            actions.append(
                ("document", collection_ids, reason, using, expected_source_hash)
            )
            or _cleanup_result(
                invalidation,
                (17, 19, 23),
                current=19,
                source_hash="b" * 64,
            )
        ),
    )
    monkeypatch.setattr(
        invalidation,
        "cleanup_document_collection_graph_state",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        builds,
        "enqueue_current_collection_refresh",
        lambda collection_id: actions.append(("refresh", collection_id)),
    )

    invalidation.schedule_document_move_invalidation(move_event)
    invalidation.schedule_document_content_invalidation(content_event)
    for callback in callbacks:
        callback()

    assert actions == [
        ("document", (19,), "document_content_changed", "default", "b" * 64),
        ("refresh", 17),
        ("refresh", 23),
    ]


def test_lifecycle_signals_cover_every_concrete_document_model(monkeypatch):
    from apps.knowledge_graph.graph import invalidation

    connections = []
    for signal in (
        invalidation.pre_save,
        invalidation.post_save,
        invalidation.pre_delete,
        invalidation.post_delete,
    ):
        monkeypatch.setattr(
            signal,
            "connect",
            lambda receiver, _signal=signal, **kwargs: connections.append(
                (_signal, receiver, kwargs)
            ),
        )
    models = tuple(
        type(
            f"DocumentType{index}",
            (),
            {"_meta": SimpleNamespace(label_lower=f"documents.documenttype{index}")},
        )
        for index in range(8)
    )

    invalidation.register_document_lifecycle_signals(models)

    assert len(connections) == 32
    assert {row[2]["sender"] for row in connections} == set(models)
    assert all(row[2]["weak"] is False for row in connections)
    assert len({row[2]["dispatch_uid"] for row in connections}) == 32


def test_raw_fixture_save_skips_lifecycle_queries_and_events():
    from apps.knowledge_graph.graph import invalidation

    class ExplodingManager:
        def using(self, _alias):
            raise AssertionError("raw fixture saves must not query lifecycle state")

    sender = SimpleNamespace(
        _base_manager=ExplodingManager(),
        _meta=SimpleNamespace(label_lower="apps_documents.rawtextdocument"),
    )
    instance = SimpleNamespace(
        pkid=7,
        id=uuid.uuid4(),
        full_text_hash="a" * 64,
        collection_id=17,
    )

    invalidation.document_pre_save(sender, instance, raw=True)
    invalidation.document_post_save(sender, instance, raw=True)

    assert invalidation.consume_document_save_lifecycle(instance) is None


def test_post_save_uses_reloaded_committed_fields_not_stale_instance_attributes():
    from apps.knowledge_graph.graph import invalidation

    document_id = uuid.uuid4()

    class RowManager:
        def __init__(self):
            self.rows = [
                {
                    "id": document_id,
                    "full_text_hash": "a" * 64,
                    "collection_id": 17,
                },
                {
                    "id": document_id,
                    "full_text_hash": "b" * 64,
                    "collection_id": 17,
                },
            ]

        def using(self, _alias):
            return self

        def filter(self, **_kwargs):
            return self

        def values(self, *_fields):
            return self

        def first(self):
            return self.rows.pop(0)

    sender = SimpleNamespace(
        _base_manager=RowManager(),
        _meta=SimpleNamespace(label_lower="apps_documents.rawtextdocument"),
    )
    instance = SimpleNamespace(
        pkid=11,
        id=document_id,
        full_text_hash="c" * 64,
        collection_id=99,
    )

    invalidation.document_pre_save(sender, instance)
    invalidation.document_post_save(sender, instance, created=False)

    kind, event, alias = invalidation.consume_document_save_lifecycle(instance)
    assert (kind, alias) == ("content", "default")
    assert event.old_source_hash == "a" * 64
    assert event.committed_source_hash == "b" * 64
    assert event.old_collection_id == event.committed_collection_id == 17


def test_document_figure_parent_descriptor_uses_the_logical_uuid_not_integer_pk():
    from apps.documents.models import DocumentFigure

    assert isinstance(DocumentFigure.__dict__["parent_document"], property)


def test_document_move_requires_the_request_actor_explicitly():
    from apps.documents.models import Document

    actor = inspect.signature(Document.move_to).parameters["actor"]
    assert actor.kind is inspect.Parameter.KEYWORD_ONLY
    assert actor.default is inspect.Parameter.empty


def test_collection_graph_references_do_not_drive_source_collection_deletion():
    from apps.knowledge_graph.models import (
        CollectionArtifactInput,
        CollectionEntity,
        GraphArtifact,
    )

    assert (
        CollectionArtifactInput._meta.get_field("collection").remote_field.on_delete
        is models.DO_NOTHING
    )
    assert (
        CollectionEntity._meta.get_field("collection").remote_field.on_delete
        is models.DO_NOTHING
    )
    assert (
        GraphArtifact._meta.get_field("collection_scope").remote_field.on_delete
        is models.DO_NOTHING
    )


def test_graph_artifact_collection_scope_auto_binds_and_document_scope_stays_null():
    from apps.knowledge_graph.models import GraphArtifact

    collection_artifact = GraphArtifact(
        scope_type=GraphArtifact.ScopeType.COLLECTION,
        scope_id="17",
        source_hash="a" * 64,
        ontology_version="ontology-v1",
        extractor_version="extractor-v1",
        resolver_version="resolver-v1",
        filter_policy_version="filter-v1",
        embedding_model_signature=(
            f"test:model@rev:endpoint={'e' * 64}:dims=1024:"
            "prep=kg-entity-v1:max_chars=8192:batch=64"
        ),
    )
    collection_artifact.prepare_for_persistence()
    assert collection_artifact.collection_scope_id == 17

    document_artifact = GraphArtifact(
        scope_type=GraphArtifact.ScopeType.DOCUMENT,
        scope_id=uuid.uuid4(),
        source_hash="a" * 64,
        ontology_version="ontology-v1",
        extractor_version="extractor-v1",
        resolver_version="resolver-v1",
        filter_policy_version="filter-v1",
    )
    document_artifact.prepare_for_persistence()
    assert document_artifact.collection_scope_id is None


def test_graph_artifact_rejects_mismatched_supplied_collection_scope():
    from django.core.exceptions import ValidationError

    from apps.knowledge_graph.models import GraphArtifact

    artifact = GraphArtifact(
        scope_type=GraphArtifact.ScopeType.COLLECTION,
        scope_id="17",
        collection_scope_id=18,
        source_hash="a" * 64,
        ontology_version="ontology-v1",
        extractor_version="extractor-v1",
        resolver_version="resolver-v1",
        filter_policy_version="filter-v1",
        embedding_model_signature=(
            f"test:model@rev:endpoint={'e' * 64}:dims=1024:"
            "prep=kg-entity-v1:max_chars=8192:batch=64"
        ),
    )

    with pytest.raises(ValidationError, match="collection|scope"):
        artifact.prepare_for_persistence()


def test_cached_document_rehydration_requires_current_collection_allowlist(monkeypatch):
    from apps.documents.services import rag_cache

    calls = []
    document = SimpleNamespace(pkid=7, collection_id=19)

    class FakeQuerySet:
        def __init__(self, rows):
            self.rows = rows

        def __iter__(self):
            return iter(self.rows)

    class FakeManager:
        def filter(self, **kwargs):
            calls.append(kwargs)
            allowed = tuple(kwargs.get("collection_id__in", (document.collection_id,)))
            return FakeQuerySet(
                (document,) if document.collection_id in allowed else ()
            )

    model = SimpleNamespace(objects=FakeManager())
    monkeypatch.setattr(
        "django.apps.apps.get_model", lambda _app_label, _model_name: model
    )

    result = rag_cache.rehydrate_documents_from_refs(
        [{"model": "RawTextDocument", "pkid": 7}],
        allowed_collection_ids=(17,),
    )

    assert result == []
    assert calls == [{"pkid__in": [7], "collection_id__in": (17,)}]


@pytest.mark.django_db(transaction=True)
@database_required
def test_document_save_rollback_never_publishes_chunking(monkeypatch):
    from django.contrib.auth.models import User

    from apps.collections.models import Collection
    from apps.documents.models import RawTextDocument
    from apps.documents.tasks.chunking import create_chunks

    published = []
    monkeypatch.setattr(create_chunks, "delay", published.append)
    user = User.objects.create_user(
        username=f"kg-rollback-{uuid.uuid4()}", password="unused"
    )
    collection = Collection.objects.create(name=f"rollback {uuid.uuid4()}")

    with pytest.raises(RuntimeError, match="rollback"):
        with transaction.atomic():
            RawTextDocument.objects.create(
                title="rolled back",
                full_text="content never committed",
                collection=collection,
                ingested_by=user,
            )
            raise RuntimeError("rollback")

    assert published == []


@pytest.mark.django_db(transaction=True)
@database_required
def test_move_api_checks_the_request_actor_on_source_and_destination(client):
    from django.contrib.auth.models import User
    from django.urls import reverse

    from apps.collections.models import Collection, CollectionPermission
    from apps.documents.models import RawTextDocument

    owner = User.objects.create_user(
        username=f"kg-move-owner-{uuid.uuid4()}", password="unused"
    )
    actor = User.objects.create_user(
        username=f"kg-move-actor-{uuid.uuid4()}", password="unused"
    )
    source = Collection.objects.create(name=f"source {uuid.uuid4()}")
    destination = Collection.objects.create(name=f"private {uuid.uuid4()}")
    CollectionPermission.objects.create(
        user=actor, collection=source, permission="EDIT"
    )
    CollectionPermission.objects.create(
        user=owner, collection=destination, permission="MANAGE"
    )
    document = RawTextDocument(
        title="private target",
        full_text="cannot be moved by actor",
        collection=source,
        ingested_by=owner,
    )
    document.save(dont_rechunk=True)
    client.force_login(actor)

    response = client.post(
        reverse("api_move_document", kwargs={"doc_id": document.id}),
        data=json.dumps({"new_collection_id": destination.pk}),
        content_type="application/json",
    )

    document.refresh_from_db()
    assert response.status_code == 403
    assert document.collection_id == source.pk


@pytest.mark.django_db(transaction=True)
@database_required
def test_cached_access_refs_fail_closed_after_document_moves_private(settings):
    from django.contrib.auth.models import User
    from django.core.cache import cache

    from apps.collections.models import Collection, CollectionPermission
    from apps.documents.models import RawTextDocument

    settings.RAG_CACHE_ENABLED = True
    cache.clear()
    owner = User.objects.create_user(
        username=f"kg-cache-owner-{uuid.uuid4()}", password="unused"
    )
    viewer = User.objects.create_user(
        username=f"kg-cache-viewer-{uuid.uuid4()}", password="unused"
    )
    shared = Collection.objects.create(name=f"shared {uuid.uuid4()}")
    private = Collection.objects.create(name=f"private {uuid.uuid4()}")
    CollectionPermission.objects.create(
        user=viewer, collection=shared, permission="VIEW"
    )
    CollectionPermission.objects.create(
        user=owner, collection=private, permission="MANAGE"
    )
    document = RawTextDocument(
        title="cached document",
        full_text="cached shared content",
        collection=shared,
        ingested_by=owner,
    )
    document.save(dont_rechunk=True)
    selected = Collection.objects.filter(pk=shared.pk)

    assert Collection.get_user_accessible_documents(viewer, selected) == [document]
    document.collection = private
    document.save(dont_rechunk=True, update_fields=["collection"])

    assert Collection.get_user_accessible_documents(viewer, selected) == []


@pytest.mark.django_db(transaction=True)
@database_required
def test_document_figure_parent_round_trips_exact_typed_uuid_and_pkid():
    from django.contrib.auth.models import User

    from apps.collections.models import Collection
    from apps.documents.models import DocumentFigure, RawTextDocument

    user = User.objects.create_user(
        username=f"kg-figure-{uuid.uuid4()}", password="unused"
    )
    collection = Collection.objects.create(name=f"figure parent {uuid.uuid4()}")
    parent = RawTextDocument(
        title="source",
        full_text="source content",
        collection=collection,
        ingested_by=user,
    )
    parent.save(dont_rechunk=True)
    figure = DocumentFigure(
        title="figure",
        full_text="figure caption",
        collection=collection,
        ingested_by=user,
        source_format="pdf",
    )
    figure.parent_document = parent
    assert figure.parent_object_id == parent.id
    figure.save(dont_rechunk=True)
    figure.refresh_from_db()
    assert figure.parent_document == parent


@pytest.mark.django_db(transaction=True)
@database_required
def test_instance_delete_cleans_active_cross_artifact_evidence_and_keeps_audit():
    from apps.documents.models import RawTextDocument, TextChunk
    from apps.knowledge_graph.models import (
        CollectionRelationEvidence,
        GraphArtifact,
        GraphBuildRun,
    )
    from apps.knowledge_graph.tests.test_models import (
        DOCUMENT_ID,
        _persist_collection_relation_fixture,
    )

    fixture = _persist_collection_relation_fixture()
    request, request_artifact = _persist_successful_document_rebuild_audit()
    fixture.collection_artifact.status = GraphArtifact.Status.ACTIVE
    fixture.collection_artifact.save(update_fields=["status"])
    audit = GraphBuildRun.objects.create(
        artifact=fixture.document_artifact,
        stage=GraphBuildRun.Stage.COMPLETE,
        status=GraphBuildRun.Status.SUCCEEDED,
        attempt=1,
        stats={"relations": 1},
    )

    RawTextDocument.objects.get(id=DOCUMENT_ID).delete()

    assert not GraphArtifact.objects.filter(pk=fixture.document_artifact.pk).exists()
    assert not GraphArtifact.objects.filter(pk=fixture.collection_artifact.pk).exists()
    assert not TextChunk.objects.filter(doc_id=DOCUMENT_ID).exists()
    assert not CollectionRelationEvidence.objects.filter(
        pk=fixture.evidence.pk
    ).exists()
    assert not GraphArtifact.objects.filter(pk=request_artifact.pk).exists()
    request.refresh_from_db()
    assert request.status == request.Status.SUCCEEDED
    assert request.activated_artifact_pk == request_artifact.pk
    request._validate_success_activation()
    audit.refresh_from_db()
    assert audit.artifact_id is None
    assert audit.stage == GraphBuildRun.Stage.STALE
    assert audit.status == GraphBuildRun.Status.CANCELLED
    assert audit.stats == {"relations": 1}


@pytest.mark.django_db(transaction=True)
@database_required
def test_queryset_delete_runs_the_same_graph_cleanup_path():
    from apps.documents.models import RawTextDocument, TextChunk
    from apps.knowledge_graph.models import CollectionRelationEvidence, GraphArtifact
    from apps.knowledge_graph.tests.test_models import (
        DOCUMENT_ID,
        _persist_collection_relation_fixture,
    )

    fixture = _persist_collection_relation_fixture()
    request, request_artifact = _persist_successful_document_rebuild_audit()
    fixture.collection_artifact.status = GraphArtifact.Status.ACTIVE
    fixture.collection_artifact.save(update_fields=["status"])

    RawTextDocument.objects.filter(id=DOCUMENT_ID).delete()

    assert not GraphArtifact.objects.filter(pk=fixture.document_artifact.pk).exists()
    assert not GraphArtifact.objects.filter(pk=fixture.collection_artifact.pk).exists()
    assert not TextChunk.objects.filter(doc_id=DOCUMENT_ID).exists()
    assert not CollectionRelationEvidence.objects.filter(
        pk=fixture.evidence.pk
    ).exists()
    assert not GraphArtifact.objects.filter(pk=request_artifact.pk).exists()
    request.refresh_from_db()
    assert request.activated_artifact_pk == request_artifact.pk
    request._validate_success_activation()


@pytest.mark.django_db(transaction=True)
@database_required
def test_collection_cascade_is_not_blocked_by_graph_manifest_or_evidence():
    from apps.collections.models import Collection
    from apps.knowledge_graph.models import (
        CollectionArtifactInput,
        CollectionRelationEvidence,
        GraphArtifact,
    )
    from apps.knowledge_graph.tests.test_models import (
        COLLECTION_ID,
        _persist_collection_relation_fixture,
    )

    fixture = _persist_collection_relation_fixture()
    request, request_artifact = _persist_successful_document_rebuild_audit()
    fixture.collection_artifact.status = GraphArtifact.Status.ACTIVE
    fixture.collection_artifact.save(update_fields=["status"])

    Collection.objects.get(pk=COLLECTION_ID).delete()

    assert not GraphArtifact.objects.filter(pk=fixture.collection_artifact.pk).exists()
    assert not CollectionArtifactInput.objects.filter(
        artifact=fixture.collection_artifact
    ).exists()
    assert not CollectionRelationEvidence.objects.filter(
        artifact=fixture.collection_artifact
    ).exists()
    assert not GraphArtifact.objects.filter(pk=request_artifact.pk).exists()
    request.refresh_from_db()
    assert request.activated_artifact_pk == request_artifact.pk
    request._validate_success_activation()


@pytest.mark.django_db(transaction=True)
@database_required
def test_content_invalidation_keeps_successful_rebuild_scalar_audit(monkeypatch):
    from apps.documents.models import RawTextDocument
    from apps.knowledge_graph.models import GraphArtifact
    from apps.knowledge_graph.services import builds
    from apps.knowledge_graph.tests.test_models import (
        DOCUMENT_ID,
        _persist_collection_relation_fixture,
    )

    _persist_collection_relation_fixture()
    request, request_artifact = _persist_successful_document_rebuild_audit()
    monkeypatch.setattr(
        builds,
        "enqueue_current_collection_refresh",
        lambda *_args, **_kwargs: None,
    )
    document = RawTextDocument.objects.get(id=DOCUMENT_ID)
    replacement = "Aquilla evaluates a changed benchmark."
    document.full_text = replacement
    document.full_text_hash = RawTextDocument.hash_fn(replacement)
    document.save(
        dont_rechunk=True,
        update_fields=["full_text", "full_text_hash"],
    )

    assert not GraphArtifact.objects.filter(pk=request_artifact.pk).exists()
    request.refresh_from_db()
    assert request.status == request.Status.SUCCEEDED
    assert request.activated_artifact_pk == request_artifact.pk
    request._validate_success_activation()


@pytest.mark.django_db(transaction=True)
@database_required
def test_parent_delete_cascades_document_figures_by_logical_uuid():
    from django.contrib.auth.models import User

    from apps.collections.models import Collection
    from apps.documents.models import DocumentFigure, RawTextDocument

    user = User.objects.create_user(
        username=f"kg-figure-cascade-{uuid.uuid4()}", password="unused"
    )
    collection = Collection.objects.create(name=f"figure cascade {uuid.uuid4()}")
    parent = RawTextDocument(
        title="source",
        full_text="source content",
        collection=collection,
        ingested_by=user,
    )
    parent.save(dont_rechunk=True)
    figure = DocumentFigure(
        title="figure",
        full_text="figure caption",
        collection=collection,
        ingested_by=user,
        source_format="pdf",
    )
    figure.parent_document = parent
    figure.save(dont_rechunk=True)

    parent.delete()

    assert not DocumentFigure.objects.filter(pk=figure.pk).exists()
