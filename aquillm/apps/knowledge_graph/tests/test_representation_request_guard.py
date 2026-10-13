"""Stored generation fences survive broker failures and every durable retry."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pytest
from django.core.exceptions import ValidationError
from django.db import transaction
from django.db.models import QuerySet

from apps.documents.services.representation_contracts import RepresentationSnapshotV1
from apps.knowledge_graph.models import GraphRebuildRequest
from apps.knowledge_graph.projection.representation_lifecycle import (
    prepare_document_representation_activation,
)
from apps.knowledge_graph.services import builds
from apps.knowledge_graph.services.failure_codes import (
    RebuildPublicationError,
    StaleBuildError,
)
from apps.knowledge_graph.services.representation_request_guard import (
    validate_requested_representation_context,
)
from apps.knowledge_graph.services.requested_representation import (
    request_has_representation_guard,
    requested_representation_generation,
)
from apps.knowledge_graph.tests.ready_subset_support import ScopeFixture

pytestmark = pytest.mark.django_db(transaction=True)


def create_guarded_request(monkeypatch):
    fixture = ScopeFixture(monkeypatch)
    fixture.persist_ready("A")
    ref = fixture.snapshot.selections[0].document
    key = uuid4()
    with monkeypatch.context() as patch:
        patch.setattr(builds, "resume_rebuild_request", lambda _request: None)
        with transaction.atomic():
            prepare_document_representation_activation(
                ref,
                (7,),
                expected_source_hash=ref.source_hash,
                old_representation_key=None,
                new_representation_key=key,
                using="default",
            )
    return fixture, GraphRebuildRequest.objects.get(), key


@pytest.mark.parametrize("failed_first", [False, True])
def test_stale_callback_and_durable_failed_retry_keep_original_generation(
    monkeypatch, failed_first
):
    fixture, request, key = create_guarded_request(monkeypatch)
    original = deepcopy(request.requested_documents)
    assert requested_representation_generation(original[0]) == key
    if failed_first:

        def broker_failure(*_args, **kwargs):
            assert kwargs["representation_request"] == original[0]
            raise RuntimeError("broker unavailable")

        with monkeypatch.context() as patch:
            patch.setattr(builds, "enqueue_document_build", broker_failure)
            with pytest.raises(RebuildPublicationError):
                builds.resume_rebuild_request(request.pk)
        request.refresh_from_db()
        assert request.document_publication_state == "failed"
        assert request.requested_documents == original

    # Negative-only future owner DTO: generation B replaces A at the same source.
    # I4 still rejects successful non-null ownership until I5 supplies its store.
    replacement = replace(fixture.snapshot.selections[0], generation_key=uuid4())
    context = SimpleNamespace(
        representation_snapshot=RepresentationSnapshotV1((replacement,)),
        identity=SimpleNamespace(document_id=replacement.document.id),
    )
    captures, publications = [], []
    monkeypatch.setattr("lib.knowledge_graph.config.get_build_enabled", lambda: True)
    monkeypatch.setattr(
        builds, "_document_context", lambda *args: captures.append(args) or context
    )
    monkeypatch.setattr(
        builds,
        "_publish_graph_task",
        lambda *args, **kwargs: publications.append(kwargs),
    )
    with pytest.raises(RebuildPublicationError):
        builds.resume_rebuild_request(request.pk)
    request.refresh_from_db()
    assert len(captures) == 1 and publications == []
    assert request.document_publication_state == "failed"
    assert request.document_publish_cursor == 0
    assert request.requested_documents == original
    assert not GraphRebuildRequest.objects.filter(predecessor_request=request).exists()


@pytest.mark.parametrize("malformation", ["version", "uuid", "owner", "extra", "null"])
def test_malformed_guard_fails_model_and_resume_before_content(
    monkeypatch, malformation
):
    _fixture, request, _key = create_guarded_request(monkeypatch)
    rows = deepcopy(request.requested_documents)
    guard = rows[0]["representation_binding"]
    if malformation == "version":
        guard["version"] = "unknown"
    elif malformation == "uuid":
        guard["generation_key"] = guard["generation_key"].upper()
    elif malformation == "owner":
        rows[0]["document_pkid"] = True
    elif malformation == "extra":
        rows[0]["unexpected"] = 1
    else:
        rows[0]["representation_binding"] = None
    request.requested_documents = rows
    with pytest.raises(ValidationError):
        request.clean()
    QuerySet.update(
        GraphRebuildRequest.objects.filter(pk=request.pk), requested_documents=rows
    )
    calls = []
    monkeypatch.setattr(builds, "_document_context", lambda *args: calls.append(args))
    with pytest.raises(ValidationError):
        builds.resume_rebuild_request(request.pk)
    assert calls == []


def test_guarded_recovery_never_creates_unbound_successor(monkeypatch):
    _fixture, request, _key = create_guarded_request(monkeypatch)
    original = deepcopy(request.requested_documents)
    builds.record_rebuild_failure(
        request.pk, error_code="stale_source", resnapshot=True
    )
    request.refresh_from_db()
    assert (
        request.status == "partial" and request.error_code == "representation_changed"
    )
    assert request.requested_documents == original
    builds.resume_rebuild_request(request.pk)
    assert GraphRebuildRequest.objects.count() == 1


def test_guard_validation_does_not_short_circuit_after_first_valid_guard(monkeypatch):
    _fixture, request, _key = create_guarded_request(monkeypatch)
    request.requested_documents.append({"representation_binding": None})
    with pytest.raises(ValidationError):
        request_has_representation_guard(request)


@pytest.mark.parametrize(
    "field,value", [("pkid", 999), ("collection_id", 999), ("source_hash", "f" * 64)]
)
def test_captured_owner_and_source_must_match_stored_guard(monkeypatch, field, value):
    fixture, request, key = create_guarded_request(monkeypatch)
    captured = fixture.snapshot.selections[0]
    captured = replace(
        captured,
        document=replace(captured.document, **{field: value}),
        generation_key=key,
    )
    context = SimpleNamespace(
        representation_snapshot=RepresentationSnapshotV1((captured,))
    )
    with pytest.raises(StaleBuildError):
        validate_requested_representation_context(
            request.requested_documents[0], context
        )


def test_guarded_pending_recovery_cannot_capture_replacement(monkeypatch):
    _fixture, request, _key = create_guarded_request(monkeypatch)
    builds.record_rebuild_failure(
        request.pk, error_code="stale_source", resnapshot=True
    )
    QuerySet.update(
        GraphRebuildRequest.objects.filter(pk=request.pk),
        error_code="resnapshot_pending",
    )
    successor, failure, _scope_type, _scope_id = builds._create_rebuild_successor_once(
        request.pk
    )
    request.refresh_from_db()
    assert successor is None and failure is None
    assert request.error_code == "representation_changed"
    assert GraphRebuildRequest.objects.count() == 1
