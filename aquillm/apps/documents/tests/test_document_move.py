"""Permission and lifecycle-preserving document move behavior."""
from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from django.core.exceptions import PermissionDenied, ValidationError
from django.test import RequestFactory

from apps.documents.models import Document
from apps.documents.views.api import move_document
from apps.knowledge_graph.graph import invalidation


def _document(
    monkeypatch,
    *,
    source_can_edit: bool,
    destination_can_edit: bool,
    has_figures=False,
):
    class FakeDocument:
        _default_manager = MagicMock()
        _meta = SimpleNamespace(label_lower="apps_documents.rawtextdocument")

    source = MagicMock()
    source.pk = 3
    source.user_can_edit.return_value = source_can_edit
    destination = MagicMock()
    destination.pk = 9
    destination.user_can_edit.return_value = destination_can_edit
    document = FakeDocument()
    document.pkid = 7
    document.id = uuid.uuid4()
    document._state = SimpleNamespace(db="default")
    document.collection = source
    document.collection_id = source.pk
    document.child_figures = MagicMock()
    document.save = MagicMock()
    document.child_figures.exists.return_value = has_figures
    document._lifecycle_lock_calls = []
    FakeDocument._locked_row = document

    @contextmanager
    def _locked_document_lifecycle_row(
        document_ref,
        collection_ids,
        *,
        using,
        active_or_building_only,
    ):
        document._lifecycle_lock_calls.append(
            (
                document_ref,
                tuple(collection_ids),
                using,
                active_or_building_only,
            )
        )
        yield FakeDocument._locked_row, tuple(sorted(set(collection_ids)))

    monkeypatch.setattr(
        invalidation,
        "locked_document_lifecycle_row",
        _locked_document_lifecycle_row,
    )
    (
        FakeDocument._default_manager.using.return_value.select_for_update.return_value.get.return_value
    ) = document
    return document, source, destination


def test_move_to_requires_actor_edit_permission_on_source_collection(monkeypatch):
    document, source, destination = _document(
        monkeypatch,
        source_can_edit=False,
        destination_can_edit=True,
    )
    actor = object()

    with pytest.raises(PermissionDenied, match="source"):
        Document.move_to(document, destination, actor=actor)

    source.user_can_edit.assert_called_once_with(actor)
    destination.user_can_edit.assert_not_called()
    document.save.assert_not_called()


def test_move_to_requires_actor_edit_permission_on_destination_collection(monkeypatch):
    document, source, destination = _document(
        monkeypatch,
        source_can_edit=True,
        destination_can_edit=False,
    )
    actor = object()

    with pytest.raises(PermissionDenied, match="destination"):
        Document.move_to(document, destination, actor=actor)

    source.user_can_edit.assert_called_once_with(actor)
    destination.user_can_edit.assert_called_once_with(actor)
    document.save.assert_not_called()


def test_move_to_rejects_parent_documents_that_own_derived_figures(monkeypatch):
    document, _source, destination = _document(
        monkeypatch,
        source_can_edit=True,
        destination_can_edit=True,
        has_figures=True,
    )

    with pytest.raises(ValidationError, match="derived figures"):
        Document.move_to(document, destination, actor=object())

    document.save.assert_not_called()


def test_move_to_uses_a_membership_only_save_to_preserve_chunks_and_artifacts(
    monkeypatch,
):
    document, _source, destination = _document(
        monkeypatch,
        source_can_edit=True,
        destination_can_edit=True,
    )

    Document.move_to(document, destination, actor=object())

    assert document.collection is destination
    document.save.assert_called_once_with(
        dont_rechunk=True,
        update_fields=["collection"],
        using="default",
    )


def test_move_to_authorizes_against_the_locked_current_source(monkeypatch):
    document, stale_source, destination = _document(
        monkeypatch,
        source_can_edit=True,
        destination_can_edit=True,
    )
    current_source = MagicMock()
    current_source.user_can_edit.return_value = False
    locked_document = SimpleNamespace(
        collection=current_source,
        child_figures=MagicMock(),
        save=MagicMock(),
    )
    locked_document.child_figures.exists.return_value = False
    type(document)._locked_row = locked_document
    actor = object()

    with pytest.raises(PermissionDenied, match="source"):
        Document.move_to(document, destination, actor=actor)

    assert len(document._lifecycle_lock_calls) == 1
    current_source.user_can_edit.assert_called_once_with(actor)
    stale_source.user_can_edit.assert_not_called()
    destination.user_can_edit.assert_not_called()
    locked_document.save.assert_not_called()


def test_move_to_routes_locking_through_the_collection_first_lifecycle_spine(
    monkeypatch,
):
    document, _source, destination = _document(
        monkeypatch,
        source_can_edit=True,
        destination_can_edit=True,
    )

    Document.move_to(document, destination, actor=object())

    assert len(document._lifecycle_lock_calls) == 1
    document_ref, collection_ids, using, active_only = (
        document._lifecycle_lock_calls[0]
    )
    assert document_ref.concrete_model_label == "apps_documents.rawtextdocument"
    assert document_ref.document_pkid == document.pkid
    assert document_ref.document_id == document.id
    assert collection_ids == (3, 9)
    assert using == "default"
    assert active_only is True


def test_move_api_passes_request_actor_and_maps_permission_denial_to_403():
    actor = MagicMock(is_authenticated=True)
    destination = MagicMock()
    document = MagicMock(id=uuid.uuid4(), title="Restricted")
    document.move_to.side_effect = PermissionDenied("destination denied")
    request = RequestFactory().post(
        "/move/",
        data=json.dumps({"new_collection_id": 23}),
        content_type="application/json",
    )
    request.user = actor

    with (
        patch("apps.documents.views.api.Document.get_by_id", return_value=document),
        patch("apps.documents.views.api.Collection.objects.get", return_value=destination),
    ):
        response = move_document(request, document.id)

    assert response.status_code == 403
    assert json.loads(response.content) == {"error": "destination denied"}
    document.move_to.assert_called_once_with(destination, actor=actor)
