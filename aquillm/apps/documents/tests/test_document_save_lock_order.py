from __future__ import annotations

import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from django.core.exceptions import ValidationError
from django.db import models


def test_existing_full_save_locks_old_and_new_collections_before_document_write(
    monkeypatch,
):
    from apps.documents.models import RawTextDocument
    from apps.documents.models import document as document_module
    from apps.knowledge_graph.graph import invalidation

    document_id = uuid.uuid4()
    source_hash = RawTextDocument.hash_fn("unchanged")
    document = RawTextDocument(
        pkid=7,
        id=document_id,
        title="moved directly",
        full_text="unchanged",
        full_text_hash=source_hash,
        collection_id=9,
        ingested_by_id=3,
    )
    document._state.adding = False
    document._state.db = "default"
    query = MagicMock()
    query.filter.return_value = query
    query.select_for_update.return_value = query
    query.values.return_value = query
    query.first.return_value = {
        "id": document_id,
        "full_text_hash": source_hash,
        "collection_id": 3,
    }
    monkeypatch.setattr(
        RawTextDocument._base_manager,
        "using",
        lambda alias: query,
    )
    actions = []
    lock_calls = []

    @contextmanager
    def atomic(*, using):
        yield

    @contextmanager
    def locked_row(
        document_ref,
        collection_ids,
        *,
        using,
        active_or_building_only,
    ):
        lock_calls.append(
            (
                document_ref,
                tuple(collection_ids),
                using,
                active_or_building_only,
            )
        )
        actions.append("collections_locked")
        yield (
            SimpleNamespace(
                id=document_id,
                full_text_hash=source_hash,
                collection_id=3,
            ),
            (3, 9),
        )

    monkeypatch.setattr(document_module.transaction, "atomic", atomic)
    monkeypatch.setattr(invalidation, "locked_document_lifecycle_row", locked_row)
    monkeypatch.setattr(
        invalidation,
        "document_lifecycle_row_is_locked",
        lambda instance, *, using: False,
    )
    monkeypatch.setattr(
        invalidation,
        "consume_document_save_lifecycle",
        lambda instance: None,
    )
    monkeypatch.setattr(
        models.Model,
        "save",
        lambda self, *args, **kwargs: actions.append("document_written"),
    )

    document.save(dont_rechunk=True)

    assert actions == ["collections_locked", "document_written"]
    assert len(lock_calls) == 1
    document_ref, collection_ids, using, active_only = lock_calls[0]
    assert document_ref.document_pkid == 7
    assert document_ref.document_id == document_id
    assert collection_ids == (3, 9)
    assert using == "default"
    assert active_only is True


def test_direct_collection_save_rejects_a_parent_with_derived_figures(monkeypatch):
    from apps.documents.models import RawTextDocument
    from apps.documents.models import document as document_module
    from apps.knowledge_graph.graph import invalidation

    document_id = uuid.uuid4()
    source_hash = RawTextDocument.hash_fn("unchanged")
    document = RawTextDocument(
        pkid=7,
        id=document_id,
        title="parent",
        full_text="unchanged",
        full_text_hash=source_hash,
        collection_id=9,
        ingested_by_id=3,
    )
    document._state.adding = False
    document._state.db = "default"
    query = MagicMock()
    query.filter.return_value = query
    query.values.return_value = query
    query.first.return_value = {
        "id": document_id,
        "full_text_hash": source_hash,
        "collection_id": 3,
    }
    monkeypatch.setattr(RawTextDocument._base_manager, "using", lambda alias: query)
    figures = MagicMock()
    figures.exists.return_value = True

    @contextmanager
    def locked_row(*_args, **_kwargs):
        yield (
            SimpleNamespace(
                id=document_id,
                full_text_hash=source_hash,
                collection_id=3,
                child_figures=figures,
            ),
            (3, 9),
        )

    monkeypatch.setattr(document_module.transaction, "atomic", lambda **_kwargs: null_atomic())
    monkeypatch.setattr(invalidation, "locked_document_lifecycle_row", locked_row)
    monkeypatch.setattr(
        invalidation,
        "document_lifecycle_row_is_locked",
        lambda instance, *, using: False,
    )
    write = MagicMock()
    monkeypatch.setattr(models.Model, "save", write)

    with pytest.raises(ValidationError, match="derived figures"):
        document.save(dont_rechunk=True, update_fields=["collection"])

    write.assert_not_called()


@contextmanager
def null_atomic():
    yield
