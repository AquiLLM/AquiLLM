from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest
from django.core.exceptions import ValidationError
from django.db import models


def test_persisted_parent_save_locks_child_old_and_new_parent_before_write(monkeypatch):
    from apps.collections.models import Collection
    from apps.knowledge_graph.graph import invalidation

    collection = Collection(pk=11, name="child", parent_id=5)
    collection._state.adding = False
    collection._state.db = "default"
    query = MagicMock()
    query.filter.return_value = query
    query.values.return_value = query
    query.first.return_value = {"parent_id": 3}
    monkeypatch.setattr(Collection._base_manager, "using", lambda alias: query)
    actions = []
    calls = []

    @contextmanager
    def locked_rows(collection_id, parent_ids, *, using):
        calls.append((collection_id, tuple(parent_ids), using))
        actions.append("collections_locked")
        yield 3, (3, 5, 11)

    monkeypatch.setattr(invalidation, "locked_collection_parent_rows", locked_rows)
    monkeypatch.setattr(
        models.Model,
        "save",
        lambda self, *args, **kwargs: actions.append(
            ("collection_written", tuple(kwargs["update_fields"]))
        ),
    )

    collection.save(update_fields=(field for field in ("parent", "name")))

    assert actions == [
        "collections_locked",
        ("collection_written", ("parent", "name")),
    ]
    assert calls == [(11, (3, 5), "default")]


def test_parent_queryset_and_bulk_mutations_are_rejected():
    from apps.collections.models import Collection

    with pytest.raises(ValidationError, match="parent"):
        Collection.objects.all().update(parent_id=7)
    with pytest.raises(ValidationError, match="parent"):
        Collection.objects.bulk_update([Collection(pk=11)], ["parent"])
    with pytest.raises(ValidationError, match="parent"):
        Collection.objects.bulk_create(
            [Collection(pk=11, name="child")],
            update_conflicts=True,
            unique_fields=["pk"],
            update_fields=["parent"],
        )
