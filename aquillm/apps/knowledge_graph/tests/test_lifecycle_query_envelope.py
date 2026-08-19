from __future__ import annotations

import uuid
from types import SimpleNamespace


def _ref(label: str, pkid: int):
    from apps.knowledge_graph.graph.invalidation import DocumentLifecycleRef

    return DocumentLifecycleRef(
        concrete_model_label=label,
        document_pkid=pkid,
        document_id=uuid.UUID(int=pkid, version=4),
    )


def test_document_ambiguity_audit_batches_once_per_model_and_predicate(monkeypatch):
    from apps.documents import models as document_models
    from apps.knowledge_graph.graph import invalidation

    labels = ("apps_documents.rawtextdocument", "apps_documents.pdfdocument")
    refs = tuple(_ref(labels[index % 2], index + 1) for index in range(6))
    rows_by_label = {
        label: {
            ref.document_id: ref.document_pkid
            for ref in refs
            if ref.concrete_model_label == label
        }
        for label in labels
    }
    calls = []

    class Query:
        def __init__(self, label):
            self.label = label
            self.document_ids = ()

        def using(self, alias):
            assert alias == "default"
            return self

        def filter(self, *, id__in):
            calls.append((self.label, tuple(id__in)))
            clone = Query(self.label)
            clone.document_ids = tuple(id__in)
            return clone

        def order_by(self, *_fields):
            return self

        def values_list(self, *_fields):
            rows = rows_by_label[self.label]
            return tuple(
                (rows[document_id], document_id)
                for document_id in self.document_ids
                if document_id in rows
            )

    fake_models = tuple(
        SimpleNamespace(
            _meta=SimpleNamespace(label_lower=label),
            _base_manager=Query(label),
        )
        for label in labels
    )
    monkeypatch.setattr(document_models, "DESCENDED_FROM_DOCUMENT", fake_models)
    monkeypatch.setattr(invalidation, "_QUERY_PREDICATE_BATCH_SIZE", 2)

    invalidation._assert_unambiguous_document_refs(refs, using="default")

    assert len(calls) == 6
    assert all(len(document_ids) <= 2 for _label, document_ids in calls)


def test_lifecycle_predicate_batcher_never_exceeds_five_thousand():
    from apps.knowledge_graph.graph.invalidation import (
        _QUERY_PREDICATE_BATCH_SIZE,
        _predicate_batches,
    )

    values = tuple(range(10_003))
    batches = tuple(_predicate_batches(values))

    assert _QUERY_PREDICATE_BATCH_SIZE == 5_000
    assert tuple(value for batch in batches for value in batch) == values
    assert tuple(map(len, batches)) == (5_000, 5_000, 3)


def test_origin_exact_document_row_locks_batch_by_concrete_model(monkeypatch):
    from django.apps import apps as django_apps

    from apps.knowledge_graph.graph import invalidation

    labels = ("apps_documents.rawtextdocument", "apps_documents.pdfdocument")
    refs = tuple(_ref(labels[index % 2], index + 1) for index in range(7))
    rows_by_label = {
        label: tuple(
            SimpleNamespace(pkid=ref.document_pkid, id=ref.document_id)
            for ref in refs
            if ref.concrete_model_label == label
        )
        for label in labels
    }
    calls = []

    class Query:
        def __init__(self, label, rows):
            self.label = label
            self.rows = rows

        def using(self, alias):
            assert alias == "default"
            return self

        def select_for_update(self):
            calls.append(self.label)
            return self

        def filter(self, *, pkid__in):
            pkids = set(pkid__in)
            return Query(
                self.label,
                tuple(row for row in self.rows if row.pkid in pkids),
            )

        def order_by(self, *_fields):
            return tuple(sorted(self.rows, key=lambda row: (row.pkid, row.id.int)))

    models = {
        label: SimpleNamespace(_base_manager=Query(label, rows))
        for label, rows in rows_by_label.items()
    }
    monkeypatch.setattr(django_apps, "get_model", models.__getitem__)
    monkeypatch.setattr(invalidation, "_QUERY_PREDICATE_BATCH_SIZE", 2)

    invalidation._lock_exact_document_rows(refs, using="default")

    assert calls.count("apps_documents.rawtextdocument") == 2
    assert calls.count("apps_documents.pdfdocument") == 2
