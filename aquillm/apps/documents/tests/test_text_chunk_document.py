from __future__ import annotations

import uuid

import pytest
from django.core.exceptions import ValidationError

from apps.documents.models import chunks


class _FakeQuerySet:
    def __init__(self, manager: _FakeManager):
        self.manager = manager

    def first(self):
        self.manager.first_calls += 1
        return self.manager.document


class _FakeManager:
    def __init__(self, document):
        self.document = document
        self.filter_calls = []
        self.first_calls = 0

    def filter(self, **kwargs):
        self.filter_calls.append(kwargs)
        return _FakeQuerySet(self)


def _fake_document_type(manager):
    return type("FakeDocument", (), {"objects": manager})


def test_document_stops_lookup_after_first_match(monkeypatch):
    doc_id = uuid.uuid4()
    document = object()
    first_manager = _FakeManager(None)
    second_manager = _FakeManager(document)
    third_manager = _FakeManager(object())
    document_types = [
        _fake_document_type(first_manager),
        _fake_document_type(second_manager),
        _fake_document_type(third_manager),
    ]
    monkeypatch.setattr(
        chunks, "_get_descended_from_document", lambda: document_types
    )

    text_chunk = chunks.TextChunk(doc_id=doc_id)

    assert text_chunk.document is document
    assert first_manager.filter_calls == [{"id": doc_id}]
    assert first_manager.first_calls == 1
    assert second_manager.filter_calls == [{"id": doc_id}]
    assert second_manager.first_calls == 1
    assert third_manager.filter_calls == []
    assert third_manager.first_calls == 0


def test_document_raises_validation_error_when_no_subtype_matches(monkeypatch):
    doc_id = uuid.uuid4()
    managers = [_FakeManager(None) for _ in range(3)]
    monkeypatch.setattr(
        chunks,
        "_get_descended_from_document",
        lambda: [_fake_document_type(manager) for manager in managers],
    )
    text_chunk = chunks.TextChunk(doc_id=doc_id)

    with pytest.raises(
        ValidationError,
        match=r"TextChunk None is not associated with a document!",
    ):
        text_chunk.document

    assert all(manager.first_calls == 1 for manager in managers)
