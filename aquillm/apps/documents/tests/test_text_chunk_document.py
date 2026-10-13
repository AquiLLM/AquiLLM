"""Exact citation ownership scans every concrete model and rejects collisions."""
from __future__ import annotations

import uuid

import pytest
from django.core.exceptions import ValidationError

from apps.documents import models as document_models
from apps.documents.models import chunks


class _FakeManager:
    def __init__(self, documents):
        self.documents = documents
        self.aliases = []
        self.filters = []
        self.slices = []

    def using(self, alias):
        self.aliases.append(alias)
        return self

    def filter(self, **kwargs):
        self.filters.append(kwargs)
        return self

    def __getitem__(self, bound):
        self.slices.append(bound)
        return self.documents[bound]


def _lookup(monkeypatch, documents):
    managers = [_FakeManager(rows) for rows in documents]
    monkeypatch.setattr(document_models, "DESCENDED_FROM_DOCUMENT", [
        type(f"FakeDocument{index}", (), {"_base_manager": manager})
        for index, manager in enumerate(managers)
    ])
    chunk = chunks.TextChunk(doc_id=uuid.uuid4())
    chunk._state.db = "citation_alias"
    return chunk, managers


def _assert_exact_bounded_reads(chunk, managers):
    for manager in managers:
        assert manager.aliases == ["citation_alias"]
        assert manager.filters == [{"id": chunk.doc_id}]
        assert manager.slices == [slice(None, 2, None)]


def test_document_returns_unique_owner_after_checking_every_subtype(monkeypatch):
    document = object()
    chunk, managers = _lookup(monkeypatch, [[], [document], []])
    assert chunk.document is document
    _assert_exact_bounded_reads(chunk, managers)


@pytest.mark.parametrize("matches", ["missing", "cross_model", "same_model"])
def test_document_rejects_missing_or_ambiguous_exact_owner(monkeypatch, matches):
    documents = {
        "missing": [[], [], []],
        "cross_model": [[], [object()], [object()]],
        "same_model": [[], [object(), object()], []],
    }[matches]
    chunk, managers = _lookup(monkeypatch, documents)
    with pytest.raises(ValidationError, match="Exact citation owner unavailable"):
        chunk.document
    _assert_exact_bounded_reads(chunk, managers)
