from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest


def test_public_sampler_skips_early_empty_documents_without_losing_later_text(monkeypatch):
    """An empty leading document must not consume a bounded sample slot."""

    from apps.collections.services import schema_generation

    rows = {"document-a": [], "document-b": [], "document-c": [(3, 0, "charlie")], "document-d": [(4, 0, "delta")]}
    calls = []
    monkeypatch.setattr(schema_generation, "_completed_collection_documents", lambda collection_id, **kwargs: iter({"id": document_id, "full_text_hash": "a" * 64} for document_id in sorted(rows)))
    monkeypatch.setattr(
        schema_generation,
        "_next_sample_chunk",
        lambda document_id, after_chunk_number, character_limit: calls.append(document_id) or next((schema_generation.SchemaSample(document_id, chunk_id, number, text[:character_limit]) for chunk_id, number, text in rows[document_id] if number > after_chunk_number), None),
    )

    samples = schema_generation.sample_collection_chunks(1, max_chunks=2, max_characters=20)

    assert [sample.document_id for sample in samples] == ["document-c", "document-d"]
    assert calls == ["document-a", "document-b", "document-c", "document-d"]


def test_locked_source_signature_locks_all_documents_but_hashes_only_completed(monkeypatch):
    """The final fence must block document completion changes and FK inserts."""

    from django.apps import apps
    from apps.collections.services import schema_generation

    names = ("PDFDocument", "TeXDocument", "RawTextDocument", "VTTDocument", "HandwrittenNotesDocument", "ImageUploadDocument", "MediaUploadDocument", "DocumentFigure")
    records = {"PDFDocument": [{"id": "complete", "full_text_hash": "a" * 64, "ingestion_complete": True}], "TeXDocument": [{"id": "pending", "full_text_hash": "b" * 64, "ingestion_complete": False}]}
    filters, locks, projections = [], [], []

    class QuerySet:
        def __init__(self, name):
            self.name = name

        def filter(self, **kwargs):
            filters.append((self.name, kwargs))
            return self

        def select_for_update(self):
            locks.append(self.name)
            return self

        def order_by(self, *fields):
            return self

        def values(self, *fields):
            projections.append((self.name, fields))
            return self

        def iterator(self):
            return iter(records.get(self.name, []))

    monkeypatch.setattr(apps, "get_model", lambda app_label, name: SimpleNamespace(objects=QuerySet(name)))

    signature = schema_generation._locked_collection_source_signature(1)

    digest = hashlib.sha256()
    digest.update(b"complete\0" + (b"a" * 64) + b"\n")
    assert signature == digest.hexdigest()
    assert filters == [(name, {"collection_id": 1}) for name in names]
    assert locks == list(names)
    assert all(fields == ("id", "full_text_hash", "ingestion_complete") for _, fields in projections)


@pytest.mark.parametrize("failure", ("timeout", "status", "malformed", "oversize"))
def test_direct_vllm_transport_rejects_unsafe_response_states(monkeypatch, failure):
    """A local transport must never buffer unbounded or malformed server output."""

    from apps.collections.services import schema_generation_support as support

    class Response:
        status = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, limit):
            assert limit == support._MAX_VLLM_RESPONSE_BYTES + 1
            if failure == "oversize":
                return b"x" * limit
            if failure == "malformed":
                return b"not-json"
            return b'{"ok":true}'

    if failure == "timeout":
        monkeypatch.setattr(support, "_open_local_request", lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError()))
    elif failure == "status":
        response = Response()
        response.status = 503
        monkeypatch.setattr(support, "_open_local_request", lambda *args, **kwargs: response)
    else:
        monkeypatch.setattr(support, "_open_local_request", lambda *args, **kwargs: Response())

    with pytest.raises(support.LocalVLLMTransportError):
        support._post_local_vllm_json("http://vllm:8000/v1/chat/completions", {}, {}, 1)
