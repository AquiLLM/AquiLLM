"""Retired context is its own snapshot, including explicit damaged-history handling."""
import pytest
from django.test import RequestFactory

from apps.documents.models import TextChunk
from apps.documents.views.api import chunk_detail
from .representation_support import retained as retained, upload as upload, activate

pytestmark = pytest.mark.django_db(transaction=True)


def test_old_citation_uses_saved_canonical_context(retained):
    from apps.documents.services.representation_store import load_chunk_source
    doc, output, old = retained
    activate(doc, output)
    source = load_chunk_source(old.pk, using="default")
    assert source.canonical_text == "old table 10"
    assert source.canonical_text[source.start:source.end] == old.content
    request = RequestFactory().get("/chunk/", {"include_full_text": "1"})
    request.user = doc.ingested_by
    response = chunk_detail(request, old.pk)
    import json
    assert response.status_code == 200
    assert json.loads(response.content)["document"]["full_text"] == "old table 10"


def test_bad_legacy_slice_does_not_block_new_generation_or_invent_context(retained):
    import json
    doc, output, old = retained
    TextChunk.objects.filter(pk=old.pk).update(start_position=99, end_position=111)
    activate(doc, output)
    doc.refresh_from_db()
    old.refresh_from_db()
    assert doc.full_text == "corrected table 100"
    assert old.content == "old table 10"
    assert old.start_position == 99
    assert old.legacy_context_status == "unavailable"
    request = RequestFactory().get("/chunk/")
    request.user = doc.ingested_by
    response = chunk_detail(request, old.pk)
    payload = json.loads(response.content)
    assert response.status_code == 200
    assert payload["source_context_status"] == "unavailable"
    assert not payload["document"].get("full_text")
    assert not payload["document"]["has_pdf"]
    assert payload["image_url"] is None


@pytest.mark.parametrize("compact", [False, True])
def test_historical_api_never_hydrates_complete_source_models(retained, monkeypatch, compact):
    import json
    from apps.documents.models import RawTextDocument, DocumentRepresentationGeneration, LegacyDocumentSourceSnapshot
    doc, output, old = retained
    activate(doc, output)
    for model, fields in ((RawTextDocument, {"full_text"}),
                          (DocumentRepresentationGeneration, {"canonical_text", "manifest"}),
                          (LegacyDocumentSourceSnapshot, {"canonical_text"})):
        original = model.from_db
        def check(db, field_names, values, original=original, forbidden=fields):
            assert not forbidden.intersection(field_names)
            return original(db, field_names, values)
        monkeypatch.setattr(model, "from_db", check)
    request = RequestFactory().get("/chunk/", {"include_full_text": "0" if compact else "1"})
    request.user = doc.ingested_by
    response = chunk_detail(request, old.pk)
    assert response.status_code == 200
    payload = json.loads(response.content)
    assert payload["source_context_status"] == "verified"
    if compact:
        assert "full_text" not in payload["document"]
    else:
        assert payload["document"]["full_text"] == old.content


def test_unicode_historical_window_uses_codepoint_offsets(retained, monkeypatch):
    import json
    from apps.documents.views import api
    doc, output, old = retained
    text = "α😀漢字 " * 1000
    activate(doc, output, text=text)
    doc.refresh_from_db()
    chunk = doc.chunks.first()
    monkeypatch.setattr(api, "FULL_TEXT_WINDOW_THRESHOLD", 30)
    monkeypatch.setattr(api, "FULL_TEXT_WINDOW_PADDING", 7)
    activate(doc, output, text="new canonical source", parser="parser-v3")
    request = RequestFactory().get("/chunk/")
    request.user = doc.ingested_by
    response = chunk_detail(request, chunk.pk)
    payload = json.loads(response.content)
    offset = max(0, chunk.start_position - 7)
    assert payload["document"]["full_text"] == text[offset:chunk.end_position + 7]
    assert payload["document"]["text_offset"] == offset


def test_historical_context_permission_revoked_during_acquisition_is_not_exposed(retained, monkeypatch):
    from apps.collections.models import CollectionPermission
    from apps.documents.services import representation_citation_context as context
    doc, output, old = retained
    activate(doc, output)
    original = context.load_citation_context
    def revoke(*args, **kwargs):
        result = original(*args, **kwargs)
        CollectionPermission.objects.filter(user=doc.ingested_by, collection=doc.collection).delete()
        return result
    monkeypatch.setattr(context, "load_citation_context", revoke)
    request = RequestFactory().get("/chunk/")
    request.user = doc.ingested_by
    assert chunk_detail(request, old.pk).status_code == 403
