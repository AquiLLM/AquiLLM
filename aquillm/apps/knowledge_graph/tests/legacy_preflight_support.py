"""Real canonical legacy owners for isolated extraction query/preflight probes."""
from uuid import uuid4


def legacy_owner(monkeypatch, text="abc"):
    from django.contrib.auth.models import User
    from apps.collections.models import Collection
    from apps.documents.models import DocumentRepresentationState, RawTextDocument, TextChunk

    monkeypatch.setenv("KG_GRAPH_REMEDIATION_MODE", "legacy")
    user = User.objects.create_user(username="preflight-" + uuid4().hex)
    collection = Collection.objects.create(name="Preflight query control")
    document = RawTextDocument(title="Preflight control", full_text=text,
        collection=collection, ingested_by=user, ingestion_complete=True)
    document.save(dont_rechunk=True)
    TextChunk.objects.create(doc_id=document.id, content=text, start_position=0,
        end_position=len(text), chunk_number=0)
    assert document.full_text_hash == document.hash_fn(text)
    assert not DocumentRepresentationState.objects.filter(document_id=document.id).exists()
    assert document.chunks.get().representation_generation_id is None
    return document
