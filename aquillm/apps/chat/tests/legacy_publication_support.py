"""Actual canonical and retained-I3 sources for default publication contracts."""
from django.contrib.auth import get_user_model

from apps.collections.models import Collection, CollectionPermission
from apps.documents.models import RawTextDocument, TextChunk


def publication_source(request, route):
    user = get_user_model().objects.create_user(username="legacy-publication")
    if route.startswith("figure_"):
        from apps.documents.services.representation_activation import (
            _begin, activate_representation, stage_representation,
        )
        from apps.documents.services.representation_caption import caption_chunker_key
        from apps.documents.services.representation_inputs import extraction_input, representation_chunks
        from apps.documents.tests.representation_support import reference
        document, output, old = request.getfixturevalue("figure")
        ref = reference(document)
        origin, revision, extraction = extraction_input(ref, output.extraction_revision_id, using="default")
        key = caption_chunker_key()
        chunks = representation_chunks(origin, extraction, key)
        lease = _begin(ref, origin, revision, extraction, chunks, key, using="default")
        if route == "figure_chunks":
            stage_representation(lease, chunks=chunks, using="default")
            assert activate_representation(lease, expected_active_key=None, using="default") == "activated"
            chunk = document.chunks.get()
            assert chunk.representation_generation_id == lease.generation_key
        else:
            old.refresh_from_db()
            assert old.representation_generation_id is None and old.legacy_source_snapshot_id is not None
            chunk = old
        document.refresh_from_db()
        collection = document.collection
    else:
        collection = Collection.objects.create(name="Synthetic")
        document = RawTextDocument(title="Synthetic source", full_text="Capacity is 18 litres.",
            collection=collection, ingested_by=user)
        document.save(dont_rechunk=True)
        chunk = TextChunk.objects.bulk_create([TextChunk(content=document.full_text,
            start_position=0, end_position=len(document.full_text), chunk_number=0,
            doc_id=document.id)])[0]
    permission = CollectionPermission.objects.create(user=user, collection=collection, permission="VIEW")
    return user, collection, permission, document, chunk
