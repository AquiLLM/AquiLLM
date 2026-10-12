"""Caption publication uses durable media; retained legacy helper coordinates stay unique."""
from unittest.mock import Mock

import pytest
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile

from aquillm.models import Collection, ImageUploadDocument, TextChunk, create_chunks
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from ._chunk_graph_lifecycle_support import (
    media_storage, persist_document, persist_media_document, assert_media_generation,
    configure_chunking_runtime,
)


@pytest.mark.django_db
def test_retained_legacy_preparation_avoids_text_image_position_collision(monkeypatch, media_storage):
    from apps.documents.tasks import chunking
    user = User.objects.create_user(username="legacy-collision")
    collection = Collection.objects.create(name="Legacy collision")
    text = "x" * 116
    document = persist_document(ImageUploadDocument, user=user, collection=collection,
        text=text, ingestion_complete=True,
        image_file=SimpleUploadedFile("collision.png", b"synthetic legacy pixels", content_type="image/png"))
    monkeypatch.setattr(chunking, "doc_image_data_url", lambda _: "data:image/png;base64,AAAA")
    monkeypatch.setenv("APP_RAG_ENABLE_IMAGE_CHUNKS", "1")
    chunks, image = chunking._prepared_chunks(document)
    assert len(chunks) == 1 and image is not None
    assert (chunks[0].start_position, chunks[0].end_position) == (0, 116)
    assert (image.start_position, image.end_position) != (0, 116)
    rows = TextChunk.objects.bulk_create([*chunks, image])
    assert len(rows) == 2
    assert len({(row.start_position, row.end_position) for row in rows}) == 2
    assert len({row.chunk_number for row in rows}) == 2
    assert {row.modality for row in rows} == {TextChunk.Modality.TEXT, TextChunk.Modality.IMAGE}
    assert all(row.representation_generation_id is None for row in rows)


@pytest.mark.django_db(transaction=True)
def test_ledger_media_publication_keeps_caption_artifact_and_unique_coordinates(monkeypatch, media_storage):
    from apps.documents.services import representation_snapshot_build
    configure_chunking_runtime(monkeypatch)
    user = User.objects.create_user(username="ledger-collision")
    collection = Collection.objects.create(name="Ledger collision")
    pixels, text = b"synthetic persisted media bytes", "x" * 116
    document, output, (args, kwargs) = persist_media_document(
        ImageUploadDocument, user=user, collection=collection, text=text,
        pixels=pixels, monkeypatch=monkeypatch)
    unsupported_fallback = Mock(side_effect=AssertionError("ledger media used canonical fallback"))
    monkeypatch.setattr(representation_snapshot_build, "begin_canonical_snapshot_representation", unsupported_fallback)
    assert create_chunks.run(*args, **kwargs) == "activated"
    chunk = assert_media_generation(document, output, pixels)
    # Caption-only sources retain pixels in the manifest, not a fabricated
    # vision embedding or duplicate image-coordinate row.
    assert chunk.modality == TextChunk.Modality.TEXT
    assert (chunk.start_position, chunk.end_position, chunk.chunk_number) == (0, 116, 0)
    assert chunk.representation_generation_id is not None
    assert list(document.chunks.values_list("pk", flat=True)) == [chunk.pk]
    # Redelivery cannot append a duplicate caption or manufacture a text row.
    assert create_chunks.run(*args, **kwargs) == "stale"
    assert list(document.chunks.values_list("pk", flat=True)) == [chunk.pk]
    unsupported_fallback.assert_not_called()


@pytest.mark.django_db(transaction=True)
def test_unbacked_image_owner_stays_closed(monkeypatch, media_storage):
    configure_chunking_runtime(monkeypatch)
    user = User.objects.create_user(username="unbacked-image")
    collection = Collection.objects.create(name="Unbacked image")
    document = persist_document(ImageUploadDocument, user=user, collection=collection,
        text="caption", ingestion_complete=False,
        image_file=SimpleUploadedFile("unbacked.png", b"pixels", content_type="image/png"))
    with pytest.raises(SourceScopeUnavailable, match="canonical_snapshot_owner_unsupported"):
        create_chunks.run(str(document.id), document.full_text_hash, document._meta.label_lower, document.pkid)
    assert not TextChunk.objects.filter(doc_id=document.id).exists()
