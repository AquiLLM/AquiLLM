import pytest
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile

from aquillm.models import Collection, ImageUploadDocument, TextChunk, create_chunks


class _DummyChannelLayer:
    async def group_send(self, *_args, **_kwargs):
        return None


@pytest.mark.django_db
def test_create_chunks_avoids_text_image_position_collision(monkeypatch, settings, tmp_path):
    settings.STORAGES = {
        "default": {
            "BACKEND": "django.core.files.storage.FileSystemStorage",
            "OPTIONS": {"location": str(tmp_path)},
        },
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    }

    monkeypatch.setattr("aquillm.models.create_chunks.delay", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("aquillm.models.get_channel_layer", lambda: _DummyChannelLayer())
    monkeypatch.setattr("aquillm.models._doc_image_data_url", lambda _doc: "data:image/png;base64,AAAA")
    monkeypatch.setattr(
        "aquillm.models.get_embeddings",
        lambda texts, input_type=None: [[0.0] * 1024 for _ in texts],
    )
    monkeypatch.setattr(
        "aquillm.models.get_embedding",
        lambda text, input_type=None: [0.0] * 1024,
    )
    monkeypatch.setenv("APP_RAG_ENABLE_IMAGE_CHUNKS", "1")

    user = User.objects.create_user(username="chunk-user", password="pw12345")
    collection = Collection.objects.create(name="Chunk Collision Collection")

    # Keep text shorter than CHUNK_SIZE (default 2048) so the first text chunk is (0, 116).
    text = "x" * 116
    doc = ImageUploadDocument.objects.create(
        title="position-collision",
        full_text=text,
        collection=collection,
        ingested_by=user,
        image_file=SimpleUploadedFile("collision.png", b"\x89PNG....", content_type="image/png"),
    )

    create_chunks.run(str(doc.id))

    chunks = list(TextChunk.objects.filter(doc_id=doc.id).order_by("chunk_number"))
    assert len(chunks) == 2

    text_chunk = next(chunk for chunk in chunks if chunk.modality == TextChunk.Modality.TEXT)
    image_chunk = next(chunk for chunk in chunks if chunk.modality == TextChunk.Modality.IMAGE)

    assert (text_chunk.start_position, text_chunk.end_position) == (0, 116)
    assert (image_chunk.start_position, image_chunk.end_position) != (0, 116)
