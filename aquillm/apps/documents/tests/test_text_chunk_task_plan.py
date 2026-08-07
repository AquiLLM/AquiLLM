from __future__ import annotations

import uuid

import pytest

from apps.documents.services.text_chunk_plan import plan_text_chunks as real_plan_text_chunks
from apps.documents.tasks import chunking


CHUNK_SIZE = 2048
OVERLAP = 384


class _DummyChannelLayer:
    async def group_send(self, *_args, **_kwargs):
        return None


class _FakeDoc:
    def __init__(self, text: str, *, with_image: bool):
        self.id = uuid.uuid4()
        self.title = "planned document"
        self.full_text = text
        self.full_text_hash = "unique-hash"
        self.ingested_by = type("User", (), {"id": 17})()
        self.ingestion_complete = False
        self.saved_update_fields = []
        if with_image:
            self.image_file = type("ImageFile", (), {"name": "source.png"})()

    def save(self, **kwargs):
        self.saved_update_fields.append(kwargs)


class _FakeChunkManager:
    def __init__(self):
        self.deleted_doc_ids = []
        self.created = []

    def filter(self, **kwargs):
        self.deleted_doc_ids.append(kwargs["doc_id"])
        return self

    def delete(self):
        return None

    def bulk_create(self, chunks):
        self.created = list(chunks)
        return self.created


class _FakeTextChunk:
    class Modality:
        TEXT = "text"
        IMAGE = "image"

    objects = _FakeChunkManager()
    image_embedding_order = []

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.embedding = kwargs.get("embedding")
        self.metadata = kwargs.get("metadata", {})

    def get_chunk_embedding(self):
        self.image_embedding_order.append(self.content)
        self.embedding = [999.0]
        return self.embedding


@pytest.mark.parametrize(("text_length", "with_image"), [(0, False), (10000, True)])
def test_create_chunks_persists_the_production_plan_without_database_boundaries(
    monkeypatch,
    text_length,
    with_image,
):
    text = "".join(str(index % 10) for index in range(text_length))
    doc = _FakeDoc(text, with_image=with_image)
    manager = _FakeChunkManager()
    _FakeTextChunk.objects = manager
    _FakeTextChunk.image_embedding_order = []
    planner_calls = []
    embedding_batches = []
    progress = []
    completions = []

    def plan_spy(value, *, chunk_size, overlap):
        planner_calls.append((value, chunk_size, overlap))
        return real_plan_text_chunks(value, chunk_size=chunk_size, overlap=overlap)

    def get_embeddings_spy(contents, *, input_type):
        embedding_batches.append((list(contents), input_type))
        return [[float(index)] for index in range(len(contents))]

    monkeypatch.setattr(
        chunking,
        "Document",
        type("Document", (), {"get_by_id": lambda _id: doc}),
    )
    monkeypatch.setattr(chunking, "DESCENDED_FROM_DOCUMENT", [])
    monkeypatch.setattr(chunking, "TextChunk", _FakeTextChunk)
    monkeypatch.setattr(chunking, "get_channel_layer", lambda: _DummyChannelLayer())
    monkeypatch.setattr(
        chunking.django_apps,
        "get_app_config",
        lambda _label: type(
            "Config",
            (),
            {"chunk_size": CHUNK_SIZE, "chunk_overlap": OVERLAP},
        )(),
    )
    monkeypatch.setattr(chunking, "plan_text_chunks", plan_spy)
    monkeypatch.setattr(chunking, "get_embeddings", get_embeddings_spy)
    monkeypatch.setattr(chunking, "doc_image_data_url", lambda _doc: "data:image/png;base64,AAAA")
    monkeypatch.setattr(
        chunking,
        "notify_ingest_monitor_progress",
        lambda doc_id, value: progress.append((doc_id, value)),
    )
    monkeypatch.setattr(
        chunking,
        "notify_ingest_monitor_complete",
        lambda doc_id: completions.append(doc_id),
    )
    monkeypatch.setenv("APP_RAG_ENABLE_IMAGE_CHUNKS", "1" if with_image else "0")

    chunking.create_chunks.run(str(doc.id))

    expected_specs = real_plan_text_chunks(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    expected_contents = [spec.content for spec in expected_specs]
    text_chunks = [
        chunk
        for chunk in manager.created
        if chunk.modality == _FakeTextChunk.Modality.TEXT
    ]
    assert planner_calls == [(text, CHUNK_SIZE, OVERLAP)]
    assert embedding_batches == [(expected_contents, "search_document")]
    assert [
        (chunk.content, chunk.start_position, chunk.end_position, chunk.chunk_number)
        for chunk in text_chunks
    ] == [
        (spec.content, spec.start_position, spec.end_position, spec.chunk_number)
        for spec in expected_specs
    ]
    assert [chunk.embedding for chunk in text_chunks] == [
        [float(index)] for index in range(len(expected_specs))
    ]
    assert progress[-1] == (doc.id, 100)
    assert completions == [doc.id]
    assert doc.ingestion_complete is True
    assert doc.saved_update_fields == [
        {"dont_rechunk": True, "update_fields": ["ingestion_complete"]}
    ]

    image_chunks = [
        chunk
        for chunk in manager.created
        if chunk.modality == _FakeTextChunk.Modality.IMAGE
    ]
    if with_image:
        assert len(image_chunks) == 1
        image_chunk = image_chunks[0]
        assert image_chunk.content == text[:800]
        assert image_chunk.chunk_number == len(expected_specs)
        assert image_chunk.start_position == expected_specs[-1].end_position
        assert image_chunk.end_position == image_chunk.start_position + 800
        assert _FakeTextChunk.image_embedding_order == [image_chunk.content]
        assert manager.created == [*text_chunks, image_chunk]
    else:
        assert image_chunks == []
        assert manager.created == []
        assert _FakeTextChunk.image_embedding_order == []
