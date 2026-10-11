"""Real durable worker delivery over retained image chunks and immutable pixels."""
from dataclasses import asdict
from unittest.mock import patch

import pytest
from django.contrib.auth.models import User

from apps.collections.models import Collection
from apps.documents.models import ImageUploadDocument, TextChunk
from apps.documents.models.embedding_index import EmbeddingGeneration
from apps.documents.services import embedding_index_store as store
from apps.documents.services.representation_contracts import DocumentRefV1
from apps.documents.services.representation_store import freeze_active_representations
from apps.documents.tasks import embedding_index as task
from apps.ingestion.tests.source_artifact_support import artifact_storage
from lib.embeddings.document_binding import validate_stored_binding
from lib.embeddings.document_contracts import build_budget
from lib.embeddings.document_provider import compatible_receipt, prepare_document
from lib.embeddings.provenance import input_digest
from lib.embeddings.tests.test_document_image_byte_budget import image_chunk, prepared_bytes
from lib.embeddings.tests.test_document_provider import transport
from lib.embeddings.utils import EmbeddingContractError

pytestmark = pytest.mark.django_db(transaction=True)


def retained_images(sides, space):
    # An explicit retained legacy image source; no invented representation manifest.
    images = tuple(image_chunk(side, index) for index, side in enumerate(sides))
    user = User.objects.create(username="image-byte-worker")
    collection = Collection.objects.create(name="Retained image byte boundary")
    captions = tuple(f"Image evidence {index}." for index in range(len(images)))
    doc = ImageUploadDocument(
        title="Retained image observations", full_text="\n".join(captions),
        image_file=images[0].media_artifact.storage_key,
        collection=collection, ingested_by=user,
    )
    doc.save(dont_rechunk=True)
    rows, offset = [], 0
    for index, (caption, image) in enumerate(zip(captions, images)):
        rows.append(TextChunk(
            doc_id=doc.id, content=caption, start_position=offset,
            end_position=offset + len(caption), chunk_number=index,
            modality="image", metadata={"media_artifact": asdict(image.media_artifact)},
        ))
        offset += len(caption) + 1
    rows = TextChunk.objects.bulk_create(rows)
    ref = DocumentRefV1(doc._meta.label_lower, doc.pkid, doc.id,
                        doc.collection_id, doc.full_text_hash)
    selection = freeze_active_representations((ref,), using="default").selections[0]
    assert selection.generation_key is None
    wires = tuple(prepare_document(store.source_chunk(row), space=space)[0] for row in rows)
    assert all(doc.full_text[row.start_position:row.end_position] == row.content for row in rows)
    store.persist_embedding_space(space, using="default")
    with patch.object(task.build_embedding_generation, "delay") as queue:
        key = task.rebuild_document_embeddings(selection, space_key=space.space_key, using="default")
    queue.assert_called_once_with(key, using="default", timeout_ms=120000)
    return doc, rows, wires, key


def citation_snapshot(doc):
    doc.refresh_from_db()
    return (doc.full_text, doc.full_text_hash, list(
        TextChunk.objects.filter(doc_id=doc.id).order_by("chunk_number").values(
            "pk", "content", "start_position", "end_position", "chunk_number",
            "modality", "metadata", "representation_generation_id", "legacy_source_snapshot_id",
        )
    ))


def test_worker_splits_individually_admissible_images_by_original_wire_cap(artifact_storage, transport):
    space, calls = transport
    doc, rows, wires, key = retained_images((200, 200), space)
    before = citation_snapshot(doc)
    budget = build_budget()
    assert (budget.max_input_bytes, budget.max_items) == (262144, 32)
    assert all(prepared_bytes(wire) < budget.max_input_bytes for wire in wires)
    assert sum(prepared_bytes(wire) for wire in wires) > budget.max_input_bytes
    assert sum(len(wire["text"].encode("utf-8")) for wire in wires) < space.max_batch_tokens
    assert calls == []
    assert task.build_embedding_generation.run(key, using="default") == "activated"
    assert [call["input"] for call in calls] == [[wire] for wire in wires]
    generation = EmbeddingGeneration.objects.select_related("owner").get(pk=key)
    assert (generation.state, generation.deliveries, generation.failure_kind) == ("ready", 1, "")
    assert generation.owner.active_generation_id == generation.key
    embedded = {row.chunk_id: row for row in generation.embeddings.all()}
    assert set(embedded) == {row.pk for row in rows}
    for chunk, wire in zip(rows, wires):
        row = embedded[chunk.pk]
        assert row.input_sha256 == input_digest(wire)
        assert row.media_sha256 == chunk.metadata["media_artifact"]["sha256"]
        assert compatible_receipt(list(row.vector), row.receipt, space=space,
            role="search_document", prepared_input_sha256=input_digest(wire)) == space.space_key
        validate_stored_binding(row.serving_binding, row.vector, row.receipt,
            space=space, modality="image", media_sha256=row.media_sha256)
    assert citation_snapshot(doc) == before


def test_worker_rejects_single_oversize_image_before_sdk_and_fails_durably(artifact_storage, transport):
    space, calls = transport
    doc, rows, wires, key = retained_images((300,), space)
    before = citation_snapshot(doc)
    assert build_budget().max_input_bytes == 262144
    assert prepared_bytes(wires[0]) > build_budget().max_input_bytes
    assert len(wires[0]["text"].encode("utf-8")) < space.max_input_tokens
    assert calls == []
    with pytest.raises(EmbeddingContractError, match="^document_batch_bytes$"):
        task.build_embedding_generation.run(key, using="default")
    generation = EmbeddingGeneration.objects.select_related("owner").get(pk=key)
    assert (generation.state, generation.deliveries, generation.failure_kind) == ("failed", 1, "contract")
    assert generation.owner.active_generation_id is None
    assert not generation.embeddings.exists()
    assert calls == []
    assert citation_snapshot(doc) == before
