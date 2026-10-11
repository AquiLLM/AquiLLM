"""I6: profile rollback reuses only verified vectors for the same representation."""
import pytest

from apps.documents.models import DocumentRepresentationGeneration, DocumentRepresentationState, TextChunk
from apps.documents.models.embedding_index import ChunkEmbedding, DocumentEmbeddingState, EmbeddingGeneration
from lib.embeddings.tests.test_document_provider import transport as transport
from .representation_support import retained as retained, upload as upload, revision
from .representation_rollback_support import active, build_dense, chunk_records, execute
from .representation_cutover_support import (
    assert_dense_generation, configure_cutover, distinct_verified_space, retrieve_original_turn,
)

pytestmark = pytest.mark.django_db(transaction=True)


def test_two_verified_profiles_switch_and_rollback_without_relabeling_retired_source(
    retained, transport, monkeypatch, settings,
):
    doc, output, legacy = retained
    original, space_a = output.extraction_revision, transport[0]
    space_c = distinct_verified_space()
    assert space_a.model == space_c.model and space_a.space_key != space_c.space_key
    assert execute(doc, original.pk, original.parser_key)["documents"][0]["status"] == "activated"
    generation_a = active(doc)
    _, dense_a = build_dense(doc, space_a)
    _, dense_c = build_dense(doc, space_c)
    key_a, key_c = dense_a.generation_keys[0], dense_c.generation_keys[0]
    assert key_a != key_c
    dense_owner_ids = set(EmbeddingGeneration.objects.filter(pk__in=(key_a, key_c)).values_list("owner_id", flat=True))
    assert len(dense_owner_ids) == 2
    before = chunk_records(generation_a)
    ids = {row["pk"] for row in before}
    before_vectors = list(ChunkEmbedding.objects.order_by("pk").values(
        "pk", "generation_id", "chunk_id", "receipt", "serving_binding"))
    configure_cutover(monkeypatch, settings, space_a)
    first = retrieve_original_turn(doc, monkeypatch)
    assert_dense_generation(first, key_a, ids)

    def switch_after_capture(runtime):
        assert runtime._frozen_retrieval_scope.space.space_key == space_a.space_key
        settings.DOCUMENT_EMBEDDING_SPACE_KEY = space_c.space_key
    changed = retrieve_original_turn(doc, monkeypatch, query="table", before_search=switch_after_capture)
    assert changed.frozen.space.space_key == space_a.space_key
    assert changed.frozen.dense.generation_keys == (key_a,)
    assert all(row.embedding_generation_key == key_a for _, rows in changed.ranked for row in rows)
    assert not changed.result[0], "a profile change withdraws dense output at final handoff"
    assert {row.pk for row in changed.result[2]} == ids, "unchanged lexical evidence remains usable"
    current = retrieve_original_turn(doc, monkeypatch)
    assert current.turn is not first.turn and current.frozen.space.space_key == space_c.space_key
    assert_dense_generation(current, key_c, ids)
    settings.DOCUMENT_EMBEDDING_SPACE_KEY = space_a.space_key
    rolled_back = retrieve_original_turn(doc, monkeypatch)
    assert_dense_generation(rolled_back, key_a, ids)
    assert chunk_records(generation_a) == before
    assert list(ChunkEmbedding.objects.order_by("pk").values(
        "pk", "generation_id", "chunk_id", "receipt", "serving_binding")) == before_vectors

    replacement = revision(output, "The corrected table is 100 units.", "i6-cutover-b")
    assert execute(doc, replacement, "i6-cutover-b")["documents"][0]["status"] == "activated"
    generation_b = active(doc)
    assert execute(doc, original.pk, original.parser_key)["documents"][0]["status"] == "activated"
    generation_a2 = active(doc)
    doc.refresh_from_db()
    assert generation_a2.pk not in (generation_a.pk, generation_b.pk)
    for space in (space_a, space_c):
        settings.DOCUMENT_EMBEDDING_SPACE_KEY = space.space_key
        lexical = retrieve_original_turn(doc, monkeypatch, query="table")
        assert lexical.frozen.dense.generation_keys == ()
        assert not any(rows for _, rows in lexical.ranked)
        assert {row.pk for row in lexical.result[2]} == set(doc.chunks.values_list("pk", flat=True))
        assert ids.isdisjoint({row.pk for row in lexical.result[2]})
    assert chunk_records(generation_a) == before and TextChunk.objects.filter(pk=legacy.pk).exists()
    document_id, historical = doc.id, {legacy.pk} | ids | {r["pk"] for r in chunk_records(generation_b)}
    current_ids = set(doc.chunks.values_list("pk", flat=True))
    doc.delete()
    assert not TextChunk.objects.filter(pk__in=historical | current_ids).exists()
    assert not DocumentRepresentationState.objects.filter(document_id=document_id).exists()
    assert not DocumentRepresentationGeneration.objects.filter(pk__in=(generation_a.pk, generation_b.pk, generation_a2.pk)).exists()
    assert not EmbeddingGeneration.objects.filter(pk__in=(key_a, key_c)).exists()
    assert not DocumentEmbeddingState.objects.filter(pk__in=dense_owner_ids).exists()
    assert not ChunkEmbedding.objects.filter(pk__in=[row["pk"] for row in before_vectors]).exists()
