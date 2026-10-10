"""Operator scope and dry-run invariants through the actual Django command."""
from io import StringIO
import json

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError
from django.db import connection
from django.test.utils import CaptureQueriesContext

from apps.collections.models import Collection
from apps.documents.models import RawTextDocument, TextChunk
from apps.documents.models.embedding_index import (
    ChunkEmbedding, DocumentEmbeddingState, EmbeddingGeneration,
)
from apps.documents.services.embedding_index_store import persist_embedding_space
from apps.documents.tests.test_embedding_generation import legacy as legacy
from lib.embeddings.tests.test_document_provider import transport as transport

pytestmark = pytest.mark.django_db(transaction=True)


def invoke(doc, space, **changes):
    options = dict(model_label=doc._meta.label_lower, document_pkid=doc.pkid,
                   document_id=str(doc.id), space_key=space.space_key,
                   database="default", timeout_ms=120000, dry_run=True)
    options.update(changes)
    output = StringIO()
    call_command("reindex_document_embeddings", stdout=output, **options)
    return json.loads(output.getvalue())


def stored_chunks():
    return [dict(row, embedding=list(row["embedding"]) if row["embedding"] is not None else None)
            for row in TextChunk.objects.order_by("pk").values()]


def test_dry_run_is_exact_and_never_builds_or_calls_provider(legacy, transport):
    doc, _ = legacy
    space, calls = transport
    persist_embedding_space(space, using="default")
    TextChunk.objects.filter(doc_id=doc.id).update(embedding=[1.0] * 1024)
    original = stored_chunks()
    with CaptureQueriesContext(connection) as queries:
        report = invoke(doc, space)
    assert calls == []
    assert not any(query["sql"].lstrip().upper().startswith(
        ("INSERT", "UPDATE", "DELETE", "ALTER", "CREATE", "DROP")
    ) for query in queries)
    assert not EmbeddingGeneration.objects.exists()
    assert not DocumentEmbeddingState.objects.exists()
    assert not ChunkEmbedding.objects.exists()
    assert stored_chunks() == original
    assert report["dry_run"] is True and report["space_key"] == space.space_key
    assert report["input_integrity_verified"] is False
    assert report["dense_coverage_verified"] is False
    assert [(row["document_id"], row["active_chunk_count"], row["status"])
            for row in report["documents"]] == [(str(doc.id), 1, "planned")]


@pytest.mark.parametrize("changes", [
    {"space_key": "display-model-name"}, {"space_key": "A" * 64},
    {"database": "unavailable"}, {"database": "projection_state"},
    {"model_label": "auth.user"}, {"model_label": "RawTextDocument"},
    {"document_id": "malformed"}, {"document_pkid": True},
    {"document_pkid": 0}, {"timeout_ms": True}, {"timeout_ms": 0},
    {"timeout_ms": 120001}, {"collection_id": True},
    {"limit": 1}, {"after_pkid": 0},
])
def test_invalid_scope_is_rejected_before_any_database_or_provider(
    legacy, transport, changes,
):
    doc, _ = legacy
    space, calls = transport

    def forbidden(*args):
        raise AssertionError("invalid command reached the database")

    with connection.execute_wrapper(forbidden), pytest.raises(CommandError):
        invoke(doc, space, **changes)
    assert calls == []


def test_unknown_digest_never_creates_a_profile_or_generation(legacy, transport):
    doc, _ = legacy
    with pytest.raises(CommandError):
        invoke(doc, transport[0], space_key="f" * 64)
    assert not EmbeddingGeneration.objects.exists()
    assert transport[1] == []


def test_collection_inventory_is_bounded_keyset_and_never_crosses_collection(
    legacy, transport,
):
    doc, _ = legacy
    space, calls = transport
    persist_embedding_space(space, using="default")
    docs = [doc]
    for index, collection in enumerate((doc.collection, Collection.objects.create(name="Excluded"))):
        source = f"source {index}"
        other = RawTextDocument(title="Next", full_text=source, collection=collection,
                                ingested_by=doc.ingested_by)
        other.save(dont_rechunk=True)
        TextChunk.objects.create(doc_id=other.id, content=source, start_position=0,
                                 end_position=len(source), chunk_number=0)
        docs.append(other)
    options = dict(document_pkid=None, document_id=None, collection_id=doc.collection_id,
                   limit=1, after_pkid=0)
    first = invoke(doc, space, **options)
    assert [row["document_id"] for row in first["documents"]] == [str(doc.id)]
    assert first["next_after_pkid"] == doc.pkid
    options["after_pkid"] = first["next_after_pkid"]
    second = invoke(doc, space, **options)
    assert [row["document_id"] for row in second["documents"]] == [str(docs[1].id)]
    options["after_pkid"] = second["next_after_pkid"]
    assert invoke(doc, space, **options)["documents"] == []
    assert calls == [] and not EmbeddingGeneration.objects.exists()


@pytest.mark.parametrize("field,value", [
    ("collection_id", None), ("collection_id", True), ("collection_id", -1),
    ("limit", True), ("limit", 0), ("limit", 101),
    ("after_pkid", True), ("after_pkid", -1),
])
def test_inventory_rejects_missing_or_unbounded_inputs(legacy, transport, field, value):
    options = dict(document_pkid=None, document_id=None,
                   collection_id=legacy[0].collection_id, limit=25, after_pkid=0)
    options[field] = value

    def forbidden(*args):
        raise AssertionError("invalid inventory reached the database")

    with connection.execute_wrapper(forbidden), pytest.raises(CommandError):
        invoke(legacy[0], transport[0], **options)


def test_non_dry_run_uses_actual_durable_generation_with_original_scope(
    legacy, transport, monkeypatch,
):
    from apps.documents.tasks import embedding_index

    doc, _ = legacy
    space, calls = transport
    persist_embedding_space(space, using="default")
    deliveries = []
    monkeypatch.setattr(embedding_index.build_embedding_generation, "delay",
                        lambda *args, **kwargs: deliveries.append((args, kwargs)))
    report = invoke(doc, space, dry_run=False, timeout_ms=12345)
    generation = EmbeddingGeneration.objects.select_related("owner").get()
    assert generation.owner.document_id == doc.id
    assert generation.owner.document_pkid == doc.pkid
    assert generation.owner.model_label == doc._meta.label_lower
    assert generation.owner.space_id == space.space_key
    assert generation.state == "building"
    assert [row["generation_key"] for row in report["documents"]] == [str(generation.key)]
    assert deliveries == [((str(generation.key),), dict(using="default", timeout_ms=12345))]
    assert calls == [] and not ChunkEmbedding.objects.exists()
