"""Retained source, dense and vocabulary history share one deletion allowance."""
from unittest.mock import patch

import pytest
from django.db import IntegrityError, InternalError, connection, transaction
from pgvector.django import VectorField

from apps.collections.models import Collection, CollectionPermission
from apps.collections.models.vocabulary import (
    VocabularyAttachment, VocabularySet, VocabularyDraft,
    VocabularyEntry, VocabularyVersion,
)
from apps.collections.tests.test_vocabulary_scope import publish, service
from apps.documents.models import (
    DocumentRepresentationGeneration, DocumentRepresentationState,
    LegacyDocumentSourceSnapshot, TextChunk,
)
from apps.documents.models.embedding_index import (
    ChunkEmbedding, DocumentEmbeddingState, EmbeddingGeneration,
)
from apps.documents.services import embedding_index_store as store
from apps.documents.services.representation_store import freeze_active_representations
from apps.documents.tasks.embedding_index import (
    build_embedding_generation, rebuild_document_embeddings,
)
from apps.documents.tests.representation_support import (
    activate, reference, retained as retained, upload as upload,
)
from lib.embeddings.tests.test_document_provider import transport as transport

pytestmark = pytest.mark.django_db(transaction=True)
HISTORY = (
    DocumentRepresentationGeneration, DocumentRepresentationState,
    LegacyDocumentSourceSnapshot, TextChunk, DocumentEmbeddingState,
    EmbeddingGeneration, ChunkEmbedding, VocabularySet, VocabularyDraft,
    VocabularyVersion, VocabularyEntry, VocabularyAttachment,
)


def build_history(retained, transport):
    doc, output, old = retained
    assert activate(doc, output) == "activated"
    doc.refresh_from_db()
    selection = freeze_active_representations((reference(doc),), using="default").selections[0]
    store.persist_embedding_space(transport[0], using="default")
    with patch.object(build_embedding_generation, "delay"):
        key = rebuild_document_embeddings(selection, space_key=transport[0].space_key, using="default")
    assert build_embedding_generation.run(key, using="default") == "activated"
    actor, collection = doc.ingested_by, doc.collection
    actor.is_staff = True
    actor.save(update_fields=["is_staff"])
    CollectionPermission.objects.update_or_create(user=actor, collection=collection,
        defaults={"permission": "MANAGE"})
    _, version = publish(actor, collection)
    service().attach_vocabulary(actor=actor, collection=collection,
        version=version, expected_revision=0)
    return doc, old, collection


def identities(models):
    return {model: frozenset(model.objects.values_list("pk", flat=True)) for model in models}


def row_state(membership):
    # Small synthetic fixture only; include pointers, leases, manifests and owner
    # bindings, not just row survival. Never log source/metadata field values.
    result = {}
    for model, keys in membership.items():
        rows = list(model.objects.filter(pk__in=keys).order_by("pk").values())
        vectors = [field.attname for field in model._meta.concrete_fields
                   if isinstance(field, VectorField)]
        for row in rows:
            for name in vectors:
                if row[name] is not None:
                    row[name] = tuple(float(value) for value in row[name])
        result[model] = rows
    return result


@pytest.mark.parametrize("queryset", [False, True])
def test_combined_history_uses_one_distinct_row_cap(retained, transport, monkeypatch, queryset):
    from apps.knowledge_graph.graph import invalidation

    doc, old, collection = build_history(retained, transport)
    before = identities(HISTORY)
    assert all(before.values()), "every retained history family must be populated"
    collector = invalidation._BoundedDeleteCollector(using="default", origin=collection)
    collector.collect([collection])
    for model, keys in before.items():
        assert keys <= {row.pk for row in collector.data[model]}
        assert not any(query.model is model for query in collector.fast_deletes)
    count = collector._bounded_source_count
    assert count == sum(len(rows) for rows in collector.data.values())
    collector.collect([collection])
    collector.collect([doc])
    assert collector._bounded_source_count == count
    assert identities(HISTORY) == before
    membership = {model: frozenset(row.pk for row in rows)
                  for model, rows in collector.data.items()}
    original_state = row_state(membership)

    # Standalone immutable rows cannot borrow an unrelated owner's deletion.
    foreign = Collection.objects.create(name="unrelated lifecycle owner")
    foreign.delete()
    for model in (DocumentRepresentationGeneration, LegacyDocumentSourceSnapshot, VocabularyVersion):
        if model is VocabularyVersion:
            with pytest.raises((IntegrityError, ValueError)), transaction.atomic():
                model.objects.filter(pk=next(iter(before[model]))).delete()
        else:
            field = "representation_generation" if model is DocumentRepresentationGeneration else "legacy_source_snapshot"
            with connection.cursor() as cursor:
                constraints = connection.introspection.get_constraints(cursor, TextChunk._meta.db_table)
            names = {name for name, value in constraints.items()
                     if value["columns"] == [TextChunk._meta.get_field(field).column]
                     and value.get("foreign_key") == (model._meta.db_table, model._meta.pk.column)}
            assert len(names) == 1
            with pytest.raises((IntegrityError, InternalError)) as error, transaction.atomic():
                model.objects.filter(pk=next(iter(before[model]))).delete()
            cause = error.value.__cause__
            if isinstance(error.value, IntegrityError):
                assert cause.pgcode == "23503"
                assert cause.diag.constraint_name in names
                assert cause.diag.table_name == TextChunk._meta.db_table
            else:
                assert cause.pgcode == "P0001" and "retained_citation_owner_exists" in str(cause)
        assert row_state(membership) == original_state
    assert identities(HISTORY) == before

    def delete():
        (Collection.objects.filter(pk=collection.pk) if queryset else collection).delete()

    monkeypatch.setattr(invalidation, "_MAX_LIFECYCLE_ROWS", count - 1)
    with pytest.raises(RuntimeError, match="bounded"):
        delete()
    assert Collection.objects.filter(pk=collection.pk).exists()
    assert type(doc).objects.filter(pk=doc.pk).exists()
    assert identities(HISTORY) == before
    assert row_state(membership) == original_state

    monkeypatch.setattr(invalidation, "_MAX_LIFECYCLE_ROWS", count)
    monkeypatch.setattr(invalidation._bounded_values, "__kwdefaults__", {"maximum": count})
    delete()
    assert not Collection.objects.filter(pk=collection.pk).exists()
    assert not TextChunk.objects.filter(pk=old.pk).exists()
    assert all(not model.objects.filter(pk__in=keys).exists() for model, keys in before.items())
