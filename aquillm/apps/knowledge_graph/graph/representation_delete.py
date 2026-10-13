"""Bound exact representation/dense history before lifecycle deletion mutates rows."""
from collections import defaultdict
from itertools import islice

from django.db.models import Q, QuerySet


def retained_models():
    from apps.documents.models import (
        DocumentRepresentationState, DocumentRepresentationGeneration,
        LegacyDocumentSourceSnapshot, TextChunk,
    )
    from apps.documents.models.embedding_index import DocumentEmbeddingState, EmbeddingGeneration, ChunkEmbedding
    return (DocumentRepresentationState, DocumentRepresentationGeneration,
            LegacyDocumentSourceSnapshot, TextChunk, DocumentEmbeddingState,
            EmbeddingGeneration, ChunkEmbedding)


def must_collect_history(objs):
    from apps.collections.models.vocabulary import (
        VocabularySet, VocabularyDraft, VocabularyVersion, VocabularyEntry, VocabularyAttachment,
    )
    model = getattr(objs, "model", None) or (objs if isinstance(objs, type) else type(objs))
    return model in (*retained_models(), VocabularySet, VocabularyDraft,
                    VocabularyVersion, VocabularyEntry, VocabularyAttachment)


def bounded_new_objects(collector, objs, maximum):
    remaining = maximum - collector._bounded_source_count
    if isinstance(objs, QuerySet):
        seen = collector.data.get(objs.model, ())
        if seen:
            objs = objs.exclude(pk__in=[row.pk for row in seen])
        values = tuple(objs[:remaining + 1])
    else:
        values, seen = [], set()
        for number, obj in enumerate(islice(iter(objs), maximum + 1)):
            if number == maximum:
                raise RuntimeError("delete origin exceeds the bounded lifecycle row limit")
            if obj in seen or obj in collector.data.get(type(obj), ()):
                continue
            seen.add(obj)
            values.append(obj)
            if len(values) > remaining:
                break
    if len(values) > remaining:
        raise RuntimeError("delete origin exceeds the bounded lifecycle row limit")
    return values


def collect_exact_dense_history(collector, documents):
    from apps.documents.models import DESCENDED_FROM_DOCUMENT
    from apps.documents.models.embedding_index import DocumentEmbeddingState
    grouped = defaultdict(list)
    for document in documents:
        if type(document) in DESCENDED_FROM_DOCUMENT:
            grouped[type(document)].append(document)
    for model, owners in grouped.items():
        for offset in range(0, len(owners), 100):
            condition = Q(pk__in=[])
            for owner in owners[offset:offset + 100]:
                condition |= Q(document_pkid=owner.pkid, document_id=owner.id)
            rows = DocumentEmbeddingState.objects.using(collector.using).filter(
                condition, model_label=model._meta.label_lower,
            )
            # A real dependency makes dense rows precede the document signal.
            # Collector.add applies the remaining global allowance before hydration.
            collector.collect(rows, source=model, nullable=False)
            collect_exact_source_chunks(collector, model, owners[offset:offset + 100])


def collect_exact_source_chunks(collector, model, owners):
    """Count retained chunks without traversing protected graph descendants.

    Actual deletion still uses the origin-locked document pre_delete cleanup.
    This Collector is the bounded read-only origin preflight, not a replacement
    deletion path or permission to erase retained rows independently.
    """
    from apps.documents.models import TextChunk
    from apps.documents.models.embedding_index import ChunkEmbedding

    condition = Q(pk__in=[])
    for owner in owners:
        typed = dict(model_label=model._meta.label_lower,
                     document_pkid=owner.pkid, document_id=owner.id)
        generation = Q(**{f"representation_generation__owner__{key}": value
                          for key, value in typed.items()})
        snapshot = Q(**{f"legacy_source_snapshot__owner__{key}": value
                        for key, value in typed.items()})
        condition |= Q(doc_id=owner.id) & (generation | snapshot)
    # add() applies the shared remaining allowance before QuerySet hydration and
    # deduplicates rows already reached from another retained owner edge.
    chunks = collector.add(TextChunk.objects.using(collector.using).filter(condition),
                           source=model, nullable=False)
    for offset in range(0, len(chunks), 100):
        rows = ChunkEmbedding.objects.using(collector.using).filter(
            chunk_id__in=[chunk.pk for chunk in chunks[offset:offset + 100]])
        collector.collect(rows, source=TextChunk, nullable=False)
