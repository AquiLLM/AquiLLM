"""Recoverable chunk dispatch; source commit and publication intent are atomic.

The due time leases publication, not document ownership. A crash before/after
broker acknowledgement is retried after the lease. The existing chunk task
uses lifecycle locks and source hashes to make such redelivery idempotent.
"""
from datetime import timedelta

import structlog
from django.apps import apps
from django.db import DEFAULT_DB_ALIAS, transaction
from django.utils import timezone

from apps.documents.models.chunk_publication import ChunkPublication

logger = structlog.stdlib.get_logger(__name__)
PUBLICATION_LEASE_SECONDS = 900


def schedule_chunk_publication(document, collection_ids, *, using=DEFAULT_DB_ALIAS):
    """Call inside the document transaction, before it can become visible."""
    identity = {
        'concrete_model_label': document._meta.label_lower,
        'document_pkid': document.pkid,
        'document_id': document.id,
    }
    rows = ChunkPublication.objects.using(using)
    rows.filter(**identity).exclude(source_hash=document.full_text_hash).delete()
    intent, _ = rows.get_or_create(
        **identity, source_hash=document.full_text_hash,
        defaults={'collection_ids': sorted(set(collection_ids))},
    )
    transaction.on_commit(
        lambda: dispatch_chunk_publication(intent.pk, using=using),
        using=using, robust=True,
    )


def acknowledge_chunk_publication(document, source_hash, *, using=DEFAULT_DB_ALIAS):
    ChunkPublication.objects.using(using).filter(
        concrete_model_label=document._meta.label_lower,
        document_pkid=document.pkid, document_id=document.id,
        source_hash=source_hash,
    ).delete()


def dispatch_chunk_publication(intent_id, *, using=DEFAULT_DB_ALIAS):
    from apps.documents.models import DESCENDED_FROM_DOCUMENT
    from apps.documents.tasks.chunking import create_chunks
    from apps.knowledge_graph.graph.invalidation import (
        DocumentLifecycleRef, prepare_document_chunk_replacement,
        schedule_collection_graph_refreshes,
    )

    now = timezone.now()
    # Do not hold this row lock while acquiring document/graph locks or while
    # talking to the broker: document saves acquire those locks first.
    with transaction.atomic(using=using):
        intent = (ChunkPublication.objects.using(using).select_for_update(skip_locked=True)
                  .filter(pk=intent_id).first())
        if intent is None:
            return False
        try:
            model = apps.get_model(intent.concrete_model_label)
        except (LookupError, ValueError):
            model = None
        document = None if model not in DESCENDED_FROM_DOCUMENT else (
            model._base_manager.using(using).filter(
                pkid=intent.document_pkid, id=intent.document_id,
                full_text_hash=intent.source_hash,
            ).first()
        )
        if document is None or document.ingestion_complete:
            intent.delete(using=using)
            return False
        if intent.next_attempt_at > now:
            return False
        intent.attempts += 1
        intent.next_attempt_at = now + timedelta(seconds=PUBLICATION_LEASE_SECONDS)
        intent.last_error = ''
        intent.save(update_fields=['attempts', 'next_attempt_at', 'last_error'], using=using)

    try:
        # This also recovers a process death before the old post-commit graph
        # cleanup. Its committed-chunk fast path never destroys a peer's result.
        affected = prepare_document_chunk_replacement(
            DocumentLifecycleRef(
                concrete_model_label=intent.concrete_model_label,
                document_pkid=intent.document_pkid, document_id=intent.document_id,
            ),
            intent.collection_ids, expected_source_hash=intent.source_hash, using=using,
        )
        if not affected:
            ChunkPublication.objects.using(using).filter(pk=intent.pk).delete()
            return False
        schedule_collection_graph_refreshes(
            [pk for pk in affected if pk != document.collection_id], using=using,
        )
        create_chunks.delay(
            str(intent.document_id), intent.source_hash,
            intent.concrete_model_label, intent.document_pkid,
        )
        return True
    except Exception as exc:
        # Compare the attempt so a slow publisher cannot undo a newer lease.
        ChunkPublication.objects.using(using).filter(pk=intent.pk, attempts=intent.attempts).update(
            next_attempt_at=timezone.now() + timedelta(seconds=min(900, 30 * 2 ** min(intent.attempts - 1, 5))),
            last_error=type(exc).__name__[:128],
        )
        logger.warning('obs.documents.chunk_publication_retry',
                       document_id=str(intent.document_id), error_type=type(exc).__name__)
        return False


def recover_due_chunk_publications(*, limit=25, using=DEFAULT_DB_ALIAS):
    limit = min(100, max(1, int(limit)))
    ids = list(ChunkPublication.objects.using(using)
               .filter(next_attempt_at__lte=timezone.now())
               .order_by('next_attempt_at', 'pk').values_list('pk', flat=True)[:limit])
    return sum(dispatch_chunk_publication(pk, using=using) for pk in ids)
