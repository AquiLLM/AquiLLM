"""Atomic upload intent and bounded, keyset-based broker recovery."""
from datetime import timedelta
from uuid import uuid4
from django.db import transaction
from django.utils import timezone
from apps.ingestion.models import IngestionBatchItem, IngestionOutput


def dispatch_item(item_id, *, using='default'):
    from aquillm.tasks import ingest_uploaded_file_task
    with transaction.atomic(using=using):
        item = IngestionBatchItem.objects.using(using).select_for_update(skip_locked=True).filter(
            pk=item_id, manifest_complete=False, status__in=['queued', 'processing'], next_attempt_at__lte=timezone.now()).first()
        if item is None:
            return False
        if item.status == 'processing':
            item.generation = uuid4()  # Fence an expired parser before broker redelivery.
            item.status = 'queued'
        item.next_attempt_at = timezone.now() + timedelta(seconds=900)
        item.save(using=using, update_fields=['next_attempt_at', 'status', 'generation'])
        generation, attempt = item.generation, item.attempts
    try:
        ingest_uploaded_file_task.delay(item_id)
        return True
    except Exception:
        IngestionBatchItem.objects.using(using).filter(pk=item_id, generation=generation, attempts=attempt,
            manifest_complete=False).update(next_attempt_at=timezone.now()+timedelta(seconds=30))
        return False


def recover_due_outputs(*, limit: int = 25, using: str = 'default') -> int:
    from apps.ingestion.tasks.output_recovery import publish_ingestion_output
    from .output_ledger import claim_output, fail_output
    limit = min(100, max(1, int(limit)))
    total, scanned = 0, 0
    for kind in ('outputs', 'items'):
        after = 0
        while scanned < limit:
            if kind == 'outputs':
                query = IngestionOutput.objects.using(using).filter(status__in=['queued', 'processing'], item__manifest_complete=True)
            else:
                query = IngestionBatchItem.objects.using(using).filter(manifest_complete=False, status__in=['queued', 'processing'])
            rows = list(query.filter(pk__gt=after, next_attempt_at__lte=timezone.now()).order_by('pk')[:min(25, limit-scanned)])
            if not rows:
                break
            for row in rows:
                scanned += 1
                after = row.pk
                if kind == 'items':
                    total += dispatch_item(row.pk, using=using)
                    continue
                lease = claim_output(row.output_key, using=using)
                if lease is None:
                    continue
                try:
                    publish_ingestion_output.delay(lease.output_id, str(lease.generation), lease.attempt, using=using)
                    total += 1
                except Exception:
                    fail_output(lease, ConnectionError('broker_unavailable'), using=using)
    return total
