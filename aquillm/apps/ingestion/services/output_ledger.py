"""Generation/attempt-fenced publication, one atomic document/checkpoint commit."""
from dataclasses import dataclass
from datetime import timedelta
from uuid import UUID, uuid4
from django.core.exceptions import PermissionDenied
from django.db import IntegrityError, transaction
from django.utils import timezone
from apps.documents.services.representation_contracts import DocumentRefV1
from apps.ingestion.models import IngestionBatchItem, IngestionOutput
from .output_status import checkpoint_item, document_ref, exact_document
from .output_materialization import authorized_scope, build_document, verify_materialized
from .output_payloads import issue_code


@dataclass(frozen=True)
class OutputLeaseV1:
    output_id: int
    generation: UUID
    attempt: int


@dataclass(frozen=True)
class OutputResultV1:
    output_key: str
    status: str
    document_ref: DocumentRefV1 | None
    error_code: str | None


def _result(row):
    return OutputResultV1(row.output_key, row.status, document_ref(row), row.error_code or None)


def _fenced(lease, *, using):
    return IngestionOutput.objects.using(using).filter(pk=lease.output_id,
        generation=lease.generation, attempt=lease.attempt, status='processing')


def claim_output(output_key: str, *, using: str) -> OutputLeaseV1 | None:
    observed = IngestionOutput.objects.using(using).filter(output_key=output_key).first()
    if observed is None:
        return None
    with transaction.atomic(using=using):
        item = IngestionBatchItem.objects.using(using).select_for_update().filter(pk=observed.item_id, manifest_complete=True).first()
        if item is None:
            return None
        row = IngestionOutput.objects.using(using).select_for_update().get(pk=observed.pk)
        if row.status in ('success', 'error') or row.next_attempt_at > timezone.now():
            return None
        if row.parent_output_id:
            parent = IngestionOutput.objects.using(using).get(pk=row.parent_output_id)
            if parent.status != 'success':
                if parent.status == 'error':
                    row.status, row.error_code = 'error', 'parent_output_unavailable'
                    row.save(using=using, update_fields=['status', 'error_code'])
                    checkpoint_item(item, using=using)
                return None
        if row.attempt >= 5:
            row.status, row.error_code = 'error', 'retry_exhausted'
            row.save(using=using, update_fields=['status', 'error_code'])
            checkpoint_item(item, using=using)
            return None
        row.status, row.error_code = 'processing', ''
        row.attempt += 1
        row.next_attempt_at = timezone.now() + timedelta(seconds=900)
        row.save(using=using, update_fields=['status', 'error_code', 'attempt', 'next_attempt_at'])
        return OutputLeaseV1(row.pk, row.generation, row.attempt)


def publish_output(lease: OutputLeaseV1, *, using: str) -> OutputResultV1:
    observed = IngestionOutput.objects.using(using).select_related('item__batch__collection', 'parent_output').filter(pk=lease.output_id).first()
    if observed is None:
        return OutputResultV1('', 'error', None, 'source_scope_unavailable')
    if observed.status == 'success':
        document = exact_document(observed, using=using)
        if document is None or not document.collection.user_can_edit(observed.item.batch.user):
            return OutputResultV1(observed.output_key, 'error', None, 'source_scope_unavailable')
        return _result(observed)
    try:
        if not observed.item.batch.collection.user_can_edit(observed.item.batch.user):
            raise PermissionDenied('source_scope_unavailable')
        verify_materialized(observed)
        with authorized_scope(observed, using=using):
            item = IngestionBatchItem.objects.using(using).select_for_update().get(pk=observed.item_id)
            row = _fenced(lease, using=using).select_for_update(of=('self',)).select_related('item__batch__collection', 'parent_output').first()
            if row is None:
                return OutputResultV1(observed.output_key, 'error', None, 'stale_output_lease')
            document = build_document(row, using=using)
            # Document.save inserts its existing chunk-publication outbox here.
            # The outer transaction commits document and output reference together.
            document.save(using=using)
            row.document_model, row.document_pkid = document._meta.label_lower, document.pkid
            row.document_id, row.document_collection_id = document.id, document.collection_id
            row.document_source_hash, row.status, row.error_code = document.full_text_hash, 'success', ''
            row.save(using=using, update_fields=['document_model', 'document_pkid', 'document_id',
                'document_collection_id', 'document_source_hash', 'status', 'error_code'])
            checkpoint_item(item, using=using)
            result = _result(row)
        return result
    except Exception as exc:
        return fail_output(lease, exc, using=using)


def fail_output(lease: OutputLeaseV1, exc: Exception, *, using: str) -> OutputResultV1:
    from lib.embeddings.errors import require_transient
    code, retry = 'output_failed', False
    constraint = getattr(getattr(exc.__cause__, 'diag', None), 'constraint_name', '') or ''
    if isinstance(exc, IntegrityError) and constraint.endswith('_document_collection_unique'):
        code = 'content_hash_collision'
    elif isinstance(exc, PermissionDenied):
        code = issue_code(str(exc), 'source_scope_unavailable')
    else:
        try:
            require_transient(exc)
        except Exception:
            pass
        else:
            retry, code = lease.attempt < 5, 'upstream_unavailable' if lease.attempt < 5 else 'retry_exhausted'
    observed = IngestionOutput.objects.using(using).filter(pk=lease.output_id).first()
    if observed is None:
        return OutputResultV1('', 'error', None, 'source_scope_unavailable')
    with transaction.atomic(using=using):
        item = IngestionBatchItem.objects.using(using).select_for_update().filter(pk=observed.item_id).first()
        row = _fenced(lease, using=using).select_for_update().first()
        if row is None or item is None:
            return OutputResultV1(observed.output_key, 'error', None, 'stale_output_lease')
        row.status, row.error_code = ('queued' if retry else 'error'), code
        row.next_attempt_at = timezone.now() + timedelta(seconds=min(900, 30 * 2 ** min(lease.attempt-1, 5)))
        row.save(using=using, update_fields=['status', 'error_code', 'next_attempt_at'])
        checkpoint_item(item, using=using)
        return _result(row)


def reset_output(output_key, *, generation, extraction_revision_id, using):
    observed = IngestionOutput.objects.using(using).filter(output_key=output_key).first()
    if (observed is None or observed.output_kind == 'registration_error'
            or not observed.payload.get('full_text', '').strip()
            or (observed.output_kind == 'figure' and not observed.parent_output_id)):
        return False
    with authorized_scope(observed, using=using):
        item = IngestionBatchItem.objects.using(using).select_for_update().get(pk=observed.item_id)
        row = IngestionOutput.objects.using(using).select_for_update().filter(pk=observed.pk,
            generation=generation, extraction_revision_id=extraction_revision_id, status='error', document_id__isnull=True).first()
        if row is None:
            return False
        row.generation, row.attempt, row.status, row.error_code = uuid4(), 0, 'queued', ''
        row.next_attempt_at = timezone.now()
        row.save(using=using, update_fields=['generation', 'attempt', 'status', 'error_code', 'next_attempt_at'])
        checkpoint_item(item, using=using)
    return True


def recover_due_outputs(*, limit: int, using: str) -> int:
    from .output_dispatch import recover_due_outputs as recover
    return recover(limit=limit, using=using)
