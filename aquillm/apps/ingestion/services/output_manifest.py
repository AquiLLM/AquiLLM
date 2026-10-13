"""Freeze the whole output set before publishing any member."""
from datetime import timedelta
from hashlib import sha256
from django.db import transaction
from django.utils import timezone
from apps.ingestion.models import IngestionBatchItem, IngestionOutput
from .output_payloads import MAX_OUTPUTS, bind_projection, issue_code, output_key


def prepare_output_manifest(item_id: int, *, using: str) -> tuple[str, ...]:
    from aquillm.ingestion.parsers import extract_text_payloads
    from .output_status import checkpoint_item
    rows = IngestionBatchItem.objects.using(using)
    with transaction.atomic(using=using):
        item = rows.select_for_update().filter(pk=item_id).first()
        if item is None:
            return ()
        if item.manifest_complete:
            return tuple(item.outputs.order_by('pk').values_list('output_key', flat=True))
        if item.status in ('success', 'partial', 'error'):
            return ()  # Historical terminal work has no fabricated provenance.
        if item.status == 'processing' and item.next_attempt_at > timezone.now():
            return ()
        if item.attempts >= 5:
            row = IngestionOutput.objects.using(using).create(item=item, **_failure(item, 0, 'retry_exhausted'))
            item.manifest_complete = True
            item.save(using=using, update_fields=['manifest_complete'])
            checkpoint_item(item, using=using)
            return (row.output_key,)
        item.status, item.started_at = 'processing', timezone.now()
        item.attempts += 1
        item.next_attempt_at = timezone.now() + timedelta(seconds=900)
        item.save(using=using, update_fields=['status', 'started_at', 'attempts', 'next_attempt_at'])
        fence = (item.generation, item.attempts)
    item = rows.select_related('batch__collection', 'batch__user').get(pk=item_id)
    prepared = []
    try:
        if not item.batch.collection.user_can_edit(item.batch.user):
            raise ValueError('source_scope_unavailable')
        with item.source_file.open('rb') as source:
            data = source.read(50 * 1024 * 1024 + 1)
        if len(data) > 50 * 1024 * 1024:
            raise ValueError('source_size_limit')
        payloads = extract_text_payloads(item.original_filename, data, content_type=item.content_type or None)
        for index, payload in enumerate(payloads[:MAX_OUTPUTS]):
            try:
                kind, element, projected, failure = bind_projection(payload, item, data, using=using)
                occurrence = payload.source_occurrence
                prepared.append(dict(output_key=output_key(item.pk, occurrence.occurrence_key,
                    payload.extraction.parser_key, kind, element), source_occurrence_id=occurrence.occurrence_id,
                    extraction_revision_id=payload.extraction_revision_id, output_kind=kind,
                    element_key=element, payload=projected, status='error' if failure else 'queued', error_code=failure))
            except Exception as exc:
                from lib.embeddings.errors import require_transient
                try:
                    require_transient(exc)
                except Exception:
                    pass
                else:
                    raise
                prepared.append(_failure(item, index, issue_code(str(exc))))
        if len(payloads) > MAX_OUTPUTS:
            prepared.append(_failure(item, MAX_OUTPUTS, 'output_limit'))
        if not prepared:
            prepared.append(_failure(item, 0, 'empty_extraction'))
    except Exception as exc:
        from lib.embeddings.errors import require_transient
        try:
            require_transient(exc)
        except Exception:
            prepared = [_failure(item, 0, issue_code(str(exc)))]
        else:
            if fence[1] < 5:
                rows.filter(pk=item_id, generation=fence[0], attempts=fence[1]).update(
                    status='queued', next_attempt_at=timezone.now()+timedelta(seconds=30 * 2 ** (fence[1]-1)))
                return ()
            prepared = [_failure(item, 0, 'retry_exhausted')]
    with transaction.atomic(using=using):
        current = rows.select_for_update().filter(pk=item_id, generation=fence[0], attempts=fence[1], manifest_complete=False).first()
        if current is None:
            return ()
        parents, keys = {}, []
        # Parent before figure, independent of extractor ordering.
        for fields in sorted(prepared, key=lambda f: f['output_kind'] == 'figure'):
            if fields['output_key'] in keys:
                fields = _failure(item, len(keys), 'duplicate_output_identity')
            source = (fields.get('source_occurrence_id'), fields.get('extraction_revision_id'))
            parent = parents.get(source) if fields['output_kind'] == 'figure' else None
            if fields['output_kind'] == 'figure' and parent is None:
                fields.update(status='error', error_code='parent_output_unavailable')
            output = IngestionOutput.objects.using(using).create(item=current, parent_output=parent, **fields)
            keys.append(output.output_key)
            if output.output_kind not in ('figure', 'registration_error'):
                parents[(output.source_occurrence_id, output.extraction_revision_id)] = output
        current.manifest_complete = True
        current.save(using=using, update_fields=['manifest_complete'])
        checkpoint_item(current, using=using)
        return tuple(keys)


def _failure(item, index, code):
    return dict(output_key=output_key(item.pk, sha256(item.source_file.name.encode()).hexdigest(),
                'registration-v1', 'registration_error', str(index)),
                output_kind='registration_error', element_key=str(index), payload={}, status='error', error_code=code)
