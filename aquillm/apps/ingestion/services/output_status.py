"""Public status derives from committed, currently authorized document identities."""
from django.apps import apps
from django.utils import timezone
from apps.documents.services.representation_contracts import DocumentRefV1


def exact_document(output, *, using):
    from apps.documents.models import DESCENDED_FROM_DOCUMENT
    if not output.document_id:
        return None
    try:
        model = apps.get_model(output.document_model)
    except (ValueError, LookupError):
        return None
    if model not in DESCENDED_FROM_DOCUMENT:
        return None
    return model.objects.using(using).filter(pkid=output.document_pkid, id=output.document_id,
        collection_id=output.document_collection_id, full_text_hash=output.document_source_hash).first()


def document_ref(output):
    if not output.document_id:
        return None
    return DocumentRefV1(output.document_model, output.document_pkid, output.document_id,
                         output.document_collection_id, output.document_source_hash)


def output_record(output, *, using, user=None, validate_scope=True):
    payload = output.payload
    document = exact_document(output, using=using) if validate_scope else None
    unavailable = validate_scope and output.document_id and (document is None or
        (user is not None and not document.collection.user_can_view(user)))
    available = bool(document) if validate_scope else bool(output.document_id)
    record = {
        'output_key': output.output_key, 'status': 'error' if unavailable else output.status,
        'error_code': 'source_scope_unavailable' if unavailable else output.error_code or None,
        'issue_codes': payload.get('issue_codes', []), 'index_status': 'unavailable',
        'document_id': str(output.document_id) if available and not unavailable else None,
        'document_model': apps.get_model(output.document_model).__name__ if available and not unavailable else None,
        'title': payload.get('title', '') if not unavailable else '',
        'normalized_type': payload.get('normalized_type', ''), 'modality': payload.get('modality', 'text'),
        'provider': payload.get('provider', ''), 'model': payload.get('model', ''),
        'raw_media_saved': bool(payload.get('media_artifact') or payload.get('normalized_type') == 'pdf'),
        'text_extracted': bool(available and payload.get('full_text')),
    }
    if output.source_occurrence_id and not unavailable:
        record.update(source_occurrence_id=output.source_occurrence_id,
                      source_occurrence_key=output.source_occurrence.occurrence_key,
                      source_artifact_sha256=payload.get('source_artifact', {}).get('sha256'),
                      extraction_revision_id=output.extraction_revision_id,
                      media_artifact_sha256=(payload.get('media_artifact') or {}).get('sha256'))
    if available and not unavailable:
        # Parser success is separate from chunk/dense readiness. Existing legacy
        # chunk completion does not prove a verified active dense generation.
        record['index_status'] = 'chunks_ready' if document and document.ingestion_complete else 'pending'
    return record


def item_snapshot(item, *, using, user=None, validate_scope=True):
    records = [output_record(row, using=using, user=user, validate_scope=validate_scope) for row in
               item.outputs.select_related('source_occurrence').order_by('pk')]
    ids = [r['document_id'] for r in records if r['status'] == 'success' and r['document_id']]
    pending = any(r['status'] in ('queued', 'processing') for r in records)
    degraded = any(r['status'] == 'error' or r['issue_codes'] for r in records)
    state = 'processing' if pending else ('partial' if ids and degraded else 'success' if ids else 'error')
    metadata = {'outputs': records, 'normalized_types': [r['normalized_type'] for r in records],
                'modalities': sorted(set(r['modality'] for r in records))}
    return state, ids, metadata


def checkpoint_item(item, *, using):
    # Internal checkpoint uses committed references in one bounded query. API
    # reads independently revalidate each current document/permission boundary.
    state, ids, metadata = item_snapshot(item, using=using, validate_scope=False)
    item.status, item.document_ids, item.parser_metadata = state, ids, metadata
    item.error_message = '' if state != 'error' else 'ingestion_failed'
    item.failure_code = next((r['error_code'] for r in metadata['outputs'] if r['error_code']), '')
    item.finished_at = None if state == 'processing' else timezone.now()
    item.save(using=using, update_fields=['status', 'document_ids', 'parser_metadata', 'error_message', 'failure_code', 'finished_at'])


def legacy_authorized_ids(item, user, *, using):
    from uuid import UUID
    from apps.documents.models import DESCENDED_FROM_DOCUMENT
    found = []
    for identity in item.document_ids:
        try:
            UUID(str(identity))
        except (TypeError, ValueError, AttributeError):
            continue
        matches = [document for model in DESCENDED_FROM_DOCUMENT for document in
                   model.objects.using(using).filter(id=identity)]
        if len(matches) == 1 and matches[0].collection_id == item.batch.collection_id and matches[0].collection.user_can_view(user):
            found.append(identity)
    return found


def legacy_metadata(item, authorized_ids):
    metadata = item.parser_metadata if isinstance(item.parser_metadata, dict) else {}
    records = metadata.get('outputs', [])
    if not isinstance(records, list):
        records = []
    safe = []
    for record in records:
        if not isinstance(record, dict) or (record.get('document_id') and record['document_id'] not in authorized_ids):
            continue
        safe.append({key: record[key] for key in ('document_id', 'document_model', 'modality',
            'normalized_type', 'provider', 'model', 'raw_media_saved', 'text_extracted') if key in record})
    return {'outputs': safe, 'modalities': sorted({str(r.get('modality') or 'text') for r in safe})}
