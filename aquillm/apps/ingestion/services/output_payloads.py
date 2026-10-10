"""Bounded durable payload projection; all byte/provider work precedes row locks."""
from dataclasses import asdict
from hashlib import sha256
from aquillm.task_ingest_helpers import sanitize_db_text
from lib.parsers.source_contracts import ExtractionV1, SourceElementV1, source_artifact_ref
from .extraction_revisions import bind_payload_extraction

MAX_TEXT = 1_000_000
MAX_OUTPUTS = 1000
PUBLIC_CODES = frozenset('''extraction_failed extraction_issue structure_unavailable
    formula_cache_unavailable figure_layout_unavailable spreadsheet_chart_structure_unavailable
    xls_formula_identity_unavailable document_wrapper_structure_unavailable
    auxiliary_document_content_unavailable slide_shape_structure_unavailable layout_unavailable
    source_format_unsupported local_parser_unavailable source_structure_limit_exceeded
    malformed_source_structure invalid_extraction_structure chunker_work_limit
    source_artifact_required invalid_output_text invalid_output_kind figure_identity_missing
    empty_extraction unsupported_member archive_member_limit archive_byte_limit member_extraction_failed
    source_size_limit output_limit duplicate_output_identity source_scope_unavailable
    parent_output_unavailable source_payload_integrity_failed source_artifact_integrity_failed
    extraction_payload_text_mismatch extraction_artifact_mismatch extraction_revision_binding_mismatch
    extraction_occurrence_mismatch extraction_revision_registration_mismatch'''.split())


def issue_code(value, fallback='extraction_failed'):
    return value if type(value) is str and value in PUBLIC_CODES else fallback


def bind_projection(payload, item, data, *, using):
    if payload.source_artifact is None:
        # Legacy top-level test/integration producers can bind exact upload bytes.
        if payload.archive_lineage or item.original_filename.lower().endswith('.zip'):
            raise ValueError('source_artifact_required')
        payload.source_artifact = source_artifact_ref(data, member_path=None, container_sha256=None)
        payload.source_bytes, payload.source_filename = data, item.original_filename
        payload.source_content_type = item.content_type or 'application/octet-stream'
    if payload.extraction is None:
        text = payload.full_text or ''
        elements = (SourceElementV1('body', 'paragraph', 0, len(text), None, None, None, None),) if text else ()
        payload.extraction = ExtractionV1('existing-' + payload.normalized_type + '-text-v1',
            payload.source_artifact, text, elements, 'partial', ('structure_unavailable',))
    bind_payload_extraction(payload, using=using)
    text = payload.full_text or ''
    if len(text) > MAX_TEXT or '\x00' in text:
        raise ValueError('invalid_output_text')
    meta = payload.metadata if type(payload.metadata) is dict else {}
    kind = 'figure' if payload.normalized_type == 'document_figure' else payload.modality
    if kind not in ('text', 'image', 'audio', 'video', 'figure'):
        raise ValueError('invalid_output_kind')
    element = 'body'
    if kind == 'figure':
        index = meta.get('figure_index')
        if type(index) is not int or not 0 <= index < MAX_OUTPUTS:
            raise ValueError('figure_identity_missing')
        element = f'figure:{index}'
    codes = [issue_code(v) for v in payload.extraction.issue_codes[:32]]
    failure = issue_code(meta['output_error']) if meta.get('output_error') else ''
    if not text.strip():
        failure = failure or 'empty_extraction'
    projected = {
        'title': sanitize_db_text(payload.title or item.original_filename)[:200],
        'full_text': text, 'normalized_type': str(payload.normalized_type)[:32],
        'modality': payload.modality, 'provider': str(payload.provider or '')[:64],
        'model': str(payload.model or '')[:200], 'issue_codes': codes,
        'quality': payload.extraction.quality,
        'source_artifact': asdict(payload.source_artifact),
        'media_artifact': asdict(payload.media_artifact) if payload.media_artifact else None,
        'source_content_type': str(payload.media_content_type or payload.source_content_type or '')[:150],
        'source_format': str(meta.get('source_format', 'unknown'))[:20],
        'figure_index': meta.get('figure_index', 0),
        'extracted_caption': sanitize_db_text(meta.get('extracted_caption', ''))[:10000],
        'location_metadata': {k: v[:256] if type(v) is str else v for k, v in (meta.get('location_metadata') or {}).items()
                              if k in ('page_number', 'slide_number', 'sheet_name') and type(v) in (str, int)},
        'collection_id': item.batch.collection_id,
        'collection_parent_id': item.batch.collection.parent_id,
    }
    return kind, element, projected, failure


def output_key(item_id, occurrence_key, parser_key, kind, element):
    from lib.parsers.canonical_projection import canonical_json
    return sha256(canonical_json(['ingestion-output-v1', item_id, occurrence_key, parser_key, kind, element])).hexdigest()
