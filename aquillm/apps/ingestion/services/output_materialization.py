"""Construct documents from frozen manifest payloads without provider I/O."""
from contextlib import contextmanager
from django.core.exceptions import PermissionDenied
from apps.collections.models import Collection, CollectionPermission
from lib.parsers.source_contracts import SourceArtifactRefV1
from .source_artifacts import _verify_blob
from .output_status import exact_document


def verify_materialized(output):
    from .output_payloads import MAX_TEXT
    payload = output.payload
    expected = {'title', 'full_text', 'normalized_type', 'modality', 'provider', 'model',
                'issue_codes', 'quality', 'source_artifact', 'media_artifact', 'source_content_type',
                'source_format', 'figure_index', 'extracted_caption', 'location_metadata',
                'collection_id', 'collection_parent_id'}
    if set(payload) != expected or type(payload['full_text']) is not str or len(payload['full_text']) > MAX_TEXT:
        raise ValueError('invalid_output_payload')
    if not payload['full_text'].strip():
        raise ValueError('empty_extraction')
    occurrence, revision = output.source_occurrence, output.extraction_revision
    artifact = occurrence.artifact
    expected_source = SourceArtifactRefV1(artifact.sha256, artifact.size_bytes, artifact.immutable_storage_key,
        occurrence.member_path, occurrence.container.sha256 if occurrence.container_id else None)
    if SourceArtifactRefV1(**payload['source_artifact']) != expected_source or revision.occurrence_id != occurrence.pk:
        raise ValueError('source_payload_integrity_failed')
    if output.output_kind == 'text' and payload['full_text'] != revision.canonical_text:
        raise ValueError('source_payload_integrity_failed')
    for name in ('source_artifact', 'media_artifact'):
        ref = output.payload.get(name)
        if ref:
            _verify_blob(SourceArtifactRefV1(**ref))


@contextmanager
def authorized_scope(output, *, using):
    from apps.knowledge_graph.graph.invalidation import locked_collection_parent_rows
    collection_id = output.item.batch.collection_id
    chain, current = [], collection_id
    while current is not None and len(chain) < 100:
        row = Collection.objects.using(using).filter(pk=current).values('pk', 'parent_id').first()
        if row is None:
            raise PermissionDenied('source_scope_unavailable')
        chain.append((row['pk'], row['parent_id']))
        current = row['parent_id']
    if current is not None:
        raise PermissionDenied('source_scope_unavailable')
    with locked_collection_parent_rows(collection_id, [pk for pk, _ in chain], using=using):
        actual = dict(Collection.objects.using(using).filter(pk__in=[pk for pk, _ in chain]).values_list('pk', 'parent_id'))
        if actual != dict(chain) or actual.get(collection_id) != output.payload['collection_parent_id']:
            raise PermissionDenied('source_scope_unavailable')
        grants = list(CollectionPermission.objects.using(using).select_for_update().filter(
            user_id=output.item.batch.user_id, collection_id__in=actual, permission__in=['EDIT', 'MANAGE']))
        if not grants or output.payload['collection_id'] != collection_id:
            raise PermissionDenied('source_scope_unavailable')
        yield


def build_document(output, *, using):
    from apps.documents.models import RawTextDocument, PDFDocument, ImageUploadDocument, MediaUploadDocument, DocumentFigure
    from aquillm.task_ingest_helpers import figures_subcollection_name
    payload, item = output.payload, output.item
    common = dict(title=payload['title'], full_text=payload['full_text'],
                  collection_id=item.batch.collection_id, ingested_by_id=item.batch.user_id)
    media = payload.get('media_artifact')
    if output.output_kind == 'figure':
        parent = exact_document(output.parent_output, using=using) if output.parent_output_id else None
        if parent is None or parent.collection_id != item.batch.collection_id:
            raise PermissionDenied('parent_output_unavailable')
        # Inherit current parent permission instead of copying grants that could
        # outlive revocation. The parent collection lock serializes creation.
        collection, _ = Collection.objects.using(using).get_or_create(
            parent_id=item.batch.collection_id, name=figures_subcollection_name(parent.title))
        # FOR UPDATE also fences FK grant inserts/repoints through commit. An
        # existing direct grant is an independent reader boundary, not inherited
        # source permission; leave it intact and refuse this destination.
        collection = Collection.objects.using(using).select_for_update().get(pk=collection.pk)
        if (collection.parent_id != item.batch.collection_id
                or CollectionPermission.objects.using(using).filter(collection=collection).exists()):
            raise PermissionDenied('source_scope_unavailable')
        common['collection_id'] = collection.pk
        doc = DocumentFigure(**common, source_format=payload['source_format'], figure_index=payload['figure_index'],
            extracted_caption=payload['extracted_caption'], location_metadata=payload['location_metadata'])
        doc.parent_document = parent
    elif output.output_kind == 'image':
        doc = ImageUploadDocument(**common)
    elif output.output_kind in ('audio', 'video'):
        doc = MediaUploadDocument(**common, media_kind=output.output_kind,
            transcribe_provider=payload['provider'], transcribe_model=payload['model'])
        doc.media_file = media['storage_key'] if media else ''
    elif payload['normalized_type'] == 'pdf':
        doc = PDFDocument(**common, pdf_file=payload['source_artifact']['storage_key'])
    else:
        doc = RawTextDocument(**common)
    if output.output_kind in ('figure', 'image'):
        doc.image_file = media['storage_key'] if media else ''
        doc.ocr_provider, doc.ocr_model = payload['provider'], payload['model']
    if hasattr(doc, 'source_content_type'):
        doc.source_content_type = payload['source_content_type']
    return doc
