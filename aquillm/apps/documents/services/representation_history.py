"""Capture old canonical context without manufacturing historical file identity."""
from django.contrib.contenttypes.models import ContentType

from apps.documents.models import DocumentRepresentationState, LegacyDocumentSourceSnapshot, TextChunk
from apps.ingestion.models import IngestionOutput
from .representation_inputs import content_hash, manifest_hash
from .representation_contracts import DocumentRefV1, SourceScopeUnavailable


def reference(document):
    return DocumentRefV1(document._meta.label_lower, document.pkid, document.id,
                         document.collection_id, document.full_text_hash)


def locked_state(document, *, using):
    content_type = ContentType.objects.db_manager(using).get_for_model(document, for_concrete_model=True)
    state, _ = DocumentRepresentationState.objects.using(using).get_or_create(
        model_label=document._meta.label_lower, document_pkid=document.pkid, document_id=document.id,
        defaults={"owner_content_type": content_type, "collection_id": document.collection_id,
                  "legacy_source_hash": document.full_text_hash},
    )
    state = DocumentRepresentationState.objects.using(using).select_for_update().get(pk=state.pk)
    validate_locked_state(state, reference(document), using=using)
    return state


def validate_locked_state(state, document, *, using):
    content_type = ContentType.objects.db_manager(using).get_by_natural_key(*document.model_label.split("."))
    if (state.model_label, state.document_pkid, state.document_id, state.owner_content_type_id,
        state.collection_id) != (document.model_label, document.pkid, document.id, content_type.pk,
                                 document.collection_id):
        raise SourceScopeUnavailable("representation_owner_changed")


def capture_source_snapshot(document, state, *, using):
    """Capture the locked current canonical identity, including subsequent edits."""
    return _snapshot(document, state, using=using)


def capture_legacy(document, state, *, using):
    """Caller holds lifecycle source/state locks. Immutable ledger metadata only."""
    if state.legacy_closed:
        return None
    snapshot = _snapshot(document, state, using=using, expected_hash=state.legacy_source_hash)
    _attach_legacy(document, snapshot, using=using)
    return snapshot


def _snapshot(document, state, *, using, expected_hash=None):
    expected_hash = expected_hash or document.full_text_hash
    origins = list(IngestionOutput.objects.using(using).filter(
        status="success", document_model=document._meta.label_lower, document_pkid=document.pkid,
        document_id=document.id, document_source_hash=document.full_text_hash,
    ).select_related("source_occurrence", "extraction_revision")[:2])
    source_matches = expected_hash == document.full_text_hash
    origin = origins[0] if source_matches and len(origins) == 1 else None
    verified = source_matches and content_hash(document.full_text) == document.full_text_hash
    fields = {
        "canonical_text": document.full_text if verified else None,
        "canonical_sha256": document.full_text_hash if verified else None,
        "collection_id_at_capture": document.collection_id,
        "source_artifact_id": origin.source_occurrence.artifact_id if origin else None,
        "source_occurrence_id": origin.source_occurrence_id if origin else None,
        "extraction_revision_id": origin.extraction_revision_id if origin else None,
        "origin_output_id": origin.pk if origin else None,
        "source_binding": origin.payload["source_artifact"] if origin else None,
        "media_binding": origin.payload.get("media_artifact") if origin else None,
        "source_content_type": origin.payload.get("source_content_type", "") if origin else "",
        "canonical_status": "verified" if verified else "unavailable",
        "canonical_failure_code": "" if verified else (
            "canonical_hash_mismatch" if source_matches else "canonical_source_unavailable"),
    }
    snapshot, _ = LegacyDocumentSourceSnapshot.objects.using(using).get_or_create(
        owner=state, captured_source_hash=expected_hash,
        manifest_sha256=manifest_hash({"version": "legacy-canonical-snapshot-v1", **fields}), defaults=fields,
    )
    return snapshot


def _attach_legacy(document, snapshot, *, using):
    rows = TextChunk.objects.using(using).filter(doc_id=document.id, representation_generation__isnull=True,
                                                legacy_source_snapshot__isnull=True)
    last = 0
    while True:
        batch = list(rows.select_for_update().filter(pk__gt=last).order_by("pk")[:100])
        if not batch:
            break
        for chunk in batch:
            reason = snapshot.canonical_failure_code
            if not reason:
                if not 0 <= chunk.start_position < chunk.end_position <= len(document.full_text):
                    reason = "invalid_offsets"
                elif document.full_text[chunk.start_position:chunk.end_position] != chunk.content:
                    reason = "slice_mismatch"
            artifact_status = "verified" if snapshot.origin_output_id else "unavailable"
            artifact_reason = "" if snapshot.origin_output_id else "artifact_unknown"
            if reason:
                artifact_status, artifact_reason = "unavailable", "historical_context_unavailable"
            TextChunk.objects.using(using).filter(pk=chunk.pk).update(
                legacy_source_snapshot=snapshot, legacy_context_status="unavailable" if reason else "verified",
                legacy_context_failure_code=reason, legacy_artifact_status=artifact_status,
                legacy_artifact_failure_code=artifact_reason,
            )
        last = batch[-1].pk
    return snapshot


def exact_chunk_document(chunk, *, using, include_full_text=True):
    from apps.documents.models import DESCENDED_FROM_DOCUMENT
    from django.core.exceptions import ValidationError
    owner = None
    if chunk.representation_generation_id:
        owner = DocumentRepresentationState.objects.using(using).filter(
            generations__pk=chunk.representation_generation_id)
    elif chunk.legacy_source_snapshot_id:
        owner = DocumentRepresentationState.objects.using(using).filter(
            source_snapshots__pk=chunk.legacy_source_snapshot_id)
    if owner is not None:
        owner = owner.values_list("model_label", "document_pkid", "document_id").get()
    rows = []
    for model in DESCENDED_FROM_DOCUMENT:
        query = model._base_manager.using(using).filter(id=chunk.doc_id)
        if not include_full_text:
            bodies = {"full_text", "extracted_caption"} & {f.name for f in model._meta.fields}
            query = query.defer(*bodies)
        rows.extend(query[:2])
    if len(rows) != 1:
        raise ValidationError("Exact citation owner unavailable")
    doc = rows[0]
    if owner and (doc._meta.label_lower, doc.pkid, doc.id) != owner:
        raise ValidationError("Exact citation owner unavailable")
    return doc


def delete_legacy_remainder(document, *, using):
    from apps.documents.models import DESCENDED_FROM_DOCUMENT
    if any(model._base_manager.using(using).filter(id=document.id).exists() for model in DESCENDED_FROM_DOCUMENT):
        raise SourceScopeUnavailable("ambiguous_deleted_document")
    TextChunk.objects.using(using).filter(doc_id=document.id, representation_generation__isnull=True,
                                         legacy_source_snapshot__isnull=True).delete()
