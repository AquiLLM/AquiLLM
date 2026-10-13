"""Verify historical context in SQL and hydrate only the requested text window."""
from dataclasses import dataclass

from django.db.models import F, Func, TextField, Value
from django.db.models.functions import JSONObject, Length, Substr

from apps.documents.models import DocumentRepresentationGeneration, DocumentRepresentationState, LegacyDocumentSourceSnapshot
from .representation_citations import HistoricalSourceUnavailable
from .representation_store import SourceSHA256


SNAPSHOT_FIELDS = (
    "canonical_text", "canonical_sha256", "collection_id_at_capture", "source_artifact_id",
    "source_occurrence_id", "extraction_revision_id", "origin_output_id", "source_binding",
    "media_binding", "source_content_type", "canonical_status", "canonical_failure_code",
)


@dataclass(frozen=True)
class CitationContext:
    text: str
    offset: int


def snapshot_manifest(snapshot):
    return {"version": "legacy-canonical-snapshot-v1", **{name: getattr(snapshot, name) for name in SNAPSHOT_FIELDS}}


def canonical_manifest_digest(expression):
    # Same frozen ASCII-escaping/key-order function used by insertion guards.
    return SourceSHA256(Func(expression, function="representation_json_v1", output_field=TextField()))


def legacy_context_allowed(chunk, document, *, using):
    state = DocumentRepresentationState.objects.using(using).filter(
        model_label=document._meta.label_lower, document_pkid=document.pkid, document_id=document.id,
    ).values("legacy_closed", "legacy_source_hash").first()
    return chunk.legacy_context_status != "unavailable" and (state is None or (
        not state["legacy_closed"] and state["legacy_source_hash"] == document.full_text_hash))


def load_citation_context(chunk, document, *, using, include_text, threshold, padding):
    if not 0 <= chunk.start_position < chunk.end_position:
        raise HistoricalSourceUnavailable("historical_source_unavailable")
    if chunk.representation_generation_id:
        query = DocumentRepresentationGeneration.objects.using(using).filter(
            pk=chunk.representation_generation_id, status__in=("active", "retired"),
        ).annotate(_manifest=canonical_manifest_digest(F("manifest"))).filter(_manifest=F("manifest_sha256"))
        text_field, hash_field = "canonical_text", "canonical_sha256"
    elif chunk.legacy_source_snapshot_id:
        if chunk.legacy_context_status != "verified":
            raise HistoricalSourceUnavailable("historical_source_unavailable")
        manifest = JSONObject(version=Value("legacy-canonical-snapshot-v1"),
                              **{name: F(name) for name in SNAPSHOT_FIELDS})
        query = LegacyDocumentSourceSnapshot.objects.using(using).filter(
            pk=chunk.legacy_source_snapshot_id, canonical_status="verified",
        ).annotate(_manifest=canonical_manifest_digest(manifest)).filter(_manifest=F("manifest_sha256"))
        text_field, hash_field = "canonical_text", "canonical_sha256"
    else:
        if not legacy_context_allowed(chunk, document, using=using):
            raise HistoricalSourceUnavailable("historical_source_unavailable")
        query = type(document)._base_manager.using(using).filter(pk=document.pk, id=document.id,
            full_text_hash=document.full_text_hash)
        text_field, hash_field = "full_text", "full_text_hash"
    query = query.annotate(_hash=SourceSHA256(F(text_field)), _length=Length(text_field),
        _slice=Substr(text_field, chunk.start_position + 1, chunk.end_position - chunk.start_position),
    ).filter(_hash=F(hash_field), _slice=chunk.content, _length__gte=chunk.end_position)
    length = query.values_list("_length", flat=True).first()
    if length is None:
        raise HistoricalSourceUnavailable("historical_source_unavailable")
    if not include_text:
        return CitationContext("", 0)
    start = max(0, chunk.start_position - padding) if length > threshold else 0
    end = min(length, chunk.end_position + padding) if length > threshold else length
    # Repeat integrity predicates for the actual window read to close a raw legacy edit.
    text = query.annotate(_window=Substr(text_field, start + 1, end - start)).values_list("_window", flat=True).first()
    if text is None:
        raise HistoricalSourceUnavailable("historical_source_unavailable")
    return CitationContext(text, start)
