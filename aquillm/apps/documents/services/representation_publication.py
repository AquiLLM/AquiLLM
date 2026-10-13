"""The original chunk-outbox delivery remains fenced through representation commit."""
from django.utils import timezone
from dataclasses import dataclass

from apps.documents.models import ChunkPublication
from .representation_contracts import SourceScopeUnavailable


def lock_publication(document, publication, *, using):
    row = ChunkPublication.objects.using(using).select_for_update().filter(
        pk=publication.pk, generation=publication.generation, attempts=publication.attempts,
        concrete_model_label=document.model_label, document_pkid=document.pkid, document_id=document.id,
        source_hash=document.source_hash, next_attempt_at__gt=timezone.now(),
    ).first()
    if row is None or row.failure_kind:
        raise SourceScopeUnavailable("stale_chunk_publication")
    return row


def publication_fields(document, publication, *, using):
    if publication is None:
        return {}
    row = lock_publication(document, publication, using=using)
    return dict(publication_id=row.pk, publication_generation=row.generation, publication_attempt=row.attempts)


def generation_publication(document, generation, *, using):
    if generation.publication_id is None:
        return None
    from types import SimpleNamespace
    return lock_publication(document, SimpleNamespace(pk=generation.publication_id,
        generation=generation.publication_generation, attempts=generation.publication_attempt), using=using)


def resume_publication_generation(state, publication, manifest, *, using):
    """Caller holds source/state locks; duplicate deliveries share this attempt."""
    if publication is None:
        return None
    from apps.documents.models import DocumentRepresentationGeneration
    from .representation_inputs import manifest_hash
    rows = list(DocumentRepresentationGeneration.objects.using(using).select_for_update().filter(
        owner=state, publication_id=publication.pk, publication_generation=publication.generation,
        publication_attempt=publication.attempts).order_by("key")[:2])
    if not rows:
        return None
    if len(rows) != 1:
        raise SourceScopeUnavailable("ambiguous_representation_delivery")
    row = rows[0]
    from lib.parsers.canonical_projection import canonical_json
    if (row.status not in ("building", "staged") or row.lease_expires_at <= timezone.now()
        or row.predecessor_generation_key != state.active_generation_id
        or row.manifest_sha256 != manifest_hash(manifest) or canonical_json(row.manifest) != canonical_json(manifest)):
        raise SourceScopeUnavailable("stale_representation")
    return row


@dataclass(frozen=True)
class _LegacySuccessAck:
    document: object
    pk: int
    generation: object
    attempt: int


def capture_legacy_success_ack(document, *, using):
    """Success-only snapshot at task entry; never used for failure accounting."""
    rows = list(ChunkPublication.objects.using(using).filter(
        concrete_model_label=document.model_label, document_pkid=document.pkid,
        document_id=document.id, source_hash=document.source_hash)[:2])
    if len(rows) != 1:
        return None
    row = rows[0]
    return _LegacySuccessAck(document, row.pk, row.generation, row.attempts)


def acknowledge_legacy_success(document, acknowledgement, *, using):
    """Only called under the successful publication's source/state locks."""
    if acknowledgement is None:
        return
    if type(acknowledgement) is not _LegacySuccessAck or acknowledgement.document != document:
        raise SourceScopeUnavailable("legacy_acknowledgement_changed")
    row = ChunkPublication.objects.using(using).select_for_update().filter(
        pk=acknowledgement.pk, generation=acknowledgement.generation, attempts=acknowledgement.attempt,
        concrete_model_label=document.model_label, document_pkid=document.pkid,
        document_id=document.id, source_hash=document.source_hash).first()
    if row is not None:
        row.delete(using=using)
