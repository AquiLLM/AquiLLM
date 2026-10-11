"""Strict structural publication; vector builds retain their independent I4 lifetime."""
import structlog

from django.conf import settings
from django.utils import timezone

from apps.documents.models import DocumentRepresentationGeneration
from apps.ingestion.models import IngestionOutput
from apps.documents.services.representation_activation import _begin, activate_representation, stage_representation
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.representation_history import reference
from apps.documents.services.representation_inputs import (
    configured_chunker_key, extraction_input, representation_chunks,
)
from apps.documents.services.representation_store import freeze_active_representations
from lib.parsers.source_contracts import SourceContractError

logger = structlog.stdlib.get_logger(__name__)


def _prepare(document, *, publication, using):
    owners = list(IngestionOutput.objects.using(using).filter(
        document_model=document.model_label, document_pkid=document.pkid,
        document_id=document.id).order_by("pk")[:2])
    if not owners:
        from apps.documents.services.representation_snapshot_build import begin_canonical_snapshot_representation
        return begin_canonical_snapshot_representation(document, publication=publication, using=using)
    if len(owners) != 1 or owners[0].status != "success":
        raise SourceScopeUnavailable("durable_output_unavailable")
    origin = owners[0]
    origin, revision, extraction = extraction_input(document, origin.extraction_revision_id, using=using)
    from apps.documents.services.representation_caption import caption_chunker_key
    key = configured_chunker_key() if origin.output_kind == "text" else caption_chunker_key()
    try:
        chunks = representation_chunks(origin, extraction, key)
    except SourceContractError as exc:
        if str(exc) != "chunker_work_limit":
            raise
        _begin(document, origin, revision, extraction, (), key, publication=publication, using=using,
               failure_code="chunker_work_limit")
        return None
    lease = _begin(document, origin, revision, extraction, chunks, key, publication=publication, using=using)
    predecessor = DocumentRepresentationGeneration.objects.using(using).values_list(
        "predecessor_generation_key", flat=True).get(pk=lease.generation_key)
    return lease, chunks, predecessor


def _fail(lease, document, *, using):
    """No new lease or authority: terminal bookkeeping under original owner locks."""
    from apps.documents.services.representation_activation import _lifecycle, _lease_valid
    from apps.documents.services.representation_history import locked_state
    from apps.documents.services.representation_store import validate_document_reference
    from apps.knowledge_graph.graph.invalidation import locked_document_lifecycle_row
    with locked_document_lifecycle_row(_lifecycle(document), (document.collection_id,), using=using) as (row, _):
        validate_document_reference(document, using=using)
        state = locked_state(row, using=using)
        generation = DocumentRepresentationGeneration.objects.using(using).select_for_update().get(pk=lease.generation_key)
        if (generation.owner_id != state.pk or not _lease_valid(generation, lease)
            or generation.status not in ("building", "staged")):
            return
        generation.status, generation.failure_code = "failed", "publication_failed"
        generation.save(using=using, update_fields=["status", "failure_code"])


def publish_document_representation(document, *, publication=None, legacy_success_ack=None, using="default"):
    ref = reference(document)
    if document.ingestion_complete and publication is None:
        from apps.documents.services.representation_replay import replay_registered_representation
        outcome = replay_registered_representation(document, legacy_success_ack, using=using)
        if outcome is not None:
            return outcome
    prepared = _prepare(ref, publication=publication, using=using)
    if prepared is None:
        return "failed"
    lease, chunks, predecessor = prepared
    try:
        stage_representation(lease, chunks=chunks, using=using)
        result = activate_representation(lease, expected_active_key=predecessor, using=using,
                                         legacy_success_ack=legacy_success_ack)
    except Exception:
        try:
            _fail(lease, ref, using=using)
        except Exception:
            pass  # Retain the original error; expiry/moves grant no new bookkeeping lease.
        raise
    if result == "activated":
        _queue_dense(document, expected_key=lease.generation_key, using=using)
    return result


def _queue_dense(document, *, expected_key, using):
    """Dense unavailability never rolls back published lexical source coverage."""
    key = getattr(settings, "DOCUMENT_EMBEDDING_SPACE_KEY", None)
    if not key:
        return
    try:
        from apps.documents.tasks.embedding_index import rebuild_document_embeddings
        document.refresh_from_db(using=using)
        selected = freeze_active_representations((reference(document),), using=using).selections[0]
        if selected.generation_key != expected_key:
            return
        rebuild_document_embeddings(selected, space_key=key, using=using)
    except Exception as exc:
        logger.warning("obs.documents.representation_dense_unavailable", error_type=type(exc).__name__)
