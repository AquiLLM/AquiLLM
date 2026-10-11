"""Persisted generation selection and lazy SQL eligibility; history never falls back."""
from django.db.models import Exists, F, Q

from apps.documents.models import DocumentRepresentationGeneration, DocumentRepresentationState
from .representation_contracts import RepresentationSelectionV1, SourceScopeUnavailable


def _states(ref, using):
    return DocumentRepresentationState.objects.using(using).filter(
        model_label=ref.model_label, document_pkid=ref.pkid, document_id=ref.id,
    )


def current_document_chunks(document):
    from apps.documents.models import TextChunk
    from .representation_history import reference
    from .representation_store import active_chunk_scope, freeze_active_representations
    rows = TextChunk.objects.using(document._state.db or "default").filter(doc_id=document.id)
    try:
        snapshot = freeze_active_representations((reference(document),), using=rows.db)
        return active_chunk_scope(rows, snapshot=snapshot)
    except SourceScopeUnavailable:
        return rows.none()


def selection_for_document(ref, *, using):
    state = _states(ref, using).values("active_generation_id", "legacy_closed", "legacy_source_hash",
        "active_generation__canonical_sha256", "active_generation__source_revision_sha256").first()
    if state and state["active_generation_id"]:
        return RepresentationSelectionV1(ref, state["active_generation_id"],
            state["active_generation__canonical_sha256"], state["active_generation__source_revision_sha256"])
    if state and (state["legacy_closed"] or state["legacy_source_hash"] != ref.source_hash):
        raise SourceScopeUnavailable("source_scope_unavailable")
    return RepresentationSelectionV1(ref, None, ref.source_hash, None)


def validate_selection_state(selection, *, using):
    ref = selection.document
    state = _states(ref, using).first()
    if selection.generation_key is None:
        if state and (state.legacy_closed or state.active_generation_id or state.legacy_source_hash != ref.source_hash):
            raise SourceScopeUnavailable("source_scope_unavailable")
    elif state is None or state.active_generation_id != selection.generation_key or not (
        state.collection_id == ref.collection_id and state.legacy_closed and
        DocumentRepresentationGeneration.objects.using(using).filter(
            pk=selection.generation_key, owner=state, status="active", collection_id=ref.collection_id,
            canonical_sha256=ref.source_hash, source_revision_sha256=selection.source_revision_sha256,
        ).exists()
    ):
        raise SourceScopeUnavailable("source_scope_unavailable")


def current_state_clause(selection, *, using):
    ref = selection.document
    states = _states(ref, using)
    if selection.generation_key is None:
        invalid = states.filter(Q(legacy_closed=True) | Q(active_generation__isnull=False) |
                                ~Q(legacy_source_hash=ref.source_hash))
        return ~Q(Exists(invalid)) & ~Q(legacy_context_status="unavailable")
    valid = states.filter(
        active_generation_id=selection.generation_key, collection_id=ref.collection_id, legacy_closed=True,
        active_generation__status="active", active_generation__owner_id=F("pk"),
        active_generation__collection_id=ref.collection_id, active_generation__canonical_sha256=ref.source_hash,
        active_generation__source_revision_sha256=selection.source_revision_sha256,
    )
    return Q(Exists(valid))
