"""Source-mode document tool hooks; callers keep one shared turn ledger."""

from contextlib import contextmanager

from apps.documents.services.source_loading import (
    SourcePreparationLimited,
    current_source_runtime,
)
from apps.documents.services.source_loading import (
    bounded_source_enabled as source_mode_enabled,
)
from lib.retrieval.evidence import SourceEvidence, fingerprint_source


def can_view_document(user, doc):
    from apps.documents.services.representation_media_acquisition import media_runtime
    public = current_source_runtime()
    runtime = public or media_runtime()
    if runtime is None and not source_mode_enabled():
        return doc.collection.user_can_view(user)
    from apps.documents.services.source_loading import bounded_source_database
    runtime = runtime or required_source_runtime()
    from apps.documents.services.representation_media_acquisition import _check_runtime
    if public is None:
        _check_runtime(runtime)
    with bounded_source_database(runtime, runtime.authorization.database_alias):
        from apps.documents.services.representation_tool_scope import tool_scope
        scope = tool_scope(runtime, using=runtime.authorization.database_alias)
        exact = (doc._meta.label_lower, doc.pk, doc.id, doc.collection_id, doc.full_text_hash)
        selected = any(exact == (s.document.model_label, s.document.pkid, s.document.id,
            s.document.collection_id, s.document.source_hash) for s in scope.representation.selections)
        return selected and doc.collection.user_can_view(user)


def required_source_runtime():
    runtime = current_source_runtime()
    if runtime is None:
        raise SourcePreparationLimited("source tool requires shared turn ledger")
    return runtime


def deferred_document_bodies(model):
    return [
        field.name
        for field in model._meta.concrete_fields
        if field.name in {"full_text", "extracted_caption"}
    ]


@contextmanager
def explicit_document_source_scope(user, doc_id):
    """Only an explicit manual caller may narrow a child scope to this exact ID.

    Automatic planners must retain the original selected scope. The child's
    lifetime is one manual operation; all reservations stay on the outer ledger.
    """
    from apps.documents.services.representation_manual_scope import create_manual_document_runtime
    from apps.documents.services.source_loading import source_runtime_scope
    from apps.documents.services.representation_media_acquisition import media_runtime
    child = create_manual_document_runtime(user, doc_id, parent=media_runtime())
    with source_runtime_scope(child):
        yield child


def selected_document_metadata(user, col_ref):
    from apps.collections.models import Collection
    from .current_document_metadata import acquisition_metadata
    documents = acquisition_metadata()
    if documents is not None:
        return documents
    return Collection.get_user_accessible_documents(
        user, Collection.objects.filter(id__in=col_ref.collections), metadata_only=True)


def document_metadata(doc_id):
    from apps.documents.models import Document
    from .current_document_metadata import acquisition_metadata
    documents = acquisition_metadata(doc_id)
    if documents is not None:
        return documents[0] if documents else None
    return Document.get_by_id(doc_id)


def document_source_evidence(results):
    if not source_mode_enabled():
        return ()
    if current_source_runtime() is None:
        raise SourcePreparationLimited("source tool requires shared turn ledger")
    return tuple(
        SourceEvidence(
            row.pk,
            str(row.doc_id),
            row.chunk_number,
            fingerprint_source(row.content),
            row.content,
        )
        for row in results
    )


async def prepare_source_tool_handoff(
    consumer,
    llm,
    conversation,
    tool_result,
    *,
    question,
    top_k,
    prepare_fn=None,
    revalidate_fn=None,
):
    """Same served selection for model tools and direct acquisition.

    Task5 binds this after acquisition and before normal-loop synthesis, using
    its existing outer runtime. No fallback or tool may create a fresh ledger.
    """
    from apps.chat.services.rag_config import evidence_selection_config
    from apps.chat.services.rag_selection_coordinator import (
        coordinate_selection,
        prepare_selection_turn,
        revalidate_selection_turn,
    )

    return await coordinate_selection(
        consumer,
        list(tool_result) if isinstance(tool_result, (tuple, list)) else [tool_result],
        question,
        question,
        evidence_selection_config(),
        top_k,
        prepare_fn=prepare_fn or prepare_selection_turn,
        revalidate_fn=revalidate_fn or revalidate_selection_turn,
        request_conversation=conversation,
        llm_if=llm,
    )
