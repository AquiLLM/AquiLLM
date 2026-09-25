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
    if not source_mode_enabled():
        return doc.collection.user_can_view(user)
    from apps.documents.services.source_loading import bounded_source_database

    runtime = required_source_runtime()
    with bounded_source_database(runtime, runtime.authorization.database_alias):
        return doc.collection.user_can_view(user)


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
    from apps.collections.services.django_retrieval_authorization import (
        build_selected_scope_authorization_context,
    )
    from apps.documents.services.source_loading import (
        SourceRuntime,
        bounded_source_database,
        source_runtime_scope,
    )

    parent = required_source_runtime()
    alias = getattr(parent.authorization, "database_alias", "default")
    with bounded_source_database(parent, alias):
        doc = document_metadata(doc_id)
        authorization = (
            None
            if doc is None
            else build_selected_scope_authorization_context(
                principal=user,
                selected_collection_ids=(doc.collection_id,),
                selected_documents=(doc,),
                database_alias=alias,
            )
        )
    if authorization is None:
        raise SourcePreparationLimited("requested document is not authorized")
    child = SourceRuntime(
        parent.budget,
        authorization,
        parent.cache,
        parent.windows,
        parent.lock,
        parent.observation,
    )
    with source_runtime_scope(child):
        yield child


def selected_document_metadata(user, col_ref):
    from apps.collections.models import Collection

    if not source_mode_enabled():
        return Collection.get_user_accessible_documents(
            user, Collection.objects.filter(id__in=col_ref.collections)
        )
    from apps.documents.models.document import _get_document_types
    from apps.documents.services.source_loading import bounded_source_database

    runtime = required_source_runtime()
    alias = getattr(runtime.authorization, "database_alias", "default")
    with bounded_source_database(runtime, alias):
        collections = (
            Collection.objects.using(alias)
            .filter(id__in=col_ref.collections)
            .filter_by_user_perm(user, "VIEW")
        )
        return [
            doc
            for model in _get_document_types()
            for doc in model.objects.using(alias)
            .filter(collection__in=collections)
            .defer(*deferred_document_bodies(model))
        ]


def document_metadata(doc_id):
    from apps.documents.models import Document

    if not source_mode_enabled():
        return Document.get_by_id(doc_id)
    from apps.documents.models.document import _get_document_types
    from apps.documents.services.source_loading import bounded_source_database

    runtime = required_source_runtime()
    alias = getattr(runtime.authorization, "database_alias", "default")
    with bounded_source_database(runtime, alias):
        for model in _get_document_types():
            doc = (
                model.objects.using(alias)
                .filter(id=doc_id)
                .defer(*deferred_document_bodies(model))
                .first()
            )
            if doc is not None:
                return doc
    return None


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
