"""Rehydrate authorized current sources and build continuity candidates."""

from __future__ import annotations

from uuid import UUID

from apps.chat.services.rag_source_anchor_resolution import (
    SourceAnchors,
    resolve_source_anchors,
)
from apps.documents.models.chunks import TextChunk
from apps.documents.services.source_loading import (
    SourcePreparationLimited,
    bounded_source_database,
    current_source_runtime,
    source_query_rows,
)
from lib.retrieval.evidence import SourceEvidence, fingerprint_source


def rehydrate_prior_evidence(
    anchors: SourceAnchors, *, user, selected_scope, budget
) -> tuple[SourceEvidence, ...]:
    """Use the injected source runtime and its charged metadata-first loader."""
    from apps.collections.services.retrieval_authorization import (
        revalidate_retrieval_authorization_context,
    )

    runtime = current_source_runtime()
    if runtime is None or budget is not runtime.budget:
        raise SourcePreparationLimited("continuity requires shared source ledger")
    if user is not runtime.authorization.reauthorization_capability._principal:
        raise SourcePreparationLimited("continuity principal differs from source scope")
    if (
        frozenset(int(value) for value in selected_scope)
        != runtime.authorization.selected_collection_ids
    ):
        raise SourcePreparationLimited("continuity requires current selected scope")
    with bounded_source_database(runtime, runtime.authorization.database_alias):
        scope = revalidate_retrieval_authorization_context(
            context=runtime.authorization
        )
    allowed = frozenset(scope.document_ids)
    wanted = []
    for identity in anchors.chunk_identities:
        try:
            document_id = UUID(identity[1])
        except ValueError:
            continue  # Keep the anchor so the existing unavailable notice survives.
        if document_id in allowed:
            wanted.append(identity)
    if not wanted:
        return ()
    chunks = source_query_rows(
        TextChunk.objects.using(runtime.authorization.database_alias).filter(
            pk__in=[identity[0] for identity in wanted]
        ),
        runtime=runtime,
    )
    current = {chunk.pk: chunk for chunk in chunks}
    return tuple(
        SourceEvidence(
            pk, doc, number, fingerprint_source(chunk.content), chunk.content
        )
        for pk, doc, number in wanted
        if (chunk := current.get(pk)) is not None
        and str(chunk.doc_id) == doc
        and chunk.chunk_number == number
    )


def current_continuity_result(sources: tuple[SourceEvidence, ...]) -> dict:
    from apps.chat.services.tool_wiring.source_documents import document_metadata
    from lib.retrieval.evidence import source_provenance

    rows = []
    titles = {}
    for source in sources:
        if source.document_id not in titles:
            document = document_metadata(source.document_id)
            titles[source.document_id] = (
                getattr(document, "title", None) if document is not None else None
            )
        title = titles[source.document_id]
        if not isinstance(title, str) or not title:
            continue
        rows.append(
            {
                "rank": len(rows) + 1,
                "doc_id": source.document_id,
                "chunk_id": source.chunk_id,
                "chunk": source.chunk_number,
                "title": title,
                "text": source.text,
                "citation": f"[doc:{source.document_id} chunk:{source.chunk_id}]",
            }
        )
    return {
        "result": rows,
        "retrieval_status": "results_found" if rows else "no_results",
        "retrieved_count": len(rows),
        "_source_provenance": [source_provenance(source) for source in sources],
    }


__all__ = [
    "SourceAnchors",
    "resolve_source_anchors",
    "rehydrate_prior_evidence",
    "current_continuity_result",
]
