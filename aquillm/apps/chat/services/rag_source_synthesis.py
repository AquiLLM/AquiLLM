"""Protect a frozen source packet across every provider call and recovery."""

from apps.chat.services.rag_context_budget import provider_context_capacity
from lib.llm.evidence_guard import ContextLimited, EvidenceProtection, protect_evidence

LIMITED_MESSAGE = (
    "I could not deliver the selected evidence within this request's context or "
    "retrieval limits. Please narrow the question or selected documents. "
    "This limit does not mean the documents contain no supporting information."
)


async def complete_source_request(llm, request, budget, stream_func):
    from apps.documents.services.source_loading import current_source_runtime

    runtime = current_source_runtime()
    if runtime is None:
        raise ContextLimited("source synthesis requires shared ledger")
    context, margin = provider_context_capacity(llm)
    if context <= 0:
        raise ContextLimited("unknown_model_context")
    protection = EvidenceProtection(
        request[-1].content,
        context,
        budget,
        margin,
        turn_budget=runtime.budget,
    )
    with protect_evidence(protection):
        completed, status = await llm.complete(request, budget, stream_func=stream_func)
        # Continuation/repair helpers can catch ordinary provider exceptions.
        # A swallowed guard failure still produces the explicit limited outcome.
        if protection.limited_reason:
            raise ContextLimited(protection.limited_reason)
        return completed, status


async def revalidate_source_packet(packet):
    """Last current authorization/revision check before provider handoff."""
    from apps.documents.services.source_loading import (
        SourcePreparationLimited,
        current_source_runtime,
    )

    if (
        packet.source_authorization is None
        or packet.selection is None
        or current_source_runtime() is None
    ):
        raise SourcePreparationLimited("source packet has no current authority")
    from channels.db import database_sync_to_async

    from apps.chat.services.rag_evidence import build_selected_evidence_packet
    from apps.chat.services.rag_selection_types import EvidenceSelection
    from apps.chat.services.rag_source_hydration import revalidate_source_candidates

    surviving = await database_sync_to_async(
        revalidate_source_candidates, thread_sensitive=False
    )(packet.selection.candidates, packet.source_authorization)
    selection = EvidenceSelection(
        surviving,
        sum(c.token_cost or 0 for c in surviving),
        packet.selection.profile,
        packet.selection.score_status,
    )
    fresh = build_selected_evidence_packet(
        selection, query=packet.query, search_scope=packet.search_scope
    )
    fresh.source_mode = True
    fresh.source_authorization = packet.source_authorization
    fresh.selection = selection
    if not fresh.chunks:
        fresh.retrieval_status = "context_limited"
    return fresh
