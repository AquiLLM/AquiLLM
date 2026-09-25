"""Protect a frozen source packet across every provider call and recovery."""

import asyncio
import json

from apps.chat.services.rag_context_budget import provider_context_capacity
from lib.llm.evidence_guard import ContextLimited, EvidenceProtection, protect_evidence

LIMITED_MESSAGE = (
    "I could not deliver the selected evidence within this request's context or "
    "retrieval limits. Please narrow the question or selected documents. "
    "This limit does not mean the documents contain no supporting information."
)


async def complete_source_request(llm, request, budget, stream_func, *, packet=None):
    from apps.documents.services.source_loading import current_source_runtime

    runtime = current_source_runtime()
    if runtime is None:
        raise ContextLimited("source synthesis requires shared ledger")
    context, margin = provider_context_capacity(llm)
    if context <= 0:
        raise ContextLimited("unknown_model_context")
    from lib.llm.synthesis_dispatch import synthesis_limits
    from lib.retrieval.synthesis_budget import seal_synthesis

    try:
        identities = tuple(
            (
                "chunk",
                p.source.document_id,
                p.source.chunk_id,
                p.source.source_fingerprint,
                tuple((s.start, s.end) for s in p.spans),
            )
            for p in (packet.source_evidence if packet else ())
        )
        if packet and packet.selection and not packet.source_evidence:
            from lib.retrieval.evidence import fingerprint_source

            identities += tuple(
                (
                    "legacy_chunk",
                    c.doc_id,
                    c.chunk_id,
                    c.source_fingerprint,
                    fingerprint_source(c.text),
                )
                for c in packet.selection.candidates
            )
        if packet and packet.auxiliary_handoff:
            identities += tuple(
                ("figure", json.dumps(record, sort_keys=True))
                for record in packet.auxiliary_handoff[0].get("_figure_provenance", ())
            )
        lease = seal_synthesis(
            runtime.budget,
            request[-1].content,
            **synthesis_limits(budget),
            authority=packet.source_authorization if packet else runtime.authorization,
            identities=identities,
        )
    except ValueError as exc:
        raise ContextLimited(str(exc)) from exc
    protection = EvidenceProtection(
        request[-1].content,
        context,
        budget,
        margin,
        turn_budget=runtime.budget,
        synthesis_lease=lease,
    )
    with protect_evidence(protection):

        async def guarded_stream(payload):
            lease.check_active()
            return await stream_func(payload)

        async with asyncio.timeout(lease.remaining_seconds()):
            completed, status = await llm.complete(
                request, budget, stream_func=guarded_stream if stream_func else None
            )
        lease.check_active()
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
    from apps.chat.services.rag_selection_hydration import (
        revalidate_selection_candidates,
    )
    from apps.chat.services.rag_selection_types import EvidenceSelection
    from apps.chat.services.rag_source_hydration import revalidate_source_candidates
    from lib.llm.turn_context import bounded_retrieval

    surviving = await bounded_retrieval(
        database_sync_to_async(
            revalidate_source_candidates
            if packet.source_mode
            else revalidate_selection_candidates,
            thread_sensitive=False,
        )(packet.selection.candidates, packet.source_authorization),
        current_source_runtime().budget,
    )
    selection = EvidenceSelection(
        surviving,
        sum(c.token_cost or 0 for c in surviving),
        packet.selection.profile,
        packet.selection.score_status,
    )
    fresh = build_selected_evidence_packet(
        selection, query=packet.query, search_scope=packet.search_scope
    )
    fresh.source_mode = packet.source_mode
    fresh.source_authorization = packet.source_authorization
    fresh.selection = selection
    fresh.coverage_assessment = packet.coverage_assessment
    fresh.auxiliary_handoff = packet.auxiliary_handoff
    fresh.diagnostic_message = packet.diagnostic_message
    if packet.retrieval_status == "partial":
        fresh.retrieval_status = "partial"
    if not fresh.chunks:
        fresh.retrieval_status = "context_limited"
    return fresh


async def finalize_source_support(packet):
    """Last delivered-span coverage and auxiliary authority checks before sealing."""
    from channels.db import database_sync_to_async

    from apps.chat.services.rag_coverage import recheck_support
    from apps.documents.services.source_loading import current_source_runtime
    from lib.llm.turn_context import bounded_retrieval

    runtime = current_source_runtime()
    if packet.auxiliary_handoff:
        from apps.chat.services.tool_wiring.source_tool_revalidation import (
            revalidate_source_tool_result,
        )

        raw, user = packet.auxiliary_handoff
        fresh = await bounded_retrieval(
            database_sync_to_async(
                revalidate_source_tool_result, thread_sensitive=False
            )(raw, user=user),
            runtime.budget,
        )
        packet.auxiliary_handoff = (fresh, user)
        payload = fresh.get("result")
        if isinstance(payload, dict):
            packet.auxiliary_figures = tuple(payload.get("figures", ()))
            packet.image_urls.extend(
                f["image_url"] for f in packet.auxiliary_figures if f.get("image_url")
            )
        if fresh.get("retrieval_status") in {"partial", "context_limited"}:
            packet.retrieval_status = "partial" if packet.chunks else "context_limited"
            packet.diagnostic_message = fresh.get("retrieval_message", LIMITED_MESSAGE)
    if packet.coverage_assessment:
        packet.coverage_assessment = recheck_support(
            packet.coverage_assessment, packet.source_evidence
        )
        if packet.coverage_assessment.unresolved_aspects:
            packet.retrieval_status = "partial" if packet.chunks else "context_limited"
    if runtime:
        runtime.observation["delivered_packet"] = packet
    return packet
