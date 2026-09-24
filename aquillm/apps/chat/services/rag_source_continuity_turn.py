"""Prepare current follow-up candidates for the direct shared selector."""

from channels.db import database_sync_to_async

from apps.chat.services.rag_source_continuity import (
    current_continuity_result,
    rehydrate_prior_evidence,
    resolve_source_anchors,
)
from apps.chat.services.rag_source_synthesis import LIMITED_MESSAGE
from apps.documents.services.source_loading import (
    SourcePreparationLimited,
    current_source_runtime,
)
from lib.llm.turn_context import bounded_retrieval

_AMBIGUOUS = (
    "I can't identify the referenced source reliably. "
    "Please provide its title or citation."
)
_UNAVAILABLE = (
    "I can no longer access every referenced source in the "
    "selected collections. Answer any separately supported current portions, "
    "and state that the unavailable historical evidence cannot be verified. "
    "Do not substitute current evidence for a requested deleted or revoked citation."
)


class ContinuityNotice(str):
    def __new__(cls, text, *, kind="unknown", unavailable=()):
        value = super().__new__(cls, text)
        value.kind = kind
        value.unavailable = unavailable
        return value


def requires_clarification(notice):
    return getattr(notice, "kind", None) == "ambiguous"


async def continuity_candidates(convo, question, *, user, selected_scope, enabled):
    """Return (current public candidate result, user notice), without a new ledger."""
    if not enabled:
        return None, None
    anchors = resolve_source_anchors(question, convo)
    if anchors.unresolved_references:
        return None, ContinuityNotice(_AMBIGUOUS, kind="ambiguous")
    if not anchors.chunk_identities:
        return None, None
    runtime = current_source_runtime()
    if runtime is None:
        raise SourcePreparationLimited("continuity requires shared source ledger")
    sources = await bounded_retrieval(
        database_sync_to_async(rehydrate_prior_evidence, thread_sensitive=False)(
            anchors,
            user=user,
            selected_scope=selected_scope,
            budget=runtime.budget,
        ),
        runtime.budget,
    )
    present = {(s.chunk_id, s.document_id, s.chunk_number) for s in sources}
    missing = tuple(
        identity for identity in anchors.chunk_identities if identity not in present
    )
    notice = (
        ContinuityNotice(
            _UNAVAILABLE
            + " Unavailable historical coordinates: "
            + "; ".join(f"document {doc}, chunk {pk}" for pk, doc, _ in missing),
            kind="unavailable",
            unavailable=missing,
        )
        if missing
        else None
    )
    if not sources:
        return None, notice
    result = await bounded_retrieval(
        database_sync_to_async(current_continuity_result, thread_sensitive=False)(
            sources
        ),
        runtime.budget,
    )
    if len(result["result"]) != len(sources):
        return result if result["result"] else None, LIMITED_MESSAGE
    return result, notice


__all__ = ["continuity_candidates"]
