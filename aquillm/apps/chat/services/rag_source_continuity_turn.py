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

_AMBIGUOUS = (
    "I can't identify the referenced source reliably. "
    "Please provide its title or citation."
)
_UNAVAILABLE = (
    "I can no longer access every referenced source in the selected collections. "
    "Please choose an available source or ask a narrower question."
)


async def continuity_candidates(convo, question, *, user, selected_scope, enabled):
    """Return (current public candidate result, user notice), without a new ledger."""
    if not enabled:
        return None, None
    anchors = resolve_source_anchors(question, convo)
    if anchors.unresolved_references:
        return None, _AMBIGUOUS
    if not anchors.chunk_identities:
        return None, None
    runtime = current_source_runtime()
    if runtime is None:
        raise SourcePreparationLimited("continuity requires shared source ledger")
    sources = await database_sync_to_async(
        rehydrate_prior_evidence, thread_sensitive=False
    )(
        anchors,
        user=user,
        selected_scope=selected_scope,
        budget=runtime.budget,
    )
    if len(sources) != len(anchors.chunk_identities):
        return None, _UNAVAILABLE
    result = await database_sync_to_async(
        current_continuity_result, thread_sensitive=False
    )(sources)
    if len(result["result"]) != len(sources):
        return None, LIMITED_MESSAGE
    return result, None


__all__ = ["continuity_candidates"]
