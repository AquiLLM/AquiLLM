"""Synchronous authorized preparation for one direct-RAG selection stage."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Any

import structlog
from channels.db import database_sync_to_async

from apps.chat.services.rag_config import (
    EvidenceSelectionConfig,
    evidence_token_budget,
    max_snippets_per_doc,
)
from apps.chat.services.rag_evidence import build_selected_evidence_packet
from apps.chat.services.rag_query import _is_retry
from apps.chat.services.rag_retrieval import (
    FusedRetrievalPool,
    fuse_ranked_tool_results,
)
from apps.chat.services.rag_selection import select_evidence
from apps.chat.services.rag_selection_hydration import revalidate_selection_candidates
from apps.chat.services.rag_selection_policy import choose_selection_profile
from apps.chat.services.rag_selection_scoring import (
    PreparedSelection,
    prepare_selection_candidates,
)
from apps.chat.services.rag_selection_types import EvidenceSelection, SelectionLimits
from apps.chat.services.retrieval_authorization import (
    resolve_document_retrieval_authorization,
)
from apps.collections.models import Collection
from lib.llm.types.messages import UserMessage

logger = structlog.stdlib.get_logger(__name__)


@dataclass(frozen=True)
class SelectionTurn:
    selection: EvidenceSelection
    authorization: object
    prepared: PreparedSelection
    selection_duration_ms: float
    candidate_count: int


def selection_question(convo, latest_text: str) -> str:
    """Use the user's original task, not a retrieved-title query prefix."""
    if not _is_retry(latest_text):
        return latest_text.strip()
    for message in reversed(convo.messages[:-1]):
        if isinstance(message, UserMessage) and not _is_retry(message.content or ""):
            return (message.content or "").strip()
    return latest_text.strip()


def selected_tool_result(packet) -> dict:
    """Persist only final public evidence rows with matching public counts."""
    titles = sorted(
        {
            title
            for row in packet.chunks
            if isinstance(title := row.get("title", row.get("n")), str) and title
        }
    )
    result = {
        "result": packet.chunks,
        "retrieval_status": packet.retrieval_status,
        "retrieved_count": len(packet.chunks),
    }
    if titles:
        result["retrieved_documents"] = titles
    if not packet.chunks:
        result["retrieval_message"] = packet.diagnostic_message
    return result


def selection_metric_fields(config, turn: SelectionTurn | None, packet) -> dict:
    """Closed aggregate values only; no passages, identities, or score arrays."""
    base = {"selection_mode": config.mode, "selection_config_error": config.error}
    if turn is None:
        return base
    proposed = turn.selection.candidates
    return base | {
        "selector_ms": turn.selection_duration_ms,
        "final_scoring_ms": turn.prepared.scoring_duration_ms,
        "candidate_count": turn.candidate_count,
        "selected_doc_count": len(
            {row.get("doc_id", row.get("d")) for row in packet.chunks}
        ),
        "estimated_tokens": packet.total_tokens,
        "reused_pairs": turn.prepared.reused_pairs,
        "new_pairs": turn.prepared.new_pairs,
        "profile_version": turn.selection.profile.version,
        "fixed_fallback_reason": turn.prepared.fallback_reason,
        "proposed_selected_count": len(proposed),
        "proposed_selected_doc_count": len({item.doc_id for item in proposed}),
        "proposed_estimated_tokens": turn.selection.estimated_tokens,
        "proposed_profile_name": turn.selection.profile.name,
        "proposed_score_status": turn.selection.score_status,
    }


async def coordinate_selection(
    consumer,
    search_results,
    query,
    question,
    config,
    top_k,
    *,
    prepare_fn,
    revalidate_fn,
):
    """Run opt-in selection on DB workers; shadow never changes served evidence."""
    try:
        pool = fuse_ranked_tool_results(search_results)
        prepare_async = database_sync_to_async(prepare_fn, thread_sensitive=False)
        turn = await prepare_async(consumer, pool, query, question, config, top_k)
        if config.mode == "shadow":
            return turn, None, None
        revalidate_async = database_sync_to_async(revalidate_fn, thread_sensitive=False)
        selection = await revalidate_async(turn.selection, turn.authorization)
        packet = build_selected_evidence_packet(
            selection,
            query=query,
            search_scope="selected documents",
        )
        return turn, packet, selected_tool_result(packet)
    except Exception:
        if config.mode == "adaptive":
            raise
        logger.warning("direct_rag_shadow_selection_failed")
        return None, None, None


def prepare_selection_turn(
    consumer: Any,
    pool: FusedRetrievalPool,
    primary_query: str,
    question: str,
    config: EvidenceSelectionConfig,
    top_k: int,
) -> SelectionTurn:
    """Resolve current scope, score once, and select on a DB-capable worker."""
    docs = Collection.get_user_accessible_documents(
        consumer.user,
        Collection.objects.filter(id__in=consumer.col_ref.collections),
    )
    authorization = resolve_document_retrieval_authorization(
        consumer.user,
        consumer.col_ref,
        docs,
        None,
    )
    if authorization is None:
        raise ValueError("retrieval authorization unavailable")
    started = monotonic()
    prepared = prepare_selection_candidates(
        pool=pool,
        primary_query=primary_query,
        authorization=authorization,
        deadline=started + config.score_timeout_ms / 1000.0,
        allow_new_scores=config.mode == "adaptive" or config.shadow_scoring,
    )
    profile = choose_selection_profile(question)
    select_started = monotonic()
    selection = select_evidence(
        prepared.candidates,
        profile=profile,
        limits=SelectionLimits(top_k, max_snippets_per_doc(), evidence_token_budget()),
        score_status=prepared.score_status,
    )
    return SelectionTurn(
        selection,
        authorization,
        prepared,
        max(0.0, (monotonic() - select_started) * 1000.0),
        len(prepared.candidates),
    )


def revalidate_selection_turn(
    selection: EvidenceSelection,
    authorization: object,
) -> EvidenceSelection:
    """Final current-source check; never refill dropped selector choices."""
    surviving = revalidate_selection_candidates(selection.candidates, authorization)
    return EvidenceSelection(
        surviving,
        sum(max(1, (len(item.text) + 3) // 4) for item in surviving),
        selection.profile,
        selection.score_status,
    )


__all__ = [
    "SelectionTurn",
    "prepare_selection_turn",
    "revalidate_selection_turn",
    "selection_question",
    "selected_tool_result",
    "selection_metric_fields",
    "coordinate_selection",
]
