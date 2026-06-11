"""Evidence-first synthesis for the direct RAG pipeline.

Given a conversation whose last turn is a (synthetic) tool result, produce a
user-facing assistant answer by reusing the standard post-tool synthesis
machinery (:meth:`LLMInterface.complete`) so citation enforcement, figure
embedding, and retrieval-notice handling stay consistent with the tool loop.
"""
from __future__ import annotations

from typing import Any

import structlog

from lib.llm.types.conversation import Conversation

from apps.chat.services.rag_config import synthesis_max_tokens
from apps.chat.services.rag_evidence import EvidencePacket

logger = structlog.stdlib.get_logger(__name__)


async def synthesize_from_evidence(
    llm_if: Any,
    convo: Conversation,
    packet: EvidencePacket,
    *,
    stream_func: Any = None,
    max_tokens: int | None = None,
) -> Conversation:
    """Produce the final assistant turn from packaged evidence."""
    budget = max_tokens if max_tokens is not None else synthesis_max_tokens()
    result_convo, _ = await llm_if.complete(convo, budget, stream_func=stream_func)
    return result_convo


__all__ = ["synthesize_from_evidence"]
