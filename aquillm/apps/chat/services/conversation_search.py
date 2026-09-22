"""Hybrid vector + trigram retrieval over a user's ConversationChunk rows.

Scoped equivalent of apps.documents.services.chunk_search.text_chunk_search:
the candidate generation (vector L2 + trigram + exact terms) is reproduced for a
user-owned conversation-chunk queryset, and the documents reranker is reused
verbatim (it is generic over the chunk model class).
"""
from __future__ import annotations

import structlog
from typing import Optional

from django.apps import apps
from django.conf import settings as django_settings
from django.contrib.auth.models import User
from django.contrib.postgres.search import TrigramSimilarity
from pgvector.django import L2Distance

from apps.chat.models import ConversationChunk
from apps.documents.services.chunk_rerank import _fallback_rerank, rerank_chunks
from apps.documents.services.chunk_search import _exact_term_query, _salient_exact_terms

logger = structlog.stdlib.get_logger(__name__)


def search_conversation_chunks(
    user: User,
    query: str,
    top_k: int,
    exclude_conversation_id: Optional[int] = None,
) -> list[ConversationChunk]:
    """Return up to ``top_k`` ConversationChunks from this user's other chats.

    Chunks are restricted to conversations the user owns; ``exclude_conversation_id``
    drops the current thread so the model doesn't "recall" the conversation it is in.
    """
    qstrip = (query or "").strip()
    if not qstrip:
        return []

    base = ConversationChunk.objects.filter(conversation__owner=user)
    if exclude_conversation_id is not None:
        base = base.exclude(conversation_id=exclude_conversation_id)

    vector_top_k = apps.get_app_config("aquillm").vector_top_k  # type: ignore
    trigram_top_k = apps.get_app_config("aquillm").trigram_top_k  # type: ignore
    mult = float(getattr(django_settings, "RAG_CANDIDATE_MULTIPLIER", 3.0))
    raw_cap = int(top_k * mult)
    vector_limit = max(top_k + 2, min(vector_top_k, raw_cap))
    trigram_limit = max(top_k + 2, min(trigram_top_k, raw_cap))
    exact_limit = max(top_k + 2, min(trigram_top_k, raw_cap))
    tri_sim_min = float(getattr(django_settings, "RAG_TRIGRAM_SIMILARITY_MIN", 0.000001))

    try:
        query_embedding = None
        try:
            from aquillm.utils import get_embedding

            query_embedding = get_embedding(query)
        except Exception as exc:
            logger.warning(
                "Conversation vector embed failed; trigram-only retrieval. Error: %s", exc
            )

        if query_embedding is not None:
            vector_results = list(
                base.exclude(embedding__isnull=True)
                .defer("embedding")
                .order_by(L2Distance("embedding", query_embedding))[:vector_limit]
            )
        else:
            vector_results = []

        trigram_results = list(
            base.annotate(similarity=TrigramSimilarity("content", query))
            .filter(similarity__gt=tri_sim_min)
            .order_by("-similarity")[:trigram_limit]
        )

        exact_terms = _salient_exact_terms(query)
        if exact_terms:
            exact_results = list(
                base.filter(_exact_term_query(exact_terms)).order_by(
                    "conversation", "chunk_number"
                )[:exact_limit]
            )
        else:
            exact_results = []

        combined = vector_results + trigram_results + exact_results
        deduped = []
        seen_pks = set()
        for candidate in combined:
            if candidate.pk in seen_pks:
                continue
            seen_pks.add(candidate.pk)
            deduped.append(candidate)

        if not deduped:
            return []
        if len(deduped) <= top_k:
            return list(_fallback_rerank(ConversationChunk, deduped, top_k))
        return list(rerank_chunks(ConversationChunk, query, deduped, top_k))
    except Exception as exc:
        logger.error("Error during conversation chunk search: %s", exc)
        raise


__all__ = ["search_conversation_chunks"]
