"""Hybrid vector + trigram chunk retrieval with reranking."""
from __future__ import annotations

import logging
from time import perf_counter
from typing import TYPE_CHECKING, List, Type

from django.apps import apps
from django.contrib.postgres.search import TrigramSimilarity
from django.core.exceptions import ValidationError
from django.db import DatabaseError
from pgvector.django import L2Distance

from apps.documents.services.chunk_rerank import _fallback_rerank, rerank_chunks

if TYPE_CHECKING:
    from apps.documents.models.chunks import TextChunk

logger = logging.getLogger(__name__)


def text_chunk_search(model_cls: Type[TextChunk], query: str, top_k: int, docs: List):
    from aquillm.utils import get_embedding

    vector_top_k = apps.get_app_config("aquillm").vector_top_k  # type: ignore
    trigram_top_k = apps.get_app_config("aquillm").trigram_top_k  # type: ignore
    candidate_multiplier = 3
    vector_limit = max(top_k + 2, min(vector_top_k, top_k * candidate_multiplier))
    trigram_limit = max(top_k + 2, min(trigram_top_k, top_k * candidate_multiplier))
    total_start = perf_counter()

    try:
        try:
            vector_start = perf_counter()
            query_embedding = get_embedding(query)
            vector_results = model_cls.objects.filter_by_documents(docs).exclude(
                embedding__isnull=True
            ).order_by(L2Distance("embedding", query_embedding))[
                :vector_limit
            ]  # type: ignore
            vector_ms = (perf_counter() - vector_start) * 1000
        except Exception as exc:
            logger.warning(
                "Vector embed/search failed; continuing with trigram-only retrieval. Error: %s",
                exc,
            )
            vector_results = model_cls.objects.none()
            vector_ms = (perf_counter() - total_start) * 1000
        trigram_start = perf_counter()
        trigram_results = (
            model_cls.objects.filter_by_documents(docs)
            .filter(modality=model_cls.Modality.TEXT)
            .annotate(similarity=TrigramSimilarity("content", query))  # type: ignore
            .filter(similarity__gt=0.000001)
            .order_by("-similarity")[:trigram_limit]
        )
        trigram_ms = (perf_counter() - trigram_start) * 1000
        combined_candidates = list(vector_results) + list(trigram_results)
        deduped_candidates = []
        seen_pks = set()
        for candidate in combined_candidates:
            if candidate.pk in seen_pks:
                continue
            seen_pks.add(candidate.pk)
            deduped_candidates.append(candidate)
        combined_candidates = deduped_candidates
        if len(combined_candidates) <= top_k:
            reranked_results = _fallback_rerank(model_cls, combined_candidates, top_k)
            rerank_ms = 0.0
        else:
            rerank_start = perf_counter()
            reranked_results = rerank_chunks(model_cls, query, combined_candidates, top_k)
            rerank_ms = (perf_counter() - rerank_start) * 1000
        total_ms = (perf_counter() - total_start) * 1000
        logger.info(
            "text_chunk_search latency %.1fms (vector=%.1fms trigram=%.1fms rerank=%.1fms docs=%d top_k=%d candidates=%d)",
            total_ms,
            vector_ms,
            trigram_ms,
            rerank_ms,
            len(docs),
            top_k,
            len(combined_candidates),
        )
        return vector_results, trigram_results, reranked_results
    except DatabaseError as e:
        logger.error(f"Database error during search: {str(e)}")
        raise e
    except ValidationError as e:
        logger.error(f"Validation error during search: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error during search: {str(e)}")
        raise e


__all__ = ["text_chunk_search"]
