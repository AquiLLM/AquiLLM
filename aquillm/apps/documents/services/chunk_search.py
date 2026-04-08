"""Hybrid vector + trigram chunk retrieval with reranking."""
from __future__ import annotations

import structlog
from time import perf_counter
from typing import TYPE_CHECKING, List, Type

from django.apps import apps
from django.conf import settings as django_settings
from django.contrib.postgres.search import TrigramSimilarity
from django.core.exceptions import ValidationError
from django.db import DatabaseError
from pgvector.django import L2Distance

from aquillm.metrics import chunk_search_duration
from apps.documents.services.chunk_rerank import _fallback_rerank, rerank_chunks

if TYPE_CHECKING:
    from apps.documents.models.chunks import TextChunk

logger = structlog.stdlib.get_logger(__name__)


def text_chunk_search(model_cls: Type[TextChunk], query: str, top_k: int, docs: List):
    from aquillm.utils import get_embedding
    from apps.documents.services import rag_cache
    from lib.embeddings.config import get_local_embed_config

    vector_top_k = apps.get_app_config("aquillm").vector_top_k  # type: ignore
    trigram_top_k = apps.get_app_config("aquillm").trigram_top_k  # type: ignore
    qstrip = query.strip()
    q_len = len(qstrip)
    short_len = int(getattr(django_settings, "RAG_QUERY_SHORT_LEN", 48))
    long_len = int(getattr(django_settings, "RAG_QUERY_LONG_LEN", 160))
    short_scale = float(getattr(django_settings, "RAG_SHORT_QUERY_CANDIDATE_SCALE", 0.9))
    long_scale = float(getattr(django_settings, "RAG_LONG_QUERY_CANDIDATE_SCALE", 1.1))
    if q_len <= short_len:
        len_scale = short_scale
    elif q_len >= long_len:
        len_scale = long_scale
    else:
        len_scale = 1.0
    mult = float(getattr(django_settings, "RAG_CANDIDATE_MULTIPLIER", 3.0))
    eff_mult = mult * len_scale
    raw_cap = int(top_k * eff_mult)
    vector_min = int(getattr(django_settings, "RAG_VECTOR_MIN_LIMIT", 0))
    trigram_min = int(getattr(django_settings, "RAG_TRIGRAM_MIN_LIMIT", 0))
    vector_limit = max(top_k + 2, vector_min, min(vector_top_k, raw_cap))
    trigram_limit = max(top_k + 2, trigram_min, min(trigram_top_k, raw_cap))
    tri_sim_min = float(getattr(django_settings, "RAG_TRIGRAM_SIMILARITY_MIN", 0.000001))
    total_start = perf_counter()

    try:
        try:
            vector_start = perf_counter()
            _embed_base, _embed_key, embed_model = get_local_embed_config()
            cached_vec = rag_cache.get_cached_query_embedding(query, "search_query", embed_model)
            if cached_vec is not None:
                query_embedding = cached_vec
            else:
                query_embedding = get_embedding(query)
                rag_cache.set_cached_query_embedding(query, "search_query", embed_model, query_embedding)
            vector_results = (
                model_cls.objects.filter_by_documents(docs)
                .exclude(embedding__isnull=True)
                .defer("embedding")
                .order_by(L2Distance("embedding", query_embedding))[:vector_limit]
            )  # type: ignore
            vector_ms = (perf_counter() - vector_start) * 1000
            chunk_search_duration.labels(phase="vector").observe(vector_ms / 1000)
        except Exception as exc:
            logger.warning("obs.rag.vector_fallback", error=str(exc))
            vector_results = model_cls.objects.none()
            vector_ms = (perf_counter() - total_start) * 1000
        trigram_start = perf_counter()
        trigram_results = (
            model_cls.objects.filter_by_documents(docs)
            .filter(modality=model_cls.Modality.TEXT)
            .annotate(similarity=TrigramSimilarity("content", query))  # type: ignore
            .filter(similarity__gt=tri_sim_min)
            .order_by("-similarity")[:trigram_limit]
        )
        trigram_ms = (perf_counter() - trigram_start) * 1000
        chunk_search_duration.labels(phase="trigram").observe(trigram_ms / 1000)
        combined_candidates = list(vector_results) + list(trigram_results)
        pre_dedupe_count = len(combined_candidates)
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
            chunk_search_duration.labels(phase="rerank").observe(rerank_ms / 1000)
        total_ms = (perf_counter() - total_start) * 1000
        chunk_search_duration.labels(phase="total").observe(total_ms / 1000)
        logger.info(
            "obs.rag.chunk_search",
            total_ms=total_ms,
            vector_ms=vector_ms,
            trigram_ms=trigram_ms,
            rerank_ms=rerank_ms,
            doc_count=len(docs),
            top_k=top_k,
            pre_dedupe=pre_dedupe_count,
            candidates=len(combined_candidates),
        )
        return vector_results, trigram_results, reranked_results
    except DatabaseError as e:
        logger.error("obs.rag.search_error", error_type="database", error=str(e))
        raise e
    except ValidationError as e:
        logger.error("obs.rag.search_error", error_type="validation", error=str(e))
        raise e
    except Exception as e:
        logger.error("obs.rag.search_error", error_type="unexpected", error=str(e))
        raise e


__all__ = ["text_chunk_search"]
