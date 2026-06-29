"""Hybrid vector + trigram chunk retrieval with reranking."""
from __future__ import annotations

import re
import structlog
from time import perf_counter
from typing import TYPE_CHECKING, List, Type

from django.apps import apps
from django.conf import settings as django_settings
from django.contrib.postgres.search import TrigramSimilarity
from django.core.exceptions import ValidationError
from django.db import DatabaseError
from django.db.models import Q
from pgvector.django import L2Distance

from apps.documents.services.chunk_rerank import _fallback_rerank, rerank_chunks

if TYPE_CHECKING:
    from apps.documents.models.chunks import TextChunk

logger = structlog.stdlib.get_logger(__name__)


_EXACT_STOPWORDS = {
    "about",
    "after",
    "answer",
    "before",
    "could",
    "document",
    "documents",
    "explain",
    "information",
    "selected",
    "should",
    "through",
    "where",
    "which",
    "would",
}


def _salient_exact_terms(query: str, *, max_terms: int = 8) -> list[str]:
    """Extract exact fallback terms that are likely to matter for document recall."""
    terms: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        cleaned = term.strip(" \t\r\n\"'`.,;:!?()[]{}")
        if len(cleaned) < 3:
            return
        key = cleaned.lower()
        if key in seen or key in _EXACT_STOPWORDS:
            return
        seen.add(key)
        terms.append(cleaned)

    for quoted in re.findall(r'"([^"]{3,96})"|\'([^\']{3,96})\'', query or ""):
        add(quoted[0] or quoted[1])

    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_./:+-]{2,}", query or ""):
        lowered = token.lower()
        has_symbol = any(ch in token for ch in "-_./:+")
        has_digit = any(ch.isdigit() for ch in token)
        uppercase_count = sum(1 for ch in token if ch.isupper())
        is_acronym = uppercase_count >= 2
        is_long_domain_word = len(token) >= 10 and lowered not in _EXACT_STOPWORDS
        if has_symbol or has_digit or is_acronym or is_long_domain_word:
            add(token)
        if len(terms) >= max_terms:
            break

    return terms[:max_terms]


def _exact_term_query(terms: list[str]) -> Q:
    query = Q()
    for term in terms:
        query |= Q(content__icontains=term)
    return query


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
    exact_limit = max(top_k + 2, min(trigram_top_k, raw_cap))
    tri_sim_min = float(getattr(django_settings, "RAG_TRIGRAM_SIMILARITY_MIN", 0.000001))
    total_start = perf_counter()

    try:
        vector_error: str | None = None
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
        except Exception as exc:
            vector_error = str(exc)
            logger.warning(
                "obs.rag.vector_search_failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )
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
        exact_start = perf_counter()
        exact_terms = _salient_exact_terms(query)
        if exact_terms:
            exact_results = (
                model_cls.objects.filter_by_documents(docs)
                .filter(modality=model_cls.Modality.TEXT)
                .filter(_exact_term_query(exact_terms))
                .order_by("doc_id", "chunk_number")[:exact_limit]
            )
        else:
            exact_results = model_cls.objects.none()
        exact_ms = (perf_counter() - exact_start) * 1000
        vec_list = list(vector_results)
        tri_list = list(trigram_results)
        exact_list = list(exact_results)
        combined_candidates = vec_list + tri_list + exact_list
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
        total_ms = (perf_counter() - total_start) * 1000
        logger.info(
            "obs.rag.search",
            total_ms=total_ms,
            vector_ms=vector_ms,
            trigram_ms=trigram_ms,
            exact_ms=exact_ms,
            rerank_ms=rerank_ms,
            doc_count=len(docs),
            top_k=top_k,
            exact_term_count=len(exact_terms),
            pre_dedupe_count=pre_dedupe_count,
            candidate_count=len(combined_candidates),
        )
        chunks_with_embeddings: int | None = None
        if not reranked_results:
            try:
                chunks_with_embeddings = (
                    model_cls.objects.filter_by_documents(docs)
                    .exclude(embedding__isnull=True)
                    .count()
                )
            except Exception as count_exc:
                logger.warning("Could not count chunks_with_embeddings: %s", count_exc)
        diagnostics: dict = {
            "doc_count": len(docs),
            "chunks_with_embeddings": chunks_with_embeddings,
            "vector_error": vector_error,
            "trigram_candidates": len(tri_list),
            "exact_terms": exact_terms,
        }
        if not reranked_results:
            logger.info(
                "text_chunk_search returned no results",
                extra={
                    "doc_count": diagnostics["doc_count"],
                    "chunks_with_embeddings": chunks_with_embeddings,
                    "vector_error": vector_error,
                    "trigram_candidates": diagnostics["trigram_candidates"],
                    "exact_terms": exact_terms,
                },
            )
        return vector_results, trigram_results, reranked_results, diagnostics
    except DatabaseError as e:
        logger.error("obs.rag.search_db_error", error=str(e), error_type=type(e).__name__)
        raise e
    except ValidationError as e:
        logger.error("obs.rag.search_validation_error", error=str(e), error_type=type(e).__name__)
        raise e
    except Exception as e:
        logger.error("obs.rag.search_error", error=str(e), error_type=type(e).__name__)
        raise e


__all__ = ["text_chunk_search"]
