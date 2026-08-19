"""Reusable baseline candidate acquisition and graph-seed preparation."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from math import fsum, isfinite
from time import perf_counter
from typing import TYPE_CHECKING, Any
from uuid import UUID

import structlog
from django.apps import apps
from django.conf import settings as django_settings
from django.contrib.postgres.search import TrigramSimilarity
from django.db.models import Q
from pgvector.django import L2Distance

if TYPE_CHECKING:
    from apps.knowledge_graph.retrieval import (
        GraphExpansionConfig,
        GraphExpansionSeed,
    )

logger = structlog.stdlib.get_logger(__name__)

_DATABASE_ID_MAX = 2**63 - 1
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


class CandidateScopeLimit(ValueError):
    """The authorized document snapshot exceeds a configured graph ceiling."""


@dataclass(frozen=True, slots=True)
class AuthorizedDocumentScope:
    """Exact scalar permission scope derived only from authorized documents."""

    documents: tuple[object, ...]
    allowed_doc_ids: tuple[UUID, ...]
    allowed_collection_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class HybridCandidateSnapshot:
    """Immutable pre-rerank snapshot shared by production and evaluation."""

    documents: tuple[object, ...]
    vector_results: object
    trigram_results: object
    exact_results: object
    vector_chunk_ids: tuple[int, ...]
    trigram_chunk_ids: tuple[int, ...]
    exact_chunk_ids: tuple[int, ...]
    baseline_candidates: tuple[object, ...]
    graph_seeds: tuple[GraphExpansionSeed, ...]
    exact_terms: tuple[str, ...]
    vector_error: str | None
    vector_ms: float
    trigram_ms: float
    exact_ms: float
    pre_dedupe_count: int
    graph_seed_error: bool


@dataclass(frozen=True, slots=True)
class _CandidateLimits:
    vector: int
    trigram: int
    exact: int
    trigram_similarity_min: float


def _salient_exact_terms(query: str, *, max_terms: int = 8) -> list[str]:
    """Extract exact fallback terms that are likely to matter for recall."""

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

    for quoted in re.findall(r'"([^\"]{3,96})"|\'([^\']{3,96})\'', query or ""):
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


def _candidate_limits(
    query: str,
    top_k: int,
    *,
    app_config_getter: Callable[[str], object],
) -> _CandidateLimits:
    app_config = app_config_getter("aquillm")
    vector_top_k = int(getattr(app_config, "vector_top_k"))
    trigram_top_k = int(getattr(app_config, "trigram_top_k"))
    q_len = len(query.strip())
    short_len = int(getattr(django_settings, "RAG_QUERY_SHORT_LEN", 48))
    long_len = int(getattr(django_settings, "RAG_QUERY_LONG_LEN", 160))
    short_scale = float(
        getattr(django_settings, "RAG_SHORT_QUERY_CANDIDATE_SCALE", 0.9)
    )
    long_scale = float(getattr(django_settings, "RAG_LONG_QUERY_CANDIDATE_SCALE", 1.1))
    if q_len <= short_len:
        length_scale = short_scale
    elif q_len >= long_len:
        length_scale = long_scale
    else:
        length_scale = 1.0
    multiplier = float(getattr(django_settings, "RAG_CANDIDATE_MULTIPLIER", 3.0))
    raw_cap = int(top_k * multiplier * length_scale)
    vector_min = int(getattr(django_settings, "RAG_VECTOR_MIN_LIMIT", 0))
    trigram_min = int(getattr(django_settings, "RAG_TRIGRAM_MIN_LIMIT", 0))
    vector_limit = max(top_k + 2, vector_min, min(vector_top_k, raw_cap))
    trigram_limit = max(top_k + 2, trigram_min, min(trigram_top_k, raw_cap))
    exact_limit = max(top_k + 2, min(trigram_top_k, raw_cap))
    similarity_min = float(
        getattr(django_settings, "RAG_TRIGRAM_SIMILARITY_MIN", 0.000001)
    )
    return _CandidateLimits(
        vector=vector_limit,
        trigram=trigram_limit,
        exact=exact_limit,
        trigram_similarity_min=similarity_min,
    )


def _positive_chunk_id(candidate: object) -> int:
    identifier = getattr(candidate, "pk", None)
    if type(identifier) is not int or not 1 <= identifier <= _DATABASE_ID_MAX:
        raise ValueError("candidate chunk IDs must be positive database integers")
    return identifier


def _source_first_ranks(rows: Sequence[object]) -> dict[int, int]:
    ranks: dict[int, int] = {}
    for rank, row in enumerate(rows, start=1):
        identifier = _positive_chunk_id(row)
        ranks.setdefault(identifier, rank)
    return ranks


def _build_graph_seeds(
    vector_rows: Sequence[object],
    trigram_rows: Sequence[object],
    exact_rows: Sequence[object],
    graph_config: GraphExpansionConfig,
) -> tuple[GraphExpansionSeed, ...]:
    # Deliberately lazy: importing the public KG composition boundary is an
    # enabled-overlay action, never a baseline-search import side effect.
    from apps.knowledge_graph.retrieval import (
        GraphExpansionConfig,
        GraphExpansionSeed,
    )

    if type(graph_config) is not GraphExpansionConfig:
        raise ValueError("graph_config must be an exact GraphExpansionConfig")
    contributions: dict[int, list[float]] = {}
    for rows in (vector_rows, trigram_rows, exact_rows):
        for identifier, rank in _source_first_ranks(rows).items():
            contribution = 1.0 / (graph_config.rrf_k + rank)
            if not isfinite(contribution) or contribution <= 0.0:
                raise ValueError("RRF produced a non-finite seed contribution")
            contributions.setdefault(identifier, []).append(contribution)

    weighted = [
        (identifier, fsum(values)) for identifier, values in contributions.items()
    ]
    if any(not isfinite(weight) or weight <= 0.0 for _identifier, weight in weighted):
        raise ValueError("RRF produced a non-finite seed weight")
    weighted.sort(key=lambda item: (-item[1], item[0]))
    selected = weighted[: graph_config.max_seeds]
    if not selected:
        return ()
    total = fsum(weight for _identifier, weight in selected)
    if not isfinite(total) or total <= 0.0:
        raise ValueError("RRF produced a non-finite normalization total")
    return tuple(
        GraphExpansionSeed(
            chunk_id=identifier,
            rank=rank,
            restart_weight=weight / total,
        )
        for rank, (identifier, weight) in enumerate(selected, start=1)
    )


def freeze_authorized_document_scope(
    documents: Iterable[object],
    graph_config: GraphExpansionConfig,
) -> AuthorizedDocumentScope:
    """Freeze exact UUID/collection scalars without following lazy relations."""

    from apps.knowledge_graph.retrieval import GraphExpansionConfig

    if type(graph_config) is not GraphExpansionConfig:
        raise ValueError("graph_config must be an exact GraphExpansionConfig")
    by_document_id: dict[UUID, object] = {}
    collection_ids: set[int] = set()
    for document in documents:
        if len(by_document_id) >= graph_config.max_scope_documents:
            raise CandidateScopeLimit("authorized document scope exceeds its cap")
        document_id = getattr(document, "id", None)
        collection_id = getattr(document, "collection_id", None)
        if type(document_id) is not UUID:
            raise ValueError("authorized documents must expose exact UUID ids")
        if type(collection_id) is not int or not 1 <= collection_id <= _DATABASE_ID_MAX:
            raise ValueError(
                "authorized documents must expose positive collection_id integers"
            )
        if document_id in by_document_id:
            raise ValueError("authorized document UUIDs must be unique")
        by_document_id[document_id] = document
        collection_ids.add(collection_id)
        if len(collection_ids) > graph_config.max_scope_collections:
            raise CandidateScopeLimit("authorized collection scope exceeds its cap")
    if not by_document_id or not collection_ids:
        raise ValueError("authorized graph scope must not be empty")
    ordered_document_ids = tuple(sorted(by_document_id, key=lambda value: value.int))
    return AuthorizedDocumentScope(
        documents=tuple(
            by_document_id[identifier] for identifier in ordered_document_ids
        ),
        allowed_doc_ids=ordered_document_ids,
        allowed_collection_ids=tuple(sorted(collection_ids)),
    )


def collect_hybrid_candidate_snapshot(
    model_cls: Any,
    query: str,
    top_k: int,
    documents: Iterable[object],
    *,
    query_embedding: object | None,
    graph_config: GraphExpansionConfig | None = None,
    initial_vector_error: str | None = None,
    app_config_getter: Callable[[str], object] | None = None,
) -> HybridCandidateSnapshot:
    """Acquire vector/trigram/exact rows once and prepare the pre-rerank pool."""

    documents_snapshot = tuple(documents)
    limits = _candidate_limits(
        query,
        top_k,
        app_config_getter=app_config_getter or apps.get_app_config,
    )
    vector_error = initial_vector_error
    vector_started = perf_counter()
    if query_embedding is None:
        vector_results = model_cls.objects.none()
        vector_rows: tuple[object, ...] = ()
    else:
        try:
            vector_results = (
                model_cls.objects.filter_by_documents(documents_snapshot)
                .exclude(embedding__isnull=True)
                .defer("embedding")
                .order_by(L2Distance("embedding", query_embedding))[: limits.vector]
            )
            # Force database evaluation inside the vector-only fail-open seam.
            vector_rows = tuple(vector_results)
        except Exception as exc:
            vector_error = str(exc)
            logger.warning(
                "obs.rag.vector_search_failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            vector_results = model_cls.objects.none()
            vector_rows = ()
    vector_ms = (perf_counter() - vector_started) * 1000

    trigram_started = perf_counter()
    trigram_results = (
        model_cls.objects.filter_by_documents(documents_snapshot)
        .filter(modality=model_cls.Modality.TEXT)
        .annotate(similarity=TrigramSimilarity("content", query))
        .filter(similarity__gt=limits.trigram_similarity_min)
        .order_by("-similarity")[: limits.trigram]
    )
    trigram_rows = tuple(trigram_results)
    trigram_ms = (perf_counter() - trigram_started) * 1000

    exact_started = perf_counter()
    exact_terms = _salient_exact_terms(query)
    if exact_terms:
        exact_results = (
            model_cls.objects.filter_by_documents(documents_snapshot)
            .filter(modality=model_cls.Modality.TEXT)
            .filter(_exact_term_query(exact_terms))
            .order_by("doc_id", "chunk_number")[: limits.exact]
        )
        exact_rows = tuple(exact_results)
    else:
        exact_results = model_cls.objects.none()
        exact_rows = ()
    exact_ms = (perf_counter() - exact_started) * 1000

    all_rows = (*vector_rows, *trigram_rows, *exact_rows)
    baseline: list[object] = []
    seen_ids: set[object] = set()
    for candidate in all_rows:
        identifier = getattr(candidate, "pk", None)
        if identifier in seen_ids:
            continue
        seen_ids.add(identifier)
        baseline.append(candidate)

    graph_seeds: tuple[GraphExpansionSeed, ...] = ()
    graph_seed_error = False
    if graph_config is not None:
        try:
            graph_seeds = _build_graph_seeds(
                vector_rows,
                trigram_rows,
                exact_rows,
                graph_config,
            )
        except Exception:
            graph_seed_error = True

    return HybridCandidateSnapshot(
        documents=documents_snapshot,
        vector_results=vector_results,
        trigram_results=trigram_results,
        exact_results=exact_results,
        vector_chunk_ids=tuple(getattr(row, "pk") for row in vector_rows),
        trigram_chunk_ids=tuple(getattr(row, "pk") for row in trigram_rows),
        exact_chunk_ids=tuple(getattr(row, "pk") for row in exact_rows),
        baseline_candidates=tuple(baseline),
        graph_seeds=graph_seeds,
        exact_terms=tuple(exact_terms),
        vector_error=vector_error,
        vector_ms=vector_ms,
        trigram_ms=trigram_ms,
        exact_ms=exact_ms,
        pre_dedupe_count=len(all_rows),
        graph_seed_error=graph_seed_error,
    )


__all__ = [
    "AuthorizedDocumentScope",
    "CandidateScopeLimit",
    "HybridCandidateSnapshot",
    "collect_hybrid_candidate_snapshot",
    "freeze_authorized_document_scope",
]
