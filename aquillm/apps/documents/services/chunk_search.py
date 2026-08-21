"""Hybrid chunk retrieval with an optional fail-open graph overlay."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING
from uuid import UUID

import structlog
from django.apps import apps
from django.conf import settings as django_settings
from django.core.exceptions import ValidationError
from django.db import DatabaseError

from apps.documents.services.chunk_rerank import (
    _STRICT_EVALUATION_RERANK,
    _fallback_rerank,
    _strict_local_rerank_chunks,
    rerank_chunks,
)
from apps.documents.services.chunk_search_candidates import (
    CandidateScopeLimit,
    collect_hybrid_candidate_snapshot,
    freeze_authorized_document_scope,
)
from apps.documents.services.chunk_search_candidates import (
    _exact_term_query as _candidate_exact_term_query,
)
from apps.documents.services.chunk_search_candidates import (
    _salient_exact_terms as _candidate_salient_exact_terms,
)
from apps.documents.services.chunk_search_legacy_graph import (
    EVALUATION_GRAPH_FAILURE,
    EVALUATION_GRAPH_MISS,
)
from apps.documents.services.chunk_search_legacy_graph import (
    apply_graph_overlay as _apply_graph_overlay,
)
from apps.documents.services.chunk_search_legacy_graph import (
    graph_diagnostics as _graph_diagnostics,
)
from apps.documents.services.hybrid_graph_authorization import (
    HybridGraphRetrievalDependencies,
    documents_match_retrieval_authorization,
    is_exact_authorization_context,
)
from apps.documents.services.hybrid_graph_orchestration import (
    hybrid_graph_candidate_pool,
)

if TYPE_CHECKING:
    from apps.documents.models.chunks import TextChunk

logger = structlog.stdlib.get_logger(__name__)
_EVALUATION_GRAPH_FAILURE = EVALUATION_GRAPH_FAILURE
_EVALUATION_GRAPH_MISS = EVALUATION_GRAPH_MISS

# Compatibility aliases used by conversation-search's independent chunk model.
_exact_term_query = _candidate_exact_term_query
_salient_exact_terms = _candidate_salient_exact_terms


@dataclass(frozen=True, slots=True)
class CandidateRankingResult:
    """One permission-checked candidate pool and its production rerank output."""

    combined_candidates: tuple[object, ...]
    graph_candidates: tuple[object, ...]
    ranked_results: tuple[object, ...]
    inaccessible_candidate_count: int
    materialization_ms: float
    rerank_ms: float


def _candidate_identifier(candidate: object) -> int:
    identifier = getattr(candidate, "pk", None)
    if type(identifier) is not int or identifier <= 0:
        raise ValueError("candidate rows require positive integer primary keys")
    return identifier


def _candidate_identity(candidate: object) -> tuple[str, int]:
    """Use a durable PK when present, otherwise exact in-memory identity."""

    identifier = getattr(candidate, "pk", None)
    if type(identifier) is int and identifier > 0:
        return ("pk", identifier)
    return ("object", id(candidate))


def materialize_and_rerank_candidates(
    model_cls: type[TextChunk],
    query: str,
    top_k: int,
    baseline_candidates: tuple[object, ...],
    *,
    authorized_scope: object | None,
    graph_chunk_ids: tuple[int, ...] = (),
    max_graph_candidates: int = 0,
    force_complete_rerank: bool = False,
    _eval_rerank_capability: object | None = None,
) -> CandidateRankingResult:
    """Permission-refetch graph rows, append to the baseline, and rerank once."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be nonempty")
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive exact integer")
    if type(baseline_candidates) is not tuple:
        raise ValueError("baseline_candidates must be an exact tuple")
    baseline_identities = tuple(_candidate_identity(row) for row in baseline_candidates)
    if len(set(baseline_identities)) != len(baseline_identities):
        raise ValueError("baseline candidates must be unique")
    if type(graph_chunk_ids) is not tuple or any(
        type(identifier) is not int or identifier <= 0 for identifier in graph_chunk_ids
    ):
        raise ValueError("graph_chunk_ids must be an exact positive integer tuple")
    if len(set(graph_chunk_ids)) != len(graph_chunk_ids):
        raise ValueError("graph_chunk_ids must be unique")
    if type(max_graph_candidates) is not int or max_graph_candidates < 0:
        raise ValueError("max_graph_candidates must be a nonnegative exact integer")
    if type(force_complete_rerank) is not bool:
        raise ValueError("force_complete_rerank must be an exact bool")
    if (
        _eval_rerank_capability is not None
        and _eval_rerank_capability is not _STRICT_EVALUATION_RERANK
    ):
        raise ValueError("invalid strict eval rerank capability")
    if len(graph_chunk_ids) > max_graph_candidates:
        raise CandidateScopeLimit("graph candidate materialization exceeds its cap")

    materialization_started = perf_counter()
    inaccessible = 0
    allowed_doc_ids: tuple[UUID, ...] = ()
    if authorized_scope is not None:
        from apps.documents.services.chunk_search_candidates import (
            AuthorizedDocumentScope,
        )

        if type(authorized_scope) is not AuthorizedDocumentScope:
            raise ValueError("authorized_scope must be an exact frozen scope")
        allowed_doc_ids = authorized_scope.allowed_doc_ids
        inaccessible += sum(
            getattr(row, "doc_id", None) not in set(allowed_doc_ids)
            for row in baseline_candidates
        )
    elif graph_chunk_ids:
        inaccessible += len(graph_chunk_ids)

    baseline_id_set = {
        identifier
        for identity_type, identifier in baseline_identities
        if identity_type == "pk"
    }
    novel_ids = tuple(
        identifier
        for identifier in graph_chunk_ids
        if identifier not in baseline_id_set
    )
    graph_rows: tuple[object, ...] = ()
    if novel_ids and authorized_scope is not None:
        loaded = tuple(
            model_cls.objects.filter(
                pk__in=novel_ids,
                doc_id__in=allowed_doc_ids,
            )
        )
        by_identifier: dict[int, object] = {}
        allowed = set(allowed_doc_ids)
        invalid_materialization = False
        for row in loaded:
            identifier = _candidate_identifier(row)
            document_id = getattr(row, "doc_id", None)
            if (
                identifier not in novel_ids
                or identifier in by_identifier
                or type(document_id) is not UUID
                or document_id not in allowed
            ):
                invalid_materialization = True
                continue
            if hasattr(model_cls, "_meta") and not isinstance(row, model_cls):
                invalid_materialization = True
                continue
            by_identifier[identifier] = row
        missing = set(novel_ids).difference(by_identifier)
        if invalid_materialization or missing:
            inaccessible += len(novel_ids)
        else:
            graph_rows = tuple(by_identifier[identifier] for identifier in novel_ids)

    combined = (*baseline_candidates, *graph_rows)
    materialization_ms = (perf_counter() - materialization_started) * 1_000
    rerank_started = perf_counter()
    if _eval_rerank_capability is _STRICT_EVALUATION_RERANK:
        reranked = _strict_local_rerank_chunks(
            model_cls,
            query,
            combined,
            top_k,
            _capability=_eval_rerank_capability,
        )
    elif not force_complete_rerank and len(combined) <= top_k:
        reranked = _fallback_rerank(model_cls, combined, top_k)
    else:
        reranked = rerank_chunks(model_cls, query, combined, top_k)
    ranked_results = tuple(reranked)
    rerank_ms = (perf_counter() - rerank_started) * 1_000
    combined_identities = set(_candidate_identity(row) for row in combined)
    ranked_identities = tuple(_candidate_identity(row) for row in ranked_results)
    if len(set(ranked_identities)) != len(ranked_identities) or any(
        identity not in combined_identities for identity in ranked_identities
    ):
        raise ValueError("reranker returned rows outside the candidate pool")
    inaccessible += sum(
        getattr(row, "doc_id", None) not in set(allowed_doc_ids)
        for row in ranked_results
        if authorized_scope is not None
    )
    return CandidateRankingResult(
        combined_candidates=tuple(combined),
        graph_candidates=graph_rows,
        ranked_results=ranked_results,
        inaccessible_candidate_count=inaccessible,
        materialization_ms=materialization_ms,
        rerank_ms=rerank_ms,
    )


def text_chunk_search(
    model_cls: type[TextChunk],
    query: str,
    top_k: int,
    docs: list,
    *,
    authorization_context: object | None = None,
    hybrid_graph_dependencies: HybridGraphRetrievalDependencies | None = None,
):
    from apps.documents.services import rag_cache
    from aquillm.utils import get_embedding
    from lib.embeddings.config import get_local_embed_config

    total_start = perf_counter()
    try:
        overlay_enabled = bool(getattr(django_settings, "KG_OVERLAY_ENABLED", False))
        hybrid_requested = overlay_enabled and hybrid_graph_dependencies is not None
        graph_config: object | None = None
        graph_scope: object | None = None
        graph_preflight_status: str | None = None
        search_documents: object = docs
        if overlay_enabled:
            try:
                from apps.knowledge_graph.retrieval import (
                    get_graph_expansion_config,
                )

                graph_config = get_graph_expansion_config()
                if hybrid_requested:
                    if not documents_match_retrieval_authorization(
                        docs, authorization_context
                    ):
                        graph_preflight_status = "error"
                else:
                    graph_scope = freeze_authorized_document_scope(docs, graph_config)
                    search_documents = graph_scope.documents
            except CandidateScopeLimit:
                graph_preflight_status = "miss"
                graph_scope = None
            except Exception:
                graph_preflight_status = "error"
                graph_config = None
                graph_scope = None

        vector_started = perf_counter()
        initial_vector_error: str | None = None
        query_embedding: object | None = None
        try:
            _embed_base, _embed_key, embed_model = get_local_embed_config()
            cached_vec = rag_cache.get_cached_query_embedding(
                query,
                "search_query",
                embed_model,
            )
            if cached_vec is not None:
                query_embedding = cached_vec
            else:
                query_embedding = get_embedding(query)
                rag_cache.set_cached_query_embedding(
                    query,
                    "search_query",
                    embed_model,
                    query_embedding,
                )
        except Exception as exc:
            initial_vector_error = str(exc)
            logger.warning(
                "obs.rag.vector_search_failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )

        snapshot = collect_hybrid_candidate_snapshot(
            model_cls,
            query,
            top_k,
            search_documents,
            query_embedding=query_embedding,
            graph_config=graph_config if graph_preflight_status is None else None,
            initial_vector_error=initial_vector_error,
            app_config_getter=apps.get_app_config,
            authorization_context=(
                authorization_context
                if hybrid_requested and graph_preflight_status is None
                else None
            ),
        )
        vector_ms = (perf_counter() - vector_started) * 1000
        graph_chunk_ids: tuple[int, ...] = ()
        graph_diagnostics: dict[str, object] = {}
        hybrid_pool: tuple[object, ...] | None = None
        if overlay_enabled and hybrid_requested:
            if (
                graph_preflight_status is None
                and type(hybrid_graph_dependencies) is HybridGraphRetrievalDependencies
                and is_exact_authorization_context(authorization_context)
            ):
                hybrid_pool, graph_diagnostics = hybrid_graph_candidate_pool(
                    snapshot,
                    query,
                    authorization_context,
                    hybrid_graph_dependencies,
                )
            else:
                hybrid_pool = snapshot.baseline_candidates
                graph_diagnostics = _graph_diagnostics(
                    started_at=total_start,
                    seed_count=len(snapshot.graph_seeds),
                    candidate_count=0,
                    status=graph_preflight_status or "error",
                    algorithm_signature=getattr(
                        graph_config, "algorithm_signature", None
                    ),
                    version_signature=None,
                )
        elif overlay_enabled:
            graph_chunk_ids, graph_diagnostics = _apply_graph_overlay(
                model_cls,
                snapshot,
                graph_scope,
                graph_config,
                preflight_status=graph_preflight_status,
            )

        if hybrid_pool is not None:
            ranking = materialize_and_rerank_candidates(
                model_cls,
                query,
                top_k,
                hybrid_pool,
                authorized_scope=None,
                force_complete_rerank=graph_diagnostics.get("graph_status") == "hit",
            )
        else:
            try:
                ranking = materialize_and_rerank_candidates(
                    model_cls,
                    query,
                    top_k,
                    snapshot.baseline_candidates,
                    authorized_scope=graph_scope,
                    graph_chunk_ids=graph_chunk_ids,
                    max_graph_candidates=(
                        int(getattr(graph_config, "max_candidates", 0))
                        if graph_chunk_ids
                        else 0
                    ),
                )
            except Exception:
                ranking = materialize_and_rerank_candidates(
                    model_cls,
                    query,
                    top_k,
                    snapshot.baseline_candidates,
                    authorized_scope=None,
                )
                if overlay_enabled:
                    graph_diagnostics = _graph_diagnostics(
                        started_at=total_start,
                        seed_count=len(snapshot.graph_seeds),
                        candidate_count=0,
                        status="error",
                        algorithm_signature=getattr(
                            graph_config,
                            "algorithm_signature",
                            None,
                        ),
                        version_signature=None,
                    )
        if overlay_enabled and ranking.inaccessible_candidate_count:
            graph_diagnostics.update(
                graph_status="error",
                graph_candidate_count=0,
            )
        if overlay_enabled:
            graph_diagnostics["graph_ms"] = (
                float(graph_diagnostics.get("graph_ms", 0.0))
                + ranking.materialization_ms
            )
            if not hybrid_requested and graph_diagnostics.get("graph_status") == "hit":
                graph_diagnostics["graph_candidate_count"] = len(
                    ranking.graph_candidates
                )
        combined_candidates = ranking.combined_candidates
        reranked_results = list(ranking.ranked_results)
        rerank_ms = ranking.rerank_ms

        total_ms = (perf_counter() - total_start) * 1000
        logger.info(
            "obs.rag.search",
            total_ms=total_ms,
            vector_ms=vector_ms,
            trigram_ms=snapshot.trigram_ms,
            exact_ms=snapshot.exact_ms,
            rerank_ms=rerank_ms,
            doc_count=len(docs),
            top_k=top_k,
            exact_term_count=len(snapshot.exact_terms),
            pre_dedupe_count=snapshot.pre_dedupe_count,
            candidate_count=len(combined_candidates),
            **graph_diagnostics,
        )
        chunks_with_embeddings: int | None = None
        if not reranked_results:
            try:
                chunks_with_embeddings = (
                    model_cls.objects.filter_by_documents(snapshot.documents)
                    .exclude(embedding__isnull=True)
                    .count()
                )
            except Exception as count_exc:
                logger.warning("Could not count chunks_with_embeddings: %s", count_exc)
        diagnostics: dict = {
            "doc_count": len(docs),
            "chunks_with_embeddings": chunks_with_embeddings,
            "vector_error": snapshot.vector_error,
            "trigram_candidates": len(snapshot.trigram_chunk_ids),
            "exact_terms": list(snapshot.exact_terms),
        }
        if overlay_enabled:
            diagnostics.update(graph_diagnostics)
        if not reranked_results:
            logger.info(
                "text_chunk_search returned no results",
                extra={
                    "doc_count": diagnostics["doc_count"],
                    "chunks_with_embeddings": chunks_with_embeddings,
                    "vector_error": snapshot.vector_error,
                    "trigram_candidates": diagnostics["trigram_candidates"],
                    "exact_terms": list(snapshot.exact_terms),
                    **graph_diagnostics,
                },
            )
        return (
            snapshot.vector_results,
            snapshot.trigram_results,
            reranked_results,
            diagnostics,
        )
    except DatabaseError as error:
        logger.error(
            "obs.rag.search_db_error",
            error=str(error),
            error_type=type(error).__name__,
        )
        raise
    except ValidationError as error:
        logger.error(
            "obs.rag.search_validation_error",
            error=str(error),
            error_type=type(error).__name__,
        )
        raise
    except Exception as error:
        logger.error(
            "obs.rag.search_error",
            error=str(error),
            error_type=type(error).__name__,
        )
        raise


__all__ = [
    "CandidateRankingResult",
    "HybridGraphRetrievalDependencies",
    "materialize_and_rerank_candidates",
    "text_chunk_search",
]
