"""Hybrid chunk retrieval with an optional fail-open graph overlay."""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING
from uuid import UUID

import structlog
from django.apps import apps
from django.conf import settings as django_settings
from django.core.exceptions import ValidationError
from django.db import DatabaseError

from apps.documents.services.chunk_rerank import _fallback_rerank, rerank_chunks
from apps.documents.services.chunk_search_candidates import (
    CandidateScopeLimit,
    HybridCandidateSnapshot,
    collect_hybrid_candidate_snapshot,
    freeze_authorized_document_scope,
)
from apps.documents.services.chunk_search_candidates import (
    _exact_term_query as _candidate_exact_term_query,
)
from apps.documents.services.chunk_search_candidates import (
    _salient_exact_terms as _candidate_salient_exact_terms,
)

if TYPE_CHECKING:
    from apps.documents.models.chunks import TextChunk

logger = structlog.stdlib.get_logger(__name__)

# Compatibility aliases used by conversation-search's independent chunk model.
_exact_term_query = _candidate_exact_term_query
_salient_exact_terms = _candidate_salient_exact_terms


def _graph_diagnostics(
    *,
    started_at: float,
    seed_count: int,
    candidate_count: int,
    status: str,
    algorithm_signature: str | None,
    version_signature: str | None,
) -> dict[str, object]:
    return {
        "graph_ms": (perf_counter() - started_at) * 1000,
        "graph_seed_count": seed_count,
        "graph_candidate_count": candidate_count,
        "graph_status": status,
        "graph_algorithm_signature": algorithm_signature,
        "graph_version_signature": version_signature,
    }


def _apply_graph_overlay(
    model_cls: type[TextChunk],
    snapshot: HybridCandidateSnapshot,
    scope: object | None,
    graph_config: object | None,
    *,
    preflight_status: str | None,
) -> tuple[list[object], dict[str, object]]:
    """Append authorized ORM chunks or return the exact baseline pool."""

    started_at = perf_counter()
    baseline = list(snapshot.baseline_candidates)
    algorithm_signature = getattr(graph_config, "algorithm_signature", None)
    seed_count = len(snapshot.graph_seeds)
    if preflight_status is not None:
        return baseline, _graph_diagnostics(
            started_at=started_at,
            seed_count=seed_count,
            candidate_count=0,
            status=preflight_status,
            algorithm_signature=algorithm_signature,
            version_signature=None,
        )
    if snapshot.graph_seed_error:
        return baseline, _graph_diagnostics(
            started_at=started_at,
            seed_count=0,
            candidate_count=0,
            status="error",
            algorithm_signature=algorithm_signature,
            version_signature=None,
        )
    if not snapshot.graph_seeds:
        return baseline, _graph_diagnostics(
            started_at=started_at,
            seed_count=0,
            candidate_count=0,
            status="miss",
            algorithm_signature=algorithm_signature,
            version_signature=None,
        )

    try:
        from apps.documents.services.chunk_search_candidates import (
            AuthorizedDocumentScope,
        )
        from apps.knowledge_graph.retrieval import (
            GraphExpansionConfig,
            GraphExpansionRequest,
            GraphExpansionResult,
            expand_chunk_candidates,
        )

        if type(scope) is not AuthorizedDocumentScope:
            raise ValueError("graph scope must be an exact authorized snapshot")
        if type(graph_config) is not GraphExpansionConfig:
            raise ValueError("graph config must be exact")
        request = GraphExpansionRequest(
            seeds=snapshot.graph_seeds,
            allowed_doc_ids=scope.allowed_doc_ids,
            allowed_collection_ids=scope.allowed_collection_ids,
        )
        result = expand_chunk_candidates(request)
        if type(result) is not GraphExpansionResult:
            raise ValueError("graph expansion returned an invalid result")
        result_diagnostics = result.diagnostics
        if result_diagnostics.algorithm_signature != graph_config.algorithm_signature:
            raise ValueError("graph expansion used a different algorithm config")
        if result_diagnostics.status != "hit":
            return baseline, _graph_diagnostics(
                started_at=started_at,
                seed_count=result_diagnostics.seed_count,
                candidate_count=0,
                status=result_diagnostics.status,
                algorithm_signature=result_diagnostics.algorithm_signature,
                version_signature=result_diagnostics.graph_version_signature,
            )

        baseline_ids = {getattr(candidate, "pk", None) for candidate in baseline}
        novel_ids = tuple(
            identifier
            for identifier in result.chunk_ids
            if identifier not in baseline_ids
        )[: graph_config.max_candidates]
        if not novel_ids:
            return baseline, _graph_diagnostics(
                started_at=started_at,
                seed_count=result_diagnostics.seed_count,
                candidate_count=0,
                status="miss",
                algorithm_signature=result_diagnostics.algorithm_signature,
                version_signature=result_diagnostics.graph_version_signature,
            )

        rows = tuple(
            model_cls.objects.filter(
                pk__in=novel_ids,
                doc_id__in=scope.allowed_doc_ids,
            )
        )
        allowed_doc_ids = set(scope.allowed_doc_ids)
        by_identifier: dict[int, object] = {}
        for row in rows:
            identifier = getattr(row, "pk", None)
            document_id = getattr(row, "doc_id", None)
            if (
                type(identifier) is not int
                or identifier not in novel_ids
                or identifier in by_identifier
                or type(document_id) is not UUID
                or document_id not in allowed_doc_ids
            ):
                raise ValueError("graph expansion returned an unauthorized chunk row")
            if hasattr(model_cls, "_meta") and not isinstance(row, model_cls):
                raise ValueError("graph expansion must resolve real TextChunk rows")
            by_identifier[identifier] = row
        if set(by_identifier) != set(novel_ids):
            raise ValueError("graph expansion chunk rows changed during validation")
        graph_rows = [by_identifier[identifier] for identifier in novel_ids]
        return baseline + graph_rows, _graph_diagnostics(
            started_at=started_at,
            seed_count=result_diagnostics.seed_count,
            candidate_count=len(graph_rows),
            status="hit",
            algorithm_signature=result_diagnostics.algorithm_signature,
            version_signature=result_diagnostics.graph_version_signature,
        )
    except Exception:
        return baseline, _graph_diagnostics(
            started_at=started_at,
            seed_count=seed_count,
            candidate_count=0,
            status="error",
            algorithm_signature=algorithm_signature,
            version_signature=None,
        )


def text_chunk_search(model_cls: type[TextChunk], query: str, top_k: int, docs: list):
    from apps.documents.services import rag_cache
    from aquillm.utils import get_embedding
    from lib.embeddings.config import get_local_embed_config

    total_start = perf_counter()
    try:
        overlay_enabled = bool(getattr(django_settings, "KG_OVERLAY_ENABLED", False))
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
        )
        vector_ms = (perf_counter() - vector_started) * 1000
        combined_candidates = list(snapshot.baseline_candidates)
        graph_diagnostics: dict[str, object] = {}
        if overlay_enabled:
            combined_candidates, graph_diagnostics = _apply_graph_overlay(
                model_cls,
                snapshot,
                graph_scope,
                graph_config,
                preflight_status=graph_preflight_status,
            )

        if len(combined_candidates) <= top_k:
            reranked_results = _fallback_rerank(
                model_cls,
                combined_candidates,
                top_k,
            )
            rerank_ms = 0.0
        else:
            rerank_start = perf_counter()
            reranked_results = rerank_chunks(
                model_cls,
                query,
                combined_candidates,
                top_k,
            )
            rerank_ms = (perf_counter() - rerank_start) * 1000

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


__all__ = ["text_chunk_search"]
