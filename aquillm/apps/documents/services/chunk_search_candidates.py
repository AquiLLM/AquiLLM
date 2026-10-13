"""Reusable baseline candidate acquisition and graph-seed preparation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from time import perf_counter
from typing import TYPE_CHECKING, Any
from uuid import UUID

import structlog
from django.apps import apps
from django.db.models.base import ModelBase

from apps.documents.services.chunk_search_graph_seeds import (
    build_graph_seeds as _build_graph_seeds,
)
from apps.documents.services.chunk_search_limits import _candidate_limits
from apps.documents.services.hybrid_graph_authorization import (
    documents_match_retrieval_authorization,
)
from lib.retrieval_redaction import RetrievalLogReason, retrieval_log_fields

from .chunk_search_exact_terms import _exact_term_query as _exact_term_query
from .chunk_search_exact_terms import _salient_exact_terms as _salient_exact_terms
from .chunk_search_scope import current_scope, freeze_search_scope
from .chunk_search_scoped_rows import dense_rows

if TYPE_CHECKING:
    from apps.knowledge_graph.retrieval import (
        GraphExpansionConfig,
        GraphExpansionSeed,
    )

from .chunk_search_rank_policy import ordered_candidates
from .lexical_collection import collect_lexical

logger = structlog.stdlib.get_logger(__name__)

_DATABASE_ID_MAX = 2**63 - 1


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
    effective_limits: object | None = None
    full_text_results: tuple[object, ...] = ()
    full_text_chunk_ids: tuple[int, ...] = ()
    full_text_coverage: object | None = None
    fusion_v2: object | None = None


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
    authorization_context: object | None = None,
) -> HybridCandidateSnapshot:
    """Acquire vector/trigram/exact rows once and prepare the pre-rerank pool."""

    documents_snapshot = tuple(documents)
    scope = current_scope()
    if scope is None and isinstance(model_cls, ModelBase):
        scope = freeze_search_scope(
            documents_snapshot,
            using=model_cls.objects.db,
            authorization=authorization_context,
        )
        from lib.embeddings.document_contracts import digest

        scope = replace(
            scope,
            bodies={},
            dense_rows={},
            lexical_ids=set(),
            fts_rows={},
            fts_invalid_ids=set(),
            fusion_v2=None,
            query_digest=digest({"version": "unprepared-query-v1", "query": query}),
        )

    def candidates():
        queryset = model_cls.objects.filter_by_documents(documents_snapshot)
        if scope is None:
            return queryset
        with scope.database():
            runtime = scope.runtime
            original = getattr(runtime, "_frozen_retrieval_scope", None)
            # The eager source loader supplies this identical original SQL fence.
            if (
                original is not None
                and original.runtime is runtime
                and original.representation is scope.representation
                and scope.authorization is not None
                and original.authorization is scope.authorization
                and scope.authorization is runtime.authorization
                and original.using == scope.using
                and scope.using == runtime.authorization.database_alias
                and scope.requested_document_ids <= original.requested_document_ids
            ):
                scope.validate()
                return queryset.using(scope.using).filter(
                    doc_id__in=scope.requested_document_ids
                )
            return scope.queryset(queryset.using(scope.using))

    if (
        authorization_context is not None
        and not documents_match_retrieval_authorization(
            documents_snapshot, authorization_context
        )
    ):
        raise ValueError("candidate documents differ from frozen authorization")
    limits = _candidate_limits(
        query,
        top_k,
        app_config_getter=app_config_getter or apps.get_app_config,
    )
    vector_error = (
        RetrievalLogReason.EMBEDDING_UNAVAILABLE.value
        if initial_vector_error is not None
        else None
    )
    vector_started = perf_counter()
    if query_embedding is None:
        vector_rows: tuple[object, ...] = ()
    else:
        try:
            vector_rows = dense_rows(model_cls, query_embedding, scope, limits.vector)
        except Exception:
            vector_error = RetrievalLogReason.UPSTREAM_UNAVAILABLE.value
            logger.warning(
                "obs.rag.vector_search_failed",
                **retrieval_log_fields(
                    reason=RetrievalLogReason.UPSTREAM_UNAVAILABLE,
                    count=0,
                    elapsed_ms=0.0,
                ),
            )
            vector_rows = ()
    vector_ms = (perf_counter() - vector_started) * 1000

    (
        trigram_rows,
        exact_rows,
        full_text_rows,
        exact_terms,
        trigram_ms,
        exact_ms,
        coverage,
    ) = collect_lexical(model_cls, candidates, scope, query, limits)
    all_rows = (*vector_rows, *trigram_rows, *exact_rows, *full_text_rows)
    baseline: list[object] = []
    seen_ids: set[object] = set()
    for candidate in all_rows:
        identifier = getattr(candidate, "pk", None)
        if identifier in seen_ids:
            continue
        seen_ids.add(identifier)
        baseline.append(candidate)

    baseline, fusion = ordered_candidates(
        baseline,
        {
            "dense": tuple(row.pk for row in vector_rows),
            "exact": tuple(row.pk for row in exact_rows),
            "trigram": tuple(row.pk for row in trigram_rows),
            "full_text": tuple(row.pk for row in full_text_rows),
        },
        query=query,
        exact_rows=exact_rows,
        scope=scope,
    )

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
        vector_results=vector_rows,
        trigram_results=trigram_rows,
        exact_results=exact_rows,
        vector_chunk_ids=tuple(getattr(row, "pk") for row in vector_rows),
        trigram_chunk_ids=tuple(getattr(row, "pk") for row in trigram_rows),
        exact_chunk_ids=tuple(getattr(row, "pk") for row in exact_rows),
        baseline_candidates=tuple(baseline),
        graph_seeds=graph_seeds,
        exact_terms=tuple(t.text for t in exact_terms),
        vector_error=vector_error,
        vector_ms=vector_ms,
        trigram_ms=trigram_ms,
        exact_ms=exact_ms,
        pre_dedupe_count=len(all_rows),
        graph_seed_error=graph_seed_error,
        effective_limits=limits,
        full_text_results=full_text_rows,
        full_text_chunk_ids=tuple(row.pk for row in full_text_rows),
        full_text_coverage=coverage,
        fusion_v2=fusion,
    )


__all__ = [
    "AuthorizedDocumentScope",
    "CandidateScopeLimit",
    "HybridCandidateSnapshot",
    "collect_hybrid_candidate_snapshot",
    "documents_match_retrieval_authorization",
    "freeze_authorized_document_scope",
]
