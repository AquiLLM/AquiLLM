"""Legacy single-branch graph overlay kept behind the existing feature flag."""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

from apps.documents.services.chunk_search_candidates import HybridCandidateSnapshot

if TYPE_CHECKING:
    from apps.documents.models.chunks import TextChunk

EVALUATION_GRAPH_FAILURE = object()
EVALUATION_GRAPH_MISS = object()


def graph_diagnostics(
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


def apply_graph_overlay(
    model_cls: type[TextChunk],
    snapshot: HybridCandidateSnapshot,
    scope: object | None,
    graph_config: object | None,
    *,
    preflight_status: str | None,
    _eval_failure_capability: object | None = None,
) -> tuple[tuple[int, ...], dict[str, object]]:
    """Return bounded graph IDs; the shared seam permission-refetches them."""

    del model_cls
    if _eval_failure_capability is not None and _eval_failure_capability not in {
        EVALUATION_GRAPH_FAILURE,
        EVALUATION_GRAPH_MISS,
    }:
        raise ValueError("invalid graph failure evaluation capability")
    if _eval_failure_capability is EVALUATION_GRAPH_MISS:
        if preflight_status != "miss":
            raise ValueError("graph miss evaluation requires miss preflight")
    elif preflight_status is not None and _eval_failure_capability is not None:
        raise ValueError("graph failure evaluation cannot set preflight status")

    started_at = perf_counter()
    baseline = list(snapshot.baseline_candidates)
    algorithm_signature = getattr(graph_config, "algorithm_signature", None)
    seed_count = len(snapshot.graph_seeds)
    if preflight_status is not None:
        return (), graph_diagnostics(
            started_at=started_at,
            seed_count=seed_count,
            candidate_count=0,
            status=preflight_status,
            algorithm_signature=algorithm_signature,
            version_signature=None,
        )
    if snapshot.graph_seed_error:
        return (), graph_diagnostics(
            started_at=started_at,
            seed_count=0,
            candidate_count=0,
            status="error",
            algorithm_signature=algorithm_signature,
            version_signature=None,
        )
    if not snapshot.graph_seeds:
        return (), graph_diagnostics(
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
        if _eval_failure_capability is EVALUATION_GRAPH_FAILURE:
            raise RuntimeError("forced evaluation of production fail-open composition")
        result = expand_chunk_candidates(request)
        if type(result) is not GraphExpansionResult:
            raise ValueError("graph expansion returned an invalid result")
        result_diagnostics = result.diagnostics
        if result_diagnostics.algorithm_signature != graph_config.algorithm_signature:
            raise ValueError("graph expansion used a different algorithm config")
        if result_diagnostics.status != "hit":
            return (), graph_diagnostics(
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
            return (), graph_diagnostics(
                started_at=started_at,
                seed_count=result_diagnostics.seed_count,
                candidate_count=0,
                status="miss",
                algorithm_signature=result_diagnostics.algorithm_signature,
                version_signature=result_diagnostics.graph_version_signature,
            )

        return novel_ids, graph_diagnostics(
            started_at=started_at,
            seed_count=result_diagnostics.seed_count,
            candidate_count=len(novel_ids),
            status="hit",
            algorithm_signature=result_diagnostics.algorithm_signature,
            version_signature=result_diagnostics.graph_version_signature,
        )
    except Exception:
        return (), graph_diagnostics(
            started_at=started_at,
            seed_count=seed_count,
            candidate_count=0,
            status="error",
            algorithm_signature=algorithm_signature,
            version_signature=None,
        )


__all__ = [
    "EVALUATION_GRAPH_FAILURE",
    "EVALUATION_GRAPH_MISS",
    "apply_graph_overlay",
    "graph_diagnostics",
]
