"""Input contracts for the shared chunk materialization seam."""

from __future__ import annotations

from apps.documents.services.chunk_rerank import _STRICT_EVALUATION_RERANK
from apps.documents.services.chunk_search_candidates import CandidateScopeLimit


def candidate_identifier(candidate: object) -> int:
    identifier = getattr(candidate, "pk", None)
    if type(identifier) is not int or identifier <= 0:
        raise ValueError("candidate rows require positive integer primary keys")
    return identifier


def candidate_identity(candidate: object) -> tuple[str, int]:
    """Use a durable PK when present, otherwise exact in-memory identity."""
    identifier = getattr(candidate, "pk", None)
    if type(identifier) is int and identifier > 0:
        return ("pk", identifier)
    return ("object", id(candidate))


def validate_candidate_request(
    query: str,
    top_k: int,
    baseline_candidates: tuple[object, ...],
    graph_chunk_ids: tuple[int, ...],
    max_graph_candidates: int,
    force_complete_rerank: bool,
    capture_scores: bool,
    eval_rerank_capability: object | None,
) -> tuple[tuple[str, int], ...]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be nonempty")
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive exact integer")
    if type(baseline_candidates) is not tuple:
        raise ValueError("baseline_candidates must be an exact tuple")
    identities = tuple(candidate_identity(row) for row in baseline_candidates)
    if len(set(identities)) != len(identities):
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
    if type(capture_scores) is not bool:
        raise ValueError("capture_scores must be an exact bool")
    if (
        eval_rerank_capability is not None
        and eval_rerank_capability is not _STRICT_EVALUATION_RERANK
    ):
        raise ValueError("invalid strict eval rerank capability")
    if len(graph_chunk_ids) > max_graph_candidates:
        raise CandidateScopeLimit("graph candidate materialization exceeds its cap")
    return identities


__all__ = ["candidate_identifier", "candidate_identity", "validate_candidate_request"]
