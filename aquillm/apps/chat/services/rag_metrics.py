"""Structured metrics for direct RAG turns."""

from __future__ import annotations

import re
from math import isfinite

import structlog

from lib.llm.providers.request_observability import safe_correlation_id

logger = structlog.stdlib.get_logger(__name__)

_GRAPH_STATUSES = frozenset({"miss", "hit", "timeout", "error"})
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_MAX_GRAPH_METRIC_MS = 60_000.0
_MAX_GRAPH_SEEDS = 64
_MAX_GRAPH_CANDIDATES = 20
_SELECTION_MODES = frozenset({"legacy", "shadow", "adaptive"})
_FALLBACK_REASONS = frozenset(
    {
        "scorer_unavailable",
        "new_scores_disabled",
        "score_incompatible",
        "deadline",
        "incomplete_scores",
    }
)
_PROFILE_VERSIONS = frozenset({"adaptive-evidence-v1-minmax-1e-9"})
_PROFILE_NAMES = frozenset({"focused", "balanced", "breadth"})
_SCORE_STATUSES = frozenset({"model", "rank_fallback"})
_CONFIG_ERRORS = frozenset({"invalid_evidence_selection_configuration"})


def _safe_graph_ms(value: object) -> float | None:
    if type(value) not in (int, float):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    if not isfinite(number) or not 0.0 <= number <= _MAX_GRAPH_METRIC_MS:
        return None
    return round(number, 1)


def _safe_graph_count(value: object, *, maximum: int) -> int | None:
    if type(value) is not int or not 0 <= value <= maximum:
        return None
    return value


def _safe_graph_status(value: object) -> str | None:
    if type(value) is not str or value not in _GRAPH_STATUSES:
        return None
    return value


def _safe_graph_signature(value: object) -> str | None:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        return None
    return value


def log_direct_rag_turn(
    *,
    correlation_id: str | None = None,
    intent_ms: float,
    query_ms: float,
    retrieval_ms: float,
    evidence_ms: float,
    synthesis_ms: float,
    persistence_ms: float = 0.0,
    total_ms: float,
    retrieved_count: int,
    retained_count: int | None = None,
    retrieval_status: str,
    graph_ms: float | None = None,
    graph_seed_count: int | None = None,
    graph_candidate_count: int | None = None,
    graph_status: str | None = None,
    graph_algorithm_signature: str | None = None,
    graph_version_signature: str | None = None,
    selection_mode: str | None = None,
    selector_ms: float | None = None,
    final_scoring_ms: float | None = None,
    candidate_count: int | None = None,
    selected_doc_count: int | None = None,
    estimated_tokens: int | None = None,
    reused_pairs: int | None = None,
    new_pairs: int | None = None,
    profile_version: str | None = None,
    fixed_fallback_reason: str | None = None,
    selection_config_error: str | None = None,
    proposed_selected_count: int | None = None,
    proposed_selected_doc_count: int | None = None,
    proposed_estimated_tokens: int | None = None,
    proposed_profile_name: str | None = None,
    proposed_score_status: str | None = None,
) -> None:
    """Emit a structlog ``rag_direct_turn`` event with per-stage timing fields."""
    fields = {
        "correlation_id": safe_correlation_id(correlation_id),
        "intent_ms": round(intent_ms, 1),
        "query_ms": round(query_ms, 1),
        "retrieval_ms": round(retrieval_ms, 1),
        "evidence_ms": round(evidence_ms, 1),
        "synthesis_ms": round(synthesis_ms, 1),
        "persistence_ms": round(persistence_ms, 1),
        "total_ms": round(total_ms, 1),
        "retrieved_count": retrieved_count,
        "retained_count": (
            retrieved_count if retained_count is None else retained_count
        ),
        "retrieval_status": retrieval_status,
    }
    optional_graph_fields = {
        "graph_ms": _safe_graph_ms(graph_ms),
        "graph_seed_count": _safe_graph_count(
            graph_seed_count,
            maximum=_MAX_GRAPH_SEEDS,
        ),
        "graph_candidate_count": _safe_graph_count(
            graph_candidate_count,
            maximum=_MAX_GRAPH_CANDIDATES,
        ),
        "graph_status": _safe_graph_status(graph_status),
        "graph_algorithm_signature": _safe_graph_signature(graph_algorithm_signature),
        "graph_version_signature": _safe_graph_signature(graph_version_signature),
    }
    fields.update(
        {
            key: value
            for key, value in optional_graph_fields.items()
            if value is not None
        }
    )
    selection_fields = {
        "selection_mode": selection_mode
        if type(selection_mode) is str and selection_mode in _SELECTION_MODES
        else None,
        "selector_ms": _safe_graph_ms(selector_ms),
        "final_scoring_ms": _safe_graph_ms(final_scoring_ms),
        "candidate_count": _safe_graph_count(candidate_count, maximum=45),
        "selected_doc_count": _safe_graph_count(selected_doc_count, maximum=15),
        "estimated_tokens": _safe_graph_count(estimated_tokens, maximum=100_000),
        "reused_pairs": _safe_graph_count(reused_pairs, maximum=45),
        "new_pairs": _safe_graph_count(new_pairs, maximum=45),
        "profile_version": profile_version
        if (type(profile_version) is str and profile_version in _PROFILE_VERSIONS)
        else None,
        "fixed_fallback_reason": fixed_fallback_reason
        if (
            type(fixed_fallback_reason) is str
            and fixed_fallback_reason in _FALLBACK_REASONS
        )
        else None,
        "selection_config_error": selection_config_error
        if type(selection_config_error) is str
        and selection_config_error in _CONFIG_ERRORS
        else None,
        "proposed_selected_count": _safe_graph_count(
            proposed_selected_count, maximum=15
        ),
        "proposed_selected_doc_count": _safe_graph_count(
            proposed_selected_doc_count,
            maximum=15,
        ),
        "proposed_estimated_tokens": _safe_graph_count(
            proposed_estimated_tokens,
            maximum=100_000,
        ),
        "proposed_profile_name": proposed_profile_name
        if type(proposed_profile_name) is str
        and proposed_profile_name in _PROFILE_NAMES
        else None,
        "proposed_score_status": proposed_score_status
        if type(proposed_score_status) is str
        and proposed_score_status in _SCORE_STATUSES
        else None,
    }
    fields.update(
        {key: value for key, value in selection_fields.items() if value is not None}
    )
    logger.info("rag_direct_turn", **fields)


__all__ = ["log_direct_rag_turn"]


def log_preservation_turn(runtime, acquired, packet, timings):
    """Aggregate-only turn ledger observations, never identifiers or source text."""
    budget = runtime.budget
    lease = budget._synthesis
    logger.info(
        "obs.rag.preservation_turn",
        rounds=budget.actions_used,
        planner_calls=budget.planner_calls,
        budget_stop_reason=acquired.stop_reason,
        coverage_state=acquired.assessment.certainty,
        unresolved_count=len(
            getattr(packet.coverage_assessment, "unresolved_aspects", ())
        ),
        sources_admitted=budget.sources_used,
        acquisition_pairs=budget.pairs_used["acquisition"],
        final_pairs=budget.pairs_used["final"],
        materialized_codepoints=budget.text_used["materialized"],
        tokenized_codepoints=budget.text_used["tokenized"],
        selected_count=len(packet.chunks),
        synthesis_dispatches=lease.calls if lease else 0,
        **{name: round(value, 1) for name, value in timings.items()},
    )
