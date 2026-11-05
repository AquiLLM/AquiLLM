"""Bounded native failure labels, independent of recovery and authority decisions."""

import structlog

from lib.graph_failure_observation import VERSION, safe_graph_failure

logger = structlog.stdlib.get_logger(__name__)
_SOURCE_REASONS = {
    "graph_original_scope_unavailable": "original_scope_unavailable",
    "source_admission_original_scope_unavailable": "original_scope_unavailable",
    "source_scope_unavailable": "source_scope_unavailable",
    "source_admission_captured_row_changed": "captured_source_changed",
    "source_runtime_unavailable": "source_runtime_unavailable",
    "source_admission_seed_unavailable": "seed_unavailable",
}
_VALIDATION_REASONS = {
    "V2 replaced captured selection": "selection_identity_changed",
    "V2 final materialization lost exact authority": "materialization_authority_changed",
    "V2 evidence changed after selection hydration": "hydrated_evidence_changed",
    "V2 final source fence changed": "final_source_changed",
    "captured baseline exceeds original pool contract": "candidate_pool_contract",
    "graph timeout is invalid": "invalid_graph_timeout",
    "captured authority changed": "captured_authority_changed",
    "authorization signature changed": "authorization_signature_changed",
    "materialization source identity changed": "materialization_source_changed",
}


def _classification(exc):
    from django.db import DatabaseError
    from apps.documents.services.representation_contracts import SourceScopeUnavailable
    from apps.documents.services.source_loading import SourcePreparationLimited
    from .closed_graph_attempt_v2 import _ClosedGraphLimited
    from .ready_scope import ReadyScopeError

    args = BaseException.args.__get__(exc, type(exc))
    message = args[0] if len(args) == 1 and type(args[0]) is str else None
    if type(exc) is _ClosedGraphLimited:
        return "graph_limit", limit_observations(exc).get("graph_failure_reason", "unclassified")
    if type(exc) is ReadyScopeError:
        return "ready_scope_error", message if message in {
            "readiness_mismatch", "authorization_context_invalid",
        } else "unclassified"
    if type(exc) is SourceScopeUnavailable:
        return "source_scope_unavailable", _SOURCE_REASONS.get(message, "unclassified")
    if type(exc) is SourcePreparationLimited:
        return "source_preparation_limited", {
            "source deadline": "source_deadline", "source preparation deadline": "source_deadline",
            "dense metadata allowance": "metadata_budget", "source_runtime_unavailable": "source_runtime_unavailable",
        }.get(message, "unclassified")
    if isinstance(exc, DatabaseError):
        cause = BaseException.__cause__.__get__(exc, type(exc))
        state = getattr(cause, "sqlstate", None) or getattr(cause, "pgcode", None)
        reason = {"57014": "statement_cancelled", "25P02": "transaction_aborted",
                  "40P01": "deadlock", "40001": "serialization_failure"}
        return "database_error", reason.get(state, "unclassified") if type(state) is str else "unclassified"
    for cls, label in ((TimeoutError, "timeout_error"), (ValueError, "value_error"),
                       (TypeError, "type_error"), (RuntimeError, "runtime_error"),
                       (AssertionError, "assertion_error")):
        if type(exc) is cls:
            return label, _VALIDATION_REASONS.get(message, "unclassified")
    return "unexpected_error", "unclassified"


def failure_diagnostics(exc, phase, *, owner_valid):
    """Observation failures must never replace the original graph outcome."""
    try:
        kind, reason = _classification(exc)
        record = safe_graph_failure(dict(version=VERSION, type=kind, reason=reason,
                                         phase=phase, owner_valid=owner_valid))
        if record is None:
            return {}
    except BaseException:
        return {}
    try:
        logger.warning("obs.rag.graph_failure", **record)
    except BaseException:
        pass
    return {"graph_failure": record}


def limit_observations(exc):
    from .closed_graph_attempt_v2 import _ClosedGraphLimited, _graph_observations
    from .graph_frontier_observation_v2 import (
        FrontierBudgetObservationV2,
        GraphFrontierInvocationV2,
    )
    from .graph_handoff import GraphCoverageV2

    try:
        if type(exc) is not _ClosedGraphLimited:
            return {}
        reason, raw = exc.reason, exc.observations
        if (type(reason) is not str or reason not in {"deadline", "frontier_budget"}
                or type(raw) is not dict):
            return {}
        snapshot = {}
        if raw:
            if (any(type(key) is not str for key in raw) or set(raw) != {
                    "graph_coverage_v2", "frontier_budget", "frontier_invocations"}):
                return {}
            invocations = raw["frontier_invocations"]
            if type(invocations) is not list or len(invocations) != 1:
                return {}
            invocation = invocations[0]
            if (type(invocation) is not dict
                    or any(type(key) is not str for key in invocation)):
                return {}
            coverage = raw["graph_coverage_v2"]
            budgets = (raw["frontier_budget"], invocation["frontier_budget"])
            for value in (coverage, invocation, *budgets):
                if type(value) is not dict or any(type(k) is not str for k in value):
                    return {}
            coverage = GraphCoverageV2(**coverage)
            copies = []
            for budget in budgets:
                copied_invocation = GraphFrontierInvocationV2(**{
                    **invocation,
                    "frontier_budget": FrontierBudgetObservationV2(**budget),
                })
                copies.append(_graph_observations(coverage, copied_invocation))
            if copies[0] != copies[1]:
                return {}
            snapshot = copies[0]
        return {**snapshot, "graph_failure_reason": reason}
    except Exception:
        return {}  # Malformed/forged exceptions remain generic graph errors.
