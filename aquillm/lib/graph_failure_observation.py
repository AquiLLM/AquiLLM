"""Fixed-label graph failure observations; no exception or source serialization."""

VERSION = "native-graph-failure-v1"
TYPES = frozenset({
    "graph_limit", "ready_scope_error", "source_scope_unavailable", "source_preparation_limited",
    "database_error", "timeout_error", "value_error", "type_error",
    "runtime_error", "assertion_error", "unexpected_error",
})
REASONS = frozenset({
    "unclassified", "deadline", "frontier_budget", "source_deadline",
    "metadata_budget", "original_scope_unavailable", "source_scope_unavailable",
    "captured_source_changed", "source_runtime_unavailable", "selection_identity_changed",
    "materialization_authority_changed", "hydrated_evidence_changed", "final_source_changed",
    "candidate_pool_contract", "invalid_graph_timeout", "statement_cancelled",
    "transaction_aborted", "deadlock", "serialization_failure",
    "readiness_mismatch", "authorization_context_invalid", "captured_authority_changed",
    "authorization_signature_changed", "materialization_source_changed", "seed_unavailable",
})
PHASES = frozenset({
    "preparation", "authorization", "authorization_cleanup", "ready_scope",
    "graph_execution", "graph_result", "materialization", "source_revalidation",
    "baseline_top_up", "pool_ordering", "authority_revalidation", "seed_revalidation",
    "final_source_validation", "database_cleanup", "preparation_cleanup", "publication",
})


def safe_graph_failure(raw):
    """Reject unknown fields/types without inspecting or stringifying objects."""
    if (type(raw) is not dict or len(raw) != 5
            or any(type(key) is not str for key in raw)
            or set(raw) != {"version", "type", "reason", "phase", "owner_valid"}):
        return None
    for key, allowed in (("version", {VERSION}), ("type", TYPES),
                         ("reason", REASONS), ("phase", PHASES)):
        if type(raw[key]) is not str or raw[key] not in allowed:
            return None
    if type(raw["owner_valid"]) is not bool:
        return None
    return {key: raw[key] for key in ("version", "type", "reason", "phase", "owner_valid")}
