"""Validate detached private coverage diagnostics without application imports."""

_FIELDS = frozenset(
    {"selected_documents", "ready_documents", "status", "reasons", "exhaustive"}
)
_REASONS = frozenset(
    {
        "not_ready",
        "incompatible_generation",
        "frontier_budget",
        "evidence_budget",
        "deadline",
    }
)


def coverage_reasons(coverage):
    """Return only exact, bounded V2 coverage reasons; malformed data is absent."""
    if (
        type(coverage) is not dict
        or len(coverage) != len(_FIELDS)
        or any(type(key) is not str for key in coverage)
        or coverage.keys() != _FIELDS
    ):
        return ()
    selected = coverage["selected_documents"]
    ready = coverage["ready_documents"]
    status = coverage["status"]
    reasons = coverage["reasons"]
    if (
        type(selected) is not int
        or not 0 <= selected <= 10000
        or type(ready) is not int
        or not 0 <= ready <= selected
        or type(status) is not str
        or status not in {"complete", "partial", "unavailable"}
        or type(reasons) is not tuple
        or len(reasons) > len(_REASONS)
        or any(type(reason) is not str or reason not in _REASONS for reason in reasons)
        or reasons != tuple(sorted(set(reasons)))
        or coverage["exhaustive"] is not False
    ):
        return ()
    if (
        (status == "complete" and (ready != selected or reasons))
        or (status == "partial" and (not ready or not reasons))
        or (status == "unavailable" and ready)
    ):
        return ()
    return reasons
