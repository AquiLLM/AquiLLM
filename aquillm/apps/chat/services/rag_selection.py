"""Pure bounded greedy selector for authorized, hydrated evidence candidates."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable

from apps.chat.services.rag_selection_similarity import (
    prepare_snippet,
    prepared_redundancy,
)
from apps.chat.services.rag_selection_types import (
    EvidenceSelection,
    SelectionCandidate,
    SelectionLimits,
    SelectionProfile,
)

MAX_CANDIDATES = 45


def _token_cost(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def candidate_token_cost(candidate):
    return (
        candidate.token_cost
        if candidate.token_cost is not None
        else _token_cost(candidate.text)
    )


def _validate(
    profile: SelectionProfile,
    limits: SelectionLimits,
    candidates: tuple[SelectionCandidate, ...],
) -> None:
    if (
        not profile.name
        or not profile.version
        or isinstance(profile.relevance_weight, bool)
        or isinstance(profile.gap_allowance, bool)
        or not math.isfinite(profile.relevance_weight)
        or not math.isfinite(profile.gap_allowance)
        or not 0 <= profile.relevance_weight <= 1
        or not 0 <= profile.gap_allowance <= 1
    ):
        raise ValueError("invalid evidence selection profile")
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (
                limits.max_passages,
                limits.max_per_document,
                limits.token_budget,
            )
        )
        or limits.max_passages < 0
        or limits.max_per_document <= 0
        or limits.token_budget < 0
    ):
        raise ValueError("invalid evidence selection limits")
    if len(candidates) > MAX_CANDIDATES:
        raise ValueError("evidence candidate union exceeds the 45-passage cap")
    if any(
        isinstance(c.relevance, bool)
        or not isinstance(c.relevance, (int, float))
        or not math.isfinite(c.relevance)
        or not 0 <= c.relevance <= 1
        for c in candidates
    ):
        raise ValueError("candidate relevance must be finite normalized scores")


def _identity(candidate: SelectionCandidate) -> tuple[str, int, int]:
    return candidate.doc_id, candidate.chunk_number, candidate.chunk_id


def _tie_break(candidate: SelectionCandidate) -> tuple[float, int, str, int, int]:
    return (-candidate.relevance, candidate.fused_rank, *_identity(candidate))


def select_evidence(
    candidates: Iterable[SelectionCandidate],
    *,
    profile: SelectionProfile,
    limits: SelectionLimits,
    score_status: str,
    mode: str = "adaptive",
) -> EvidenceSelection:
    """Choose feasible passages by score and marginal textual novelty.

    The single loop enforces text cost, passage count, and per-document cap.
    Similarity only affects ordering inside the relevance-gap frontier.
    """
    pool = tuple(candidates)
    from lib.evidence_observation import publish

    publish("final_selection", {"mode": mode, "candidates": len(pool)})
    _validate(profile, limits, pool)
    if mode not in ("legacy", "adaptive"):
        raise ValueError("unsupported selector mode")

    # Verified document/chunk coordinates identify one source passage. Preserve
    # the strongest rendition of a duplicate, independently of input order.
    by_identity: dict[tuple[str, int, int], SelectionCandidate] = {}
    for candidate in pool:
        key = _identity(candidate)
        previous = by_identity.get(key)
        if previous is None or _tie_break(candidate) < _tie_break(previous):
            by_identity[key] = candidate
    remaining = list(by_identity.values())
    if mode == "legacy":
        from apps.chat.services.rag_legacy_selection import diversify_evidence_chunks

        ordered = diversify_evidence_chunks(
            [
                {"doc_id": c.doc_id, "candidate": c}
                for c in sorted(remaining, key=lambda c: c.fused_rank)
            ],
            MAX_CANDIDATES,
        )
        legacy_order = {_identity(row["candidate"]): i for i, row in enumerate(ordered)}
    features = {
        identity: prepare_snippet(candidate.text)
        for identity, candidate in by_identity.items()
    }
    selected: list[SelectionCandidate] = []
    document_counts: dict[str, int] = defaultdict(int)
    # Update each remaining candidate's maximum similarity once per pick.
    # At most 45 * 44 / 2 pairs are evaluated over the whole selection.
    maximum_redundancy: dict[tuple[str, int, int], float] = {
        identity: 0.0 for identity in by_identity
    }
    estimated_tokens = 0

    while len(selected) < limits.max_passages:
        available = limits.token_budget - estimated_tokens
        feasible = [
            candidate
            for candidate in remaining
            if document_counts[candidate.doc_id] < limits.max_per_document
            and candidate_token_cost(candidate) <= available
        ]
        if not feasible:
            break
        best_relevance = max(candidate.relevance for candidate in feasible)
        frontier = [
            candidate
            for candidate in feasible
            if candidate.relevance >= best_relevance - profile.gap_allowance
        ]

        def order(
            candidate: SelectionCandidate,
        ) -> tuple[float, float, int, str, int, int]:
            redundancy = maximum_redundancy[_identity(candidate)]
            value = (
                profile.relevance_weight * candidate.relevance
                - (1 - profile.relevance_weight) * redundancy
            )
            return (-value, *_tie_break(candidate))

        best = (
            min(feasible, key=lambda c: legacy_order[_identity(c)])
            if mode == "legacy"
            else min(frontier, key=order)
        )
        selected.append(best)
        estimated_tokens += candidate_token_cost(best)
        document_counts[best.doc_id] += 1
        remaining.remove(best)
        for candidate in remaining:
            identity = _identity(candidate)
            maximum_redundancy[identity] = max(
                maximum_redundancy[identity],
                prepared_redundancy(
                    features[identity],
                    features[_identity(best)],
                    same_document=candidate.doc_id == best.doc_id,
                ),
            )

    from .rag_selection_observation import observe_exclusions

    observe_exclusions(remaining, selected, document_counts, estimated_tokens, limits)
    return EvidenceSelection(tuple(selected), estimated_tokens, profile, score_status)
