"""Exact successful window inputs and truthful versioned max aggregation."""

from dataclasses import dataclass
from math import isfinite
from time import monotonic

from .chunk_rerank_results import fingerprint_pair
from .chunk_rerank_windows import WindowPlan


@dataclass(frozen=True)
class SuccessfulWindow:
    window_id: str
    pair: tuple[str, str]
    value: float

    @property
    def pair_fingerprint(self):
        return fingerprint_pair(*self.pair)


@dataclass(frozen=True)
class WindowScoreSet:
    plan: WindowPlan
    successful: tuple[SuccessfulWindow, ...]
    attempted_pairs: int


@dataclass(frozen=True)
class ChunkWindowScore:
    value: float | None
    prepared_input_scoring_coverage: str
    effective_pair_fingerprint: str
    aggregation_version: str = "window-max-v1"


class _AttemptBudget:
    """Invocation-local diagnostics; all admission remains on the shared ledger."""

    def __init__(self, budget):
        self.budget, self.attempts = budget, 0

    def __getattr__(self, name):
        return getattr(self.budget, name)

    def start_pair(self, *, phase):
        admitted = self.budget.start_pair(phase=phase)
        self.attempts += int(admitted)
        return admitted


def score_window_plan(
    plan, *, scorer, budget, phase="acquisition", timeout_seconds=3.0, clock=monotonic
):
    """Complete candidates in rank order, windows in source order.

    Providers with ``score_budgeted_pair`` own transport leases (including retries).
    Simple nonretrying scorers get a single lease here. No ledger means no inference.
    """
    successful = []
    deadline = clock() + timeout_seconds
    if budget is None or plan.preparation_coverage != "complete":
        return WindowScoreSet(plan, (), 0)
    budget = _AttemptBudget(budget)
    for window_id, window in zip(plan.required_window_ids, plan.windows):
        remaining = min(deadline - clock(), budget.scoring_remaining_ms(phase) / 1000)
        if remaining <= 0:
            break
        pair = (plan.query, window.text)
        try:
            if hasattr(scorer, "score_budgeted_pair"):
                result = scorer.score_budgeted_pair(
                    pair, remaining, budget=budget, phase=phase
                )
            else:
                if not budget.start_pair(phase=phase):
                    break
                try:
                    result = scorer.score_pair(pair, remaining)
                finally:
                    budget.finish_pair()
            if result is None:
                break
            value, actual = result
            if (
                type(value) not in (int, float)
                or not isfinite(value)
                or type(actual) is not tuple
                or len(actual) != 2
                or any(type(x) is not str for x in actual)
            ):
                break
            item = SuccessfulWindow(window_id, actual, float(value))
            if clock() >= deadline or not budget.publish(
                lambda: successful.append(item)
            ):
                break
        except (TimeoutError, ValueError, TypeError):
            break
    return WindowScoreSet(plan, tuple(successful), budget.attempts)


def aggregate_window_scores(scores):
    plan = scores.plan
    expected = tuple(
        (wid, (plan.query, window.text))
        for wid, window in zip(plan.required_window_ids, plan.windows)
    )
    actual = tuple((item.window_id, item.pair) for item in scores.successful)
    coverage = plan.preparation_coverage
    if coverage == "complete" and (actual != expected or not expected):
        coverage = "partial"
    from lib.evidence_observation import publish

    publish(
        "window_score",
        {
            "windows": len(expected),
            "score": max((s.value for s in scores.successful), default=None),
            "coverage": coverage,
            "attempted_pairs": scores.attempted_pairs,
        },
    )
    return ChunkWindowScore(
        max((s.value for s in scores.successful), default=None),
        coverage,
        plan.fingerprint,
    )
