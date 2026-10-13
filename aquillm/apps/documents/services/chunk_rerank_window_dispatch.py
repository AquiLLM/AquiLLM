"""Bounded pointwise source work; preparation and publication stay on the caller."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextvars import copy_context
from threading import Event

from .chunk_rerank_config import rerank_score_concurrency
from .chunk_rerank_window_scores import score_window_plan


def _complete(result):
    plan = result.plan
    expected = tuple(
        (wid, (plan.query, window.text))
        for wid, window in zip(plan.required_window_ids, plan.windows)
    )
    return (
        plan.preparation_coverage == "complete"
        and bool(expected)
        and tuple((item.window_id, item.pair) for item in result.successful) == expected
    )


def score_source_plans(
    plans, *, provider, budget, state, phase, deadline, clock, on_submit=None
):
    """Return indexed complete results, or cancel pending work without joining I/O."""
    if not plans:
        return {}
    if any(plan.preparation_coverage != "complete" for plan in plans.values()):
        return None
    width = min(6, rerank_score_concurrency(), budget.limits.in_flight_pairs)
    if width == 0:
        return None
    stopped = Event()
    remaining = iter(plans.items())
    results, pending = {}, {}
    executor = ThreadPoolExecutor(max_workers=width, thread_name_prefix="window-source")

    def open_now():
        return not stopped.is_set() and clock() < deadline and state.can_admit()

    def score(plan):
        return score_window_plan(
            plan,
            scorer=provider,
            budget=budget,
            phase=phase,
            timeout_seconds=max(0, deadline - clock()),
            clock=clock,
            on_submit=on_submit,
            should_stop=stopped.is_set,
        )

    def fill():
        while len(pending) < width and open_now():
            item = next(remaining, None)
            if item is None:
                break
            index, plan = item
            pending[executor.submit(copy_context().run, score, plan)] = index

    try:
        fill()
        while pending and open_now():
            done, _ = wait(
                pending,
                timeout=min(0.05, max(0, deadline - clock())),
                return_when=FIRST_COMPLETED,
            )
            if not done:
                continue
            for future in done:
                index = pending.pop(future)
                result = future.result()
                if not _complete(result):
                    return None
                results[index] = result
            fill()
        return results if len(results) == len(plans) and open_now() else None
    finally:
        stopped.set()
        for future in pending:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
