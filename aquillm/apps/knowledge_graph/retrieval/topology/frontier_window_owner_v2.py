"""Original-owner dispatch and authoritative replay of finite physical frames."""

from copy import copy, deepcopy
from hashlib import sha256

from apps.knowledge_graph.projection.frontier_reads_v2 import schema_error

from .frontier_contracts_v2 import FrontierBudgetExceeded
from .frontier_window_budget_v2 import preview_window
from .frontier_window_checkpoint_v2 import snapshot
from .frontier_window_contracts_v2 import (
    CheckpointTooLarge,
    CompletedWindowBudgetFailure,
    encode_request,
)


def replay(state, frames, grant):
    """Copy bounded operation containers, sharing only immutable source records."""
    staged = copy(state)
    staged.reads = copy(state.reads)
    staged.reads._index_records = dict(state.reads._index_records)
    for name in (
        "pending",
        "heads",
        "first",
        "wave",
        "staged",
        "expected",
        "raw_pending",
        "guard",
    ):
        if hasattr(state, name):
            setattr(staged, name, deepcopy(getattr(state, name)))
    staged.fresh = {g: dict(rows) for g, rows in state.fresh.items()}
    if hasattr(state, "session"):
        staged.session = copy(state.session)
        staged.session.reads = staged.reads
        staged.entries = {
            pair: list(entries) for pair, entries in state.entries.items()
        }
        staged.prepared = {
            pair: list(values) for pair, values in state.prepared.items()
        }
    prior = 0
    for kind, rows in frames:
        planned = staged.next_read(grant.planning_rows - prior)
        if planned is None or kind != planned.kind or len(rows) > planned.maximum:
            raise schema_error()
        staged.apply_result(rows)
        prior += len(rows)
    try:
        staged.next_read(grant.planning_rows - prior)
    except FrontierBudgetExceeded:
        # No additional physical read is required to validate this completed prefix.
        # The owner will encounter the same exhaustion when requesting further work.
        pass
    state.reads._check_index_deadline()
    return staged, prior


def publish(state, staged):
    state.reads._index_records = staged.reads._index_records
    for name, value in staged.__dict__.items():
        if name not in {"reads", "session"}:
            setattr(state, name, value)


def run_preparation(state):
    reads, budget = state.reads, state.reads.budget
    while (effect := state.next_read()) is not None:
        execute = getattr(reads.driver, "execute_frontier_window", None)
        if not callable(execute) or not getattr(
            reads.driver, "preparation_windows", False
        ):
            state.apply_result(reads.read_batch(effect.kind, effect.pairs))
            continue
        operation, checkpoint = snapshot(state)
        proposed = preview_window(budget)
        sequence = getattr(reads, "_window_sequence", 0) + 1
        try:
            body = encode_request(
                reads.ready,
                operation,
                checkpoint,
                proposed,
                sequence,
                reads._check_index_deadline,
            )
        except CheckpointTooLarge:
            state.apply_result(reads.read_batch(effect.kind, effect.pairs))
            continue
        grant = budget.reserve_window()
        try:
            if grant != proposed:
                raise ValueError("frontier reservation changed during preflight")
            reads._window_sequence = sequence
            frames = execute(
                body=body, grant=grant, deadline=reads.deadline, budget=budget
            )
            staged, scanned = replay(state, frames, grant)
            budget.settle_window(grant, queries=len(frames), scanned_rows=scanned)
            publish(state, staged)
        except CompletedWindowBudgetFailure as exc:
            if exc.request_digest != sha256(body).hexdigest():
                budget.stop_uncertain()
                raise schema_error() from exc
            budget.settle_window(
                grant, queries=exc.queries, scanned_rows=exc.scanned_rows
            )
            raise
        except BaseException:
            budget.stop_uncertain()
            raise
    return state.result
