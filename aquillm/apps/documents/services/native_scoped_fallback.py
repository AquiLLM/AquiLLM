"""A bounded baseline read after an optional graph child reaches its deadline."""

from contextlib import contextmanager

from django.db import OperationalError

from .chunk_search_scope import validate_requested_authorization
from .graph_admission_read import read_captured_refs
from .native_search_capture import _CapturedPhaseExpired
from .native_search_schedule import NATIVE_CLOSURE_RESERVE_MS, SCOPED_BASELINE_RESERVE_MS
from .source_deadline import _DEADLINE, source_preparation_scope, source_remaining_ms
from .source_loading import SourcePreparationLimited


def scoped_fallback_window(captured):
    """Capture the cutoff before graph work on the original clock and allowance."""
    scope = captured.scope
    if (len(scope.requested_document_ids) != 1
            or scope.requested_document_ids == scope.authorization.selected_document_ids
            or not scope.runtime.budget.remaining_source_capacity()
            or not (captured.admission.references.keys() - set(captured.admission.initial_ids))):
        return None
    parent = captured.parent_deadline
    clock = parent[1] if parent is not None else captured.operation.clock
    now = clock()
    free_ms = source_remaining_ms(scope.runtime.budget.remaining_ms()) - NATIVE_CLOSURE_RESERVE_MS
    if free_ms <= SCOPED_BASELINE_RESERVE_MS:
        return None
    return clock, now + free_ms / 1000


def recover_scoped_baseline(captured, window):
    """A fresh eager read; never retry the graph, re-rank or refund any charge."""
    clock, cutoff = window
    started = clock()
    deadline = min(cutoff, started + SCOPED_BASELINE_RESERVE_MS / 1000)
    plan = captured.admission
    initial = captured.baseline
    if (deadline - started) * 1000 < 1:
        return initial, 0
    ids = plan.top_up_ids()
    if not ids:
        return initial, 0
    phase, restored = None, False

    def check():
        captured.check(phase_deadline=phase, restored=restored)

    @contextmanager
    def prepare():
        # Only pre-publication work is allowed to convert expiry into an
        # initial-only result. A later hard owner failure invalidates the search.
        nonlocal phase, restored
        try:
            with source_preparation_scope(deadline, clock, captured.operation.budget):
                phase = _DEADLINE.get()
                check()
                with captured.scope.database():
                    validate_requested_authorization(
                        captured.scope.runtime, captured.authorize.authorization, captured.snapshot.documents
                    )
                check()
                yield
        except (SourcePreparationLimited, TimeoutError, OperationalError) as exc:
            captured.check()
            if isinstance(exc, OperationalError) and getattr(exc.__cause__, "pgcode", None) != "57014":
                raise
            if (deadline - clock()) * 1000 >= 1:
                raise
            raise _CapturedPhaseExpired("scoped baseline deadline") from exc
        finally:
            restored = True

    try:
        rows = read_captured_refs(
            plan, ids, baseline=initial, order=True, owner_check=check, preparation=prepare,
        )
    except _CapturedPhaseExpired:
        captured.check()
        return initial, 0
    return rows, len({r.pk for r in rows} - set(plan.initial_ids))
