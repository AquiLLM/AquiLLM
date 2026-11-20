"""Private child-deadline witnesses; diagnostic labels cannot authorize a read."""

from dataclasses import dataclass
from typing import Literal

from apps.documents.services.source_deadline import _DEADLINE
from lib.retrieval.operation import _OPERATION


class _ReadyAuthorityDeadline(TimeoutError):
    def __init__(self, bound, operation):
        super().__init__("graph authority child deadline")
        self.bound, self.operation = bound, operation


class _ClosedGraphLimited(TimeoutError):
    """Only a known graph limit and detached, validated observations escape."""

    def __init__(self, reason, *, coverage=None, invocation=None):
        from .closed_graph_attempt_v2 import _graph_observations

        if type(reason) is not str or reason not in {"deadline", "frontier_budget"}:
            raise ValueError("unsupported closed graph limit")
        super().__init__(reason)
        self.reason = reason
        self.bound, self.operation = _DEADLINE.get(), _OPERATION.get()
        self.observations = (
            {} if coverage is None and invocation is None
            else _graph_observations(coverage, invocation)
        )


@dataclass(frozen=True, slots=True)
class _BaselineCompletion:
    operation: object
    parent_deadline: object
    admission: object
    child_bound: object


@dataclass(frozen=True)
class _ClosedGraphAttemptResult:
    kind: Literal["success", "failure", "invalid"]
    pool: tuple | None
    diagnostics: dict
    admit_deferred: bool = False
    baseline_completion: _BaselineCompletion | None = None


def completion_after_deadline(exc, *, bound, operation, parent_deadline, admission,
                              owned_deadline=None):
    """Called only after the native graph owner unwinds and validates itself."""
    if operation is None or admission is None:
        return None
    if type(exc) is _ClosedGraphLimited:
        if exc.reason != "deadline":
            return None
        if exc is owned_deadline:
            return _BaselineCompletion(operation, parent_deadline, admission, bound)
    elif type(exc) is not _ReadyAuthorityDeadline:
        return None
    if bound is None or exc.bound is not bound or exc.operation is not operation:
        return None
    return _BaselineCompletion(operation, parent_deadline, admission, bound)


def permits_baseline_completion(disposition, captured):
    return (
        type(disposition) is _BaselineCompletion
        and disposition.operation is captured.operation
        and disposition.parent_deadline is captured.parent_deadline
        and disposition.admission is captured.admission
    )
