"""Request-bound early readiness/direct work, with post-baseline extended work."""

from concurrent.futures import Future, TimeoutError
from math import isfinite
from threading import Event

from .branch_contracts import (
    DirectBranchFailureReason,
    ExtendedBranchFailureReason,
    SharedBranchFailureReason,
)
from .scheduler_support import (
    CompletedBranch,
    SharedSchedulerFailure,
    failed_branch,
    map_topology_failure,
    shared_outcome,
    timeout_branch,
)
from .scheduler_workers import BRANCH_WORKERS
from .topology.contracts import HybridBranchKind
from .topology.failures import TopologyLoadError


class OverlappingBranches:
    """Own a single request's futures; never wait on the pool from a worker."""

    def __init__(self, scheduler, query, authorization, settings, deadline):
        self.scheduler = scheduler
        self.started = scheduler._clock()
        if type(query) is not str:
            raise TypeError("query must be an exact string")
        self._validate_deadline(deadline, self.started)
        self.budgets = scheduler._budgets(settings)
        self.enabled = scheduler._enabled(settings)
        self.query, self.authorization, self.settings = query, authorization, settings
        self.readiness_deadline = deadline
        self.direct_deadline = min(
            deadline, self.started + self.budgets[HybridBranchKind.DIRECT] / 1000
        )
        self._ready = Future()
        self._closed = Event()
        self._finished = False
        self._extended = None
        self._early = BRANCH_WORKERS.submit(self._prepare_and_direct)

    @staticmethod
    def _validate_deadline(deadline, now):
        if type(deadline) is not float or not isfinite(deadline) or deadline <= now:
            raise ValueError("deadline must be a future finite monotonic float")

    def _prepare_and_direct(self):
        try:
            shared = self.scheduler._shared(
                self.authorization, self.settings, self.readiness_deadline
            )
            if self.scheduler._clock() >= self.readiness_deadline:
                raise SharedSchedulerFailure(SharedBranchFailureReason.OVERALL_DEADLINE)
        except BaseException as error:
            self._ready.set_exception(error)
            raise
        self._ready.set_result(shared)
        if self._closed.is_set() or not self.enabled[HybridBranchKind.DIRECT]:
            return None
        if self.scheduler._clock() >= self.direct_deadline:
            return CompletedBranch(
                timeout_branch(
                    HybridBranchKind.DIRECT,
                    baseline=None,
                    elapsed_ms=self.budgets[HybridBranchKind.DIRECT],
                ),
                self.scheduler._clock(),
            )
        return self.scheduler._direct(
            self.query, shared, self.authorization, self.settings, self.direct_deadline
        )

    def __enter__(self):
        return self

    def __exit__(self, *_error):
        self.close()

    def close(self):
        self._closed.set()
        for future in (self._early, self._extended):
            if future is not None:
                future.cancel()

    def finish(self, *, baseline, deadline):
        if self._finished or self._closed.is_set():
            raise ValueError("request lifecycle has already finished")
        self._finished = True
        now = self.scheduler._clock()
        try:
            self._validate_deadline(deadline, now)
            return self._finish(baseline, deadline, now)
        finally:
            self.close()

    def _finish(self, baseline, deadline, started):
        if self._early is None:
            return shared_outcome(SharedBranchFailureReason.BACKEND_UNAVAILABLE)
        try:
            shared = self._ready.result(
                timeout=max(
                    0.0,
                    min(deadline, self.readiness_deadline) - self.scheduler._clock(),
                )
            )
        except TimeoutError:
            return shared_outcome(SharedBranchFailureReason.OVERALL_DEADLINE)
        except SharedSchedulerFailure as error:
            return shared_outcome(error.reason)
        except TopologyLoadError as error:
            reason = map_topology_failure(HybridBranchKind.DIRECT, error.reason)
            if type(reason) is not SharedBranchFailureReason:
                reason = SharedBranchFailureReason.BACKEND_PROVENANCE_MISMATCH
            return shared_outcome(reason)
        except Exception:
            return shared_outcome(SharedBranchFailureReason.BACKEND_UNAVAILABLE)
        deadlines = {
            HybridBranchKind.DIRECT: self.direct_deadline,
            HybridBranchKind.EXTENDED: min(
                deadline, started + self.budgets[HybridBranchKind.EXTENDED] / 1000
            ),
        }
        results = {
            kind: failed_branch(
                kind,
                DirectBranchFailureReason.DIRECT_NO_SEEDS
                if kind is HybridBranchKind.DIRECT
                else ExtendedBranchFailureReason.EXTENDED_NO_SEEDS,
            )
            for kind in HybridBranchKind
            if not self.enabled[kind]
        }
        futures = {}
        if self.enabled[HybridBranchKind.DIRECT]:
            futures[HybridBranchKind.DIRECT] = self._early
        if self.enabled[HybridBranchKind.EXTENDED]:
            if self.scheduler._clock() >= deadlines[HybridBranchKind.EXTENDED]:
                results[HybridBranchKind.EXTENDED] = timeout_branch(
                    HybridBranchKind.EXTENDED,
                    baseline=baseline,
                    elapsed_ms=self.budgets[HybridBranchKind.EXTENDED],
                )
            else:
                self._extended = BRANCH_WORKERS.submit(
                    self.scheduler._extended,
                    self.query,
                    baseline,
                    shared,
                    self.authorization,
                    self.settings,
                    deadlines[HybridBranchKind.EXTENDED],
                )
                if self._extended is None:
                    return shared_outcome(SharedBranchFailureReason.BACKEND_UNAVAILABLE)
                futures[HybridBranchKind.EXTENDED] = self._extended
        return self.scheduler._collect(
            futures=futures,
            deadlines=deadlines,
            budgets=self.budgets,
            baseline=baseline,
            overall_deadline=deadline,
            results=results,
        )
