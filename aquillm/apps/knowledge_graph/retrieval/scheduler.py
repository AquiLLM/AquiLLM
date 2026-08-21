"""Bounded independent scheduling for direct and extended graph branches."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, TimeoutError, wait
from math import isfinite
from time import monotonic

from .branch_contracts import (
    DirectBranchFailureReason,
    ExtendedBranchFailureReason,
    SharedBranchFailureReason,
)
from .scheduler_support import (
    CompletedBranch,
    HybridBranchRuntime,
    LocalBranchSchedulerFailure,
    SharedSchedulerFailure,
    failed_branch,
    map_topology_failure,
    shared_outcome,
    timeout_branch,
    validate_envelope,
)
from .scheduler_workers import BRANCH_WORKERS
from .topology.contracts import HybridBranchKind
from .topology.failures import TopologyLoadError


class HybridGraphBranchScheduler:
    """Schedule cooperative calls on one process-wide bounded worker pool."""

    def __init__(self, runtime: HybridBranchRuntime, *, clock=monotonic):
        if not isinstance(runtime, HybridBranchRuntime):
            raise TypeError("runtime must implement HybridBranchRuntime")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._runtime = runtime
        self._clock = clock

    def _shared(self, authorization, settings, deadline):
        return self._runtime.prepare_shared(
            authorization=authorization,
            settings=settings,
            deadline=deadline,
        )

    def _direct(self, query, shared, authorization, settings, deadline):
        envelope = self._runtime.run_direct(
            query=query,
            shared=shared,
            authorization=authorization,
            settings=settings,
            deadline=deadline,
        )
        return CompletedBranch(
            validate_envelope(envelope, HybridBranchKind.DIRECT), self._clock()
        )

    def _extended(self, baseline, shared, authorization, settings, deadline):
        prepared = self._runtime.prepare_extended(
            baseline=baseline,
            shared=shared,
            authorization=authorization,
            settings=settings,
            deadline=deadline,
        )
        envelope = self._runtime.run_extended(
            prepared=prepared,
            shared=shared,
            authorization=authorization,
            settings=settings,
            deadline=deadline,
        )
        return CompletedBranch(
            validate_envelope(envelope, HybridBranchKind.EXTENDED), self._clock()
        )

    def run(
        self,
        *,
        query: str,
        baseline: object,
        authorization: object,
        settings: object,
        deadline: float,
    ):
        started = self._clock()
        if type(query) is not str:
            raise TypeError("query must be an exact string")
        if type(deadline) is not float or not isfinite(deadline) or deadline <= started:
            raise ValueError("deadline must be a future finite monotonic float")
        budgets = self._budgets(settings)
        enabled = self._enabled(settings)
        shared_future = BRANCH_WORKERS.submit(
            self._shared, authorization, settings, deadline
        )
        if shared_future is None:
            return shared_outcome(SharedBranchFailureReason.BACKEND_UNAVAILABLE)
        try:
            shared = shared_future.result(timeout=max(0.0, deadline - self._clock()))
        except TimeoutError:
            shared_future.cancel()
            return shared_outcome(SharedBranchFailureReason.OVERALL_DEADLINE)
        except SharedSchedulerFailure as error:
            return shared_outcome(error.reason)
        except TopologyLoadError as error:
            mapped = map_topology_failure(HybridBranchKind.DIRECT, error.reason)
            if type(mapped) is not SharedBranchFailureReason:
                mapped = SharedBranchFailureReason.BACKEND_PROVENANCE_MISMATCH
            return shared_outcome(mapped)
        except Exception:
            return shared_outcome(SharedBranchFailureReason.BACKEND_UNAVAILABLE)
        if self._clock() >= deadline:
            return shared_outcome(SharedBranchFailureReason.OVERALL_DEADLINE)
        deadlines = {
            kind: min(deadline, started + budget / 1000.0)
            for kind, budget in budgets.items()
        }
        requests = {
            HybridBranchKind.DIRECT: (
                self._direct,
                (
                    query,
                    shared,
                    authorization,
                    settings,
                    deadlines[HybridBranchKind.DIRECT],
                ),
            ),
            HybridBranchKind.EXTENDED: (
                self._extended,
                (
                    baseline,
                    shared,
                    authorization,
                    settings,
                    deadlines[HybridBranchKind.EXTENDED],
                ),
            ),
        }
        enabled_kinds = tuple(kind for kind in HybridBranchKind if enabled[kind])
        submitted = BRANCH_WORKERS.submit_batch(
            tuple(requests[kind] for kind in enabled_kinds)
        )
        if submitted is None:
            return shared_outcome(SharedBranchFailureReason.BACKEND_UNAVAILABLE)
        futures = dict(zip(enabled_kinds, submitted, strict=True))
        results = {
            kind: failed_branch(
                kind,
                DirectBranchFailureReason.DIRECT_NO_SEEDS
                if kind is HybridBranchKind.DIRECT
                else ExtendedBranchFailureReason.EXTENDED_NO_SEEDS,
            )
            for kind in HybridBranchKind
            if not enabled[kind]
        }
        try:
            return self._collect(
                futures=futures,
                deadlines=deadlines,
                budgets=budgets,
                baseline=baseline,
                overall_deadline=deadline,
                results=results,
            )
        finally:
            for future in futures.values():
                future.cancel()

    @staticmethod
    def _enabled(settings: object) -> dict[HybridBranchKind, bool]:
        flag_names = ("graph_direct_enabled", "graph_extended_enabled")
        if not any(hasattr(settings, name) for name in flag_names):
            return dict.fromkeys(HybridBranchKind, True)
        values = {
            HybridBranchKind.DIRECT: getattr(settings, "graph_direct_enabled", None),
            HybridBranchKind.EXTENDED: getattr(
                settings, "graph_extended_enabled", None
            ),
        }
        if any(type(value) is not bool for value in values.values()) or not any(
            values.values()
        ):
            raise ValueError("at least one branch must be explicitly enabled")
        return values  # type: ignore[return-value]

    @staticmethod
    def _budgets(settings: object) -> dict[HybridBranchKind, int]:
        values = {
            HybridBranchKind.DIRECT: getattr(settings, "graph_direct_timeout_ms", None),
            HybridBranchKind.EXTENDED: getattr(
                settings, "graph_extended_timeout_ms", None
            ),
        }
        if any(
            type(value) is not int or not 1 <= value <= 5000
            for value in values.values()
        ):
            raise ValueError("branch timeouts must be exact integers in [1, 5000]")
        return values  # type: ignore[return-value]

    def _collect(
        self,
        *,
        futures,
        deadlines,
        budgets,
        baseline,
        overall_deadline,
        results,
    ):
        while len(results) < 2:
            shared = self._read_completed(futures, deadlines, results)
            if shared is not None:
                return shared_outcome(shared)
            now = self._clock()
            overall_expired = now >= overall_deadline
            if overall_expired and not results:
                return shared_outcome(SharedBranchFailureReason.OVERALL_DEADLINE)
            for kind in HybridBranchKind:
                if kind not in results and (overall_expired or now >= deadlines[kind]):
                    results[kind] = timeout_branch(
                        kind, baseline=baseline, elapsed_ms=budgets[kind]
                    )
                    futures[kind].cancel()
            if len(results) == 2:
                break
            pending = [
                futures[kind] for kind in HybridBranchKind if kind not in results
            ]
            wake_at = min(
                overall_deadline,
                *(deadlines[kind] for kind in HybridBranchKind if kind not in results),
            )
            wait(
                pending,
                timeout=max(0.0, wake_at - self._clock()),
                return_when=FIRST_COMPLETED,
            )
        return self._outcome(results)

    @staticmethod
    def _read_completed(futures, deadlines, results):
        for kind in HybridBranchKind:
            if kind in results:
                continue
            future: Future = futures[kind]
            if not future.done():
                continue
            try:
                completed = future.result()
            except SharedSchedulerFailure as error:
                return error.reason
            except TopologyLoadError as error:
                mapped = map_topology_failure(kind, error.reason)
                if type(mapped) is SharedBranchFailureReason:
                    return mapped
                results[kind] = failed_branch(kind, mapped, seed_count=1)
                continue
            except LocalBranchSchedulerFailure as error:
                if error.kind is not kind:
                    return SharedBranchFailureReason.BACKEND_PROVENANCE_MISMATCH
                results[kind] = failed_branch(kind, error.reason)
                continue
            except Exception:
                return SharedBranchFailureReason.BACKEND_UNAVAILABLE
            if completed.completed_at > deadlines[kind]:
                continue
            envelope = completed.envelope
            if type(envelope.failure_reason) is SharedBranchFailureReason:
                return envelope.failure_reason
            results[kind] = envelope
        return None

    @staticmethod
    def _outcome(results):
        from .branch_contracts import HybridBranchOutcomeV1

        return HybridBranchOutcomeV1(
            results[HybridBranchKind.DIRECT],
            results[HybridBranchKind.EXTENDED],
            None,
        )


def run_hybrid_graph_branches(*, runtime: HybridBranchRuntime, **request):
    """Functional adapter for callers that do not retain a scheduler instance."""

    return HybridGraphBranchScheduler(runtime).run(**request)


__all__ = [
    "HybridBranchRuntime",
    "HybridGraphBranchScheduler",
    "LocalBranchSchedulerFailure",
    "SharedSchedulerFailure",
    "run_hybrid_graph_branches",
]
