# ruff: noqa: E501
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from math import isfinite
from time import monotonic
from typing import Protocol, runtime_checkable

from .branch_contracts import (
    BranchEnvelopeV1,
    BranchSafeDiagnosticsV1,
    BranchStatusV1,
    DirectBranchFailureReason,
    ExtendedBranchFailureReason,
    HybridBranchOutcomeV1,
    SharedBranchFailureReason,
)
from .topology.contracts import HybridBranchKind


class SharedSchedulerFailure(RuntimeError):
    """A fixed safe failure that invalidates both graph branches."""

    def __init__(self, reason: SharedBranchFailureReason):
        if type(reason) is not SharedBranchFailureReason:
            raise TypeError("reason must be an exact shared branch failure")
        self.reason = reason
        super().__init__(reason.value)


@runtime_checkable
class HybridBranchRuntime(Protocol):
    """Provider adapter whose every operation receives an absolute deadline."""

    # fmt: off
    def prepare_shared(self, *, authorization: object, settings: object, deadline: float) -> object: ...
    def run_direct(self, *, query: str, shared: object, authorization: object, settings: object, deadline: float) -> BranchEnvelopeV1: ...
    def prepare_extended(self, *, baseline: object, shared: object, authorization: object, settings: object, deadline: float) -> object: ...
    def run_extended(self, *, prepared: object, shared: object, authorization: object, settings: object, deadline: float) -> BranchEnvelopeV1: ...
    # fmt: on


@dataclass(frozen=True, slots=True)
class _Completed:
    envelope: BranchEnvelopeV1
    completed_at: float


def _failed(
    kind: HybridBranchKind,
    reason: SharedBranchFailureReason
    | DirectBranchFailureReason
    | ExtendedBranchFailureReason,
    *,
    seed_count: int = 0,
    elapsed_ms: int = 0,
) -> BranchEnvelopeV1:
    return BranchEnvelopeV1(
        kind,
        BranchStatusV1.FAILED,
        None,
        reason,
        BranchSafeDiagnosticsV1(seed_count, 0, 0, 0, elapsed_ms),
    )


def _shared(reason: SharedBranchFailureReason) -> HybridBranchOutcomeV1:
    return HybridBranchOutcomeV1(
        _failed(HybridBranchKind.DIRECT, reason),
        _failed(HybridBranchKind.EXTENDED, reason),
        reason,
    )


def _timeout(
    kind: HybridBranchKind, *, baseline: object, elapsed_ms: int
) -> BranchEnvelopeV1:
    if kind is HybridBranchKind.DIRECT:
        return _failed(
            kind,
            DirectBranchFailureReason.EXTRACTOR_TIMEOUT,
            elapsed_ms=elapsed_ms,
        )
    raw_seeds = getattr(baseline, "graph_seeds", ())
    seed_count = min(len(raw_seeds), 64) if type(raw_seeds) is tuple else 0
    if not seed_count:
        return _failed(
            kind,
            ExtendedBranchFailureReason.EXTENDED_NO_SEEDS,
            elapsed_ms=elapsed_ms,
        )
    return _failed(
        kind,
        ExtendedBranchFailureReason.EXTENDED_TOPOLOGY_TIMEOUT,
        seed_count=seed_count,
        elapsed_ms=elapsed_ms,
    )


def _validate_envelope(
    envelope: object, expected: HybridBranchKind
) -> BranchEnvelopeV1:
    if type(envelope) is not BranchEnvelopeV1 or envelope.branch_kind is not expected:
        raise SharedSchedulerFailure(
            SharedBranchFailureReason.BACKEND_PROVENANCE_MISMATCH
        )
    return envelope


class HybridGraphBranchScheduler:
    """Run direct and baseline-derived extended work under separate budgets."""

    def __init__(self, runtime: HybridBranchRuntime, *, clock=monotonic):
        if not isinstance(runtime, HybridBranchRuntime):
            raise TypeError("runtime must implement HybridBranchRuntime")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._runtime = runtime
        self._clock = clock

    def _direct(self, query, shared, authorization, settings, deadline) -> _Completed:
        envelope = self._runtime.run_direct(
            query=query,
            shared=shared,
            authorization=authorization,
            settings=settings,
            deadline=deadline,
        )
        return _Completed(
            _validate_envelope(envelope, HybridBranchKind.DIRECT), self._clock()
        )

    def _extended(
        self, baseline, shared, authorization, settings, deadline
    ) -> _Completed:
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
        return _Completed(
            _validate_envelope(envelope, HybridBranchKind.EXTENDED), self._clock()
        )

    def run(
        self,
        *,
        query: str,
        baseline: object,
        authorization: object,
        settings: object,
        deadline: float,
    ) -> HybridBranchOutcomeV1:
        started = self._clock()
        if type(query) is not str:
            raise TypeError("query must be an exact string")
        if type(deadline) is not float or not isfinite(deadline) or deadline <= started:
            raise ValueError("deadline must be a future finite monotonic float")
        budgets = self._budgets(settings)
        try:
            shared = self._runtime.prepare_shared(
                authorization=authorization,
                settings=settings,
                deadline=deadline,
            )
        except SharedSchedulerFailure as error:
            return _shared(error.reason)
        except Exception:
            return _shared(SharedBranchFailureReason.BACKEND_UNAVAILABLE)
        if self._clock() >= deadline:
            return _shared(SharedBranchFailureReason.OVERALL_DEADLINE)
        deadlines = {
            kind: min(deadline, started + budget / 1000.0)
            for kind, budget in budgets.items()
        }
        executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="kg-branch")
        futures: dict[HybridBranchKind, Future[_Completed]] = {
            HybridBranchKind.DIRECT: executor.submit(
                self._direct,
                query,
                shared,
                authorization,
                settings,
                deadlines[HybridBranchKind.DIRECT],
            ),
            HybridBranchKind.EXTENDED: executor.submit(
                self._extended,
                baseline,
                shared,
                authorization,
                settings,
                deadlines[HybridBranchKind.EXTENDED],
            ),
        }
        try:
            return self._collect(
                futures=futures,
                deadlines=deadlines,
                budgets=budgets,
                baseline=baseline,
                overall_deadline=deadline,
            )
        finally:
            for future in futures.values():
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)

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

    def _collect(self, *, futures, deadlines, budgets, baseline, overall_deadline):
        results: dict[HybridBranchKind, BranchEnvelopeV1] = {}
        while len(results) < 2:
            shared = self._read_completed(futures, deadlines, results)
            if shared is not None:
                return _shared(shared)
            now = self._clock()
            if now >= overall_deadline:
                return _shared(SharedBranchFailureReason.OVERALL_DEADLINE)
            for kind in HybridBranchKind:
                if kind not in results and now >= deadlines[kind]:
                    results[kind] = _timeout(
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
        return HybridBranchOutcomeV1(
            results[HybridBranchKind.DIRECT],
            results[HybridBranchKind.EXTENDED],
            None,
        )

    @staticmethod
    def _read_completed(futures, deadlines, results):
        for kind in HybridBranchKind:
            future = futures[kind]
            if kind in results or not future.done():
                continue
            try:
                completed = future.result()
            except SharedSchedulerFailure as error:
                return error.reason
            except Exception:
                return SharedBranchFailureReason.BACKEND_UNAVAILABLE
            if completed.completed_at > deadlines[kind]:
                continue
            envelope = completed.envelope
            if type(envelope.failure_reason) is SharedBranchFailureReason:
                return envelope.failure_reason
            results[kind] = envelope
        return None


def run_hybrid_graph_branches(*, runtime: HybridBranchRuntime, **request):
    """Functional adapter for callers that do not retain a scheduler instance."""

    return HybridGraphBranchScheduler(runtime).run(**request)


__all__ = [
    "HybridBranchRuntime",
    "HybridGraphBranchScheduler",
    "SharedSchedulerFailure",
    "run_hybrid_graph_branches",
]
