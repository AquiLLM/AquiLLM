"""Single-use native graph work; recovery remains outside this owner."""

from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from typing import Literal

from apps.documents.services.chunk_search_rank_policy import controls
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.source_deadline import _DEADLINE
from lib.retrieval.operation import _OPERATION, capture_operation_bound


@dataclass(frozen=True)
class _ClosedGraphAttemptResult:
    kind: Literal["success", "failure", "invalid"]
    pool: tuple | None
    diagnostics: dict
    admit_deferred: bool = False


class _ClosedGraphLimited(TimeoutError):
    """Only a known graph limit and detached, validated observations escape."""

    def __init__(self, reason, *, coverage=None, invocation=None):
        if type(reason) is not str or reason not in {"deadline", "frontier_budget"}:
            raise ValueError("unsupported closed graph limit")
        super().__init__(reason)
        self.reason = reason
        self.observations = (
            {}
            if coverage is None and invocation is None
            else _graph_observations(coverage, invocation)
        )


def _graph_observations(coverage, invocation):
    from ._graph_v2_validation import checksum, count
    from .graph_frontier_observation_v2 import (
        FrontierBudgetObservationV2,
        GraphFrontierInvocationV2,
    )
    from .graph_handoff import GraphCoverageV2

    if (type(coverage) is not GraphCoverageV2
            or type(invocation) is not GraphFrontierInvocationV2):
        raise TypeError("graph observations require exact captured DTOs")
    coverage.__post_init__()
    frontier = invocation.frontier_budget
    if type(frontier) is not FrontierBudgetObservationV2:
        raise TypeError("graph observations require the exact frontier DTO")
    for name in ("query_sha256", "requested_subset_signature",
                 "request_algorithm_signature"):
        checksum(getattr(invocation, name), name)
    identity = invocation.invocation_id
    if (type(identity) is not str or len(identity) != 32
            or any(c not in "0123456789abcdef" for c in identity)):
        raise ValueError("invalid frontier invocation identity")
    maxima = dict(max_queries=64, max_rows=5000, max_bytes=1048576,
                  max_identities=200, max_edges=1000, max_depth=2)
    for field in fields(frontier):
        name, value = field.name, getattr(frontier, field.name)
        if name == "policy_signature":
            checksum(value, name)
        elif name in {"accounting_complete", "uncertain"}:
            if type(value) is not bool:
                raise TypeError("frontier accounting flags must be exact booleans")
        else:
            # Physical charges may exceed admission caps; retain actual debits.
            count(value, name, maxima.get(name, 9_223_372_036_854_775_807))
    return {"graph_coverage_v2": asdict(coverage),
            "frontier_budget": asdict(frontier),
            "frontier_invocations": [asdict(invocation)]}


def _capture_closed_graph_attempt(
    snapshot, query, authorization, dependencies, *, diagnostics
):
    # Keep the existing owner's import/patch targets for native callers and tests.
    from . import graph_candidate_operation_v2 as owner

    scope, graph, settings = (
        owner.current_scope(),
        dependencies.runtime,
        dependencies.settings,
    )
    if (
        scope is None
        or type(graph) is not owner.ProductionHybridBranchRuntime
        or graph.settings is not settings
        or getattr(graph.run_v2, "__func__", None) is not owner._NATIVE_GRAPH_RUN_V2
        or getattr(settings, "graph_remediation_mode", None) != "v2"
    ):
        return None
    base = owner.TextChunk.objects.using(scope.using).all()
    header = owner._closed_header(base, scope)
    if header is None:
        return None
    return _ClosedGraphAttempt(
        owner,
        snapshot,
        query,
        authorization,
        dependencies,
        diagnostics,
        owner._GraphSourceStage(scope, base, header),
    )


class _ClosedGraphAttempt:
    def __init__(
        self, owner, snapshot, query, authorization, dependencies, diagnostics, stage
    ):
        self._owner, self._stage = owner, stage
        self._snapshot, self._query, self._authorization = (
            snapshot,
            query,
            authorization,
        )
        self._dependencies, self._diagnostics = dependencies, diagnostics
        self._graph, self._settings = dependencies.runtime, dependencies.settings
        self._admission = getattr(snapshot, "source_admission", None)
        self._dependencies_header = (
            dependencies.runtime,
            dependencies.settings,
            dependencies.materialize,
        )
        self._graph_header = (
            self._graph.authorization,
            self._graph.settings,
            self._graph.codec,
            self._graph.clock,
        )
        self._settings_before = deepcopy(owner._settings_values(self._settings))
        self._ranking_before = controls()
        self._operation, self._parent_deadline = _OPERATION.get(), _DEADLINE.get()
        self._consumed = False

    def _check(self):
        owner, stage = self._owner, self._stage
        parent, operation = self._parent_deadline, self._operation
        if parent is not None and (
            parent[1]() >= parent[0]
            or (parent[2] is not None and not parent[2].can_publish())
        ):
            raise SourceScopeUnavailable("graph_original_scope_unavailable")
        if operation is not None and not operation.active():
            raise SourceScopeUnavailable("graph_original_scope_unavailable")
        stage.check()
        if getattr(self._snapshot, "source_admission", None) is not self._admission:
            raise SourceScopeUnavailable("graph_original_scope_unavailable")
        owner.captured_admission(self._snapshot, stage.scope, query=self._query)
        dependencies, graph = self._dependencies, self._graph
        if (
            not owner._same_references(
                self._dependencies_header,
                (dependencies.runtime, dependencies.settings, dependencies.materialize),
            )
            or getattr(graph.run_v2, "__func__", None) is not owner._NATIVE_GRAPH_RUN_V2
            or not owner._same_references(
                self._graph_header,
                (graph.authorization, graph.settings, graph.codec, graph.clock),
            )
            or owner._settings_values(self._settings) != self._settings_before
            or controls() != self._ranking_before
            or _OPERATION.get() is not operation
        ):
            raise SourceScopeUnavailable("graph_original_scope_unavailable")

    def run(self, *, downstream_reserve_ms):
        return self._consume(downstream_reserve_ms, native_closure=False)

    def run_native_closure(self, *, closure_reserve_ms):
        return self._consume(closure_reserve_ms, native_closure=True)

    def _consume(self, reserve_ms, *, native_closure):
        if self._consumed:
            raise SourceScopeUnavailable("graph_attempt_consumed")
        self._consumed = True
        if type(reserve_ms) is not int or reserve_ms < 0:
            self._stage.withdraw()
            raise ValueError("reserve must be a nonnegative exact integer")
        token = _DEADLINE.set(self._parent_deadline) if native_closure else None
        try:
            return self._run(reserve_ms, native_closure=native_closure)
        finally:
            if token is not None:
                _DEADLINE.reset(token)

    def _run(self, reserve_ms, *, native_closure):
        owner, stage, graph = self._owner, self._stage, self._graph
        deadline = None
        try:
            bound = capture_operation_bound(stage.budget)
            timeout = self._settings.graph_overall_timeout_ms
            if type(timeout) is not int or not 1 <= timeout <= 20_000:
                raise ValueError("graph timeout is invalid")
            clock = graph.clock
            if native_closure:
                # Fresh requested authorization is bounded by the captured original
                # source context. Its cleanup cost is paid before partitioning time.
                self._check()
                with stage.header[0][5]:
                    self._check()
                    with owner.bounded_source_database(
                        stage.runtime, stage.scope.using
                    ):
                        self._check()
                        owner.validate_requested_authorization(
                            stage.runtime,
                            self._authorization,
                            self._snapshot.documents,
                        )
                        self._check()
                    self._check()
                self._check()
                now = clock()
                effective_ms = owner.source_remaining_ms(stage.budget.remaining_ms())
                if bound is not None:
                    effective_ms = min(effective_ms, bound.remaining_ms())
                deadline = now + min(timeout, max(0, effective_ms - reserve_ms)) / 1000
            else:
                # Compatibility formula: reserve only from the original action.
                now = clock()
                deadline = min(
                    now + timeout / 1000,
                    now + owner.source_remaining_ms(stage.budget.remaining_ms()) / 1000,
                )
                if bound is not None:
                    deadline = min(
                        deadline, bound.deadline_for(clock, reserve_ms=reserve_ms)
                    )

            def phase_check():
                if clock() >= deadline:
                    raise _ClosedGraphLimited("deadline")
                self._check()

            parent = self._parent_deadline
            preparation_clock = parent[1] if parent is not None else clock
            preparation_now = preparation_clock()
            preparation_deadline = preparation_now + max(0, deadline - clock())
            phase_check()
            with stage.header[0][5]:
                phase_check()
                with owner.source_preparation_scope(
                    preparation_deadline, preparation_clock, stage.budget
                ):
                    phase_check()
                    with owner.bounded_source_database(
                        stage.runtime, stage.scope.using
                    ):
                        phase_check()
                        result, final_rows, selected_pks = owner._run_closed_graph(
                            stage,
                            self._snapshot,
                            self._query,
                            self._authorization,
                            graph,
                            self._settings,
                            deadline,
                            phase_check,
                            self._diagnostics,
                            self._admission,
                            native_closure=native_closure,
                        )
                    phase_check()  # Classify DB cleanup expiry before child scope exit.
                # Preparation and DB cleanup precede publication under the same lock.
                phase_check()
                stage.publish(final_rows, phase_check)
                owner.observe_published_graph(selected_pks)
                return _ClosedGraphAttemptResult("success", result[0], result[1])
        except Exception as exc:
            # Never recover through a replacement or expired original authority.
            try:
                self._check()
            except Exception:
                return _ClosedGraphAttemptResult(
                    "invalid", None, self._diagnostics("error")
                )
            stage.withdraw()
            observations = self._diagnostics("error")
            observations.update(owner._limit_observations(exc))
            return _ClosedGraphAttemptResult(
                "failure",
                None,
                observations,
                admit_deferred=deadline is None or clock() < deadline,
            )
        finally:
            stage.withdraw()
